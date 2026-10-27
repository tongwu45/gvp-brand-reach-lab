import json
from dataclasses import replace

import numpy as np
import pytest

from gvplab import CampaignConfig, MarketConfig, build_world, lift_study, private_reach_report, run_campaign, summarize
from gvplab.auction import shading_table
from gvplab.brand_safety import (HybridClassifier, KeywordClassifier, LLMClassifier, NoFilter, OracleClassifier,
                                 evaluate, parse_verdict)
from gvplab.config import HIGH, MEDIUM, SAFE, TOLERANCE


@pytest.fixture(scope="module")
def world():
    return build_world(MarketConfig(n_users=8_000), seed=3)


# ---------------------------------------------------------------- brand safety
def test_keyword_blocklist_is_fooled_by_context():
    kw = KeywordClassifier()
    assert kw.predict_title("Portrait photography: shooting in golden hour light") >= MEDIUM  # false positive
    assert kw.predict_title("Why those people should not be allowed in our country") == SAFE  # false negative
    assert kw.predict_title("Uncensored nude scenes compilation") == HIGH


def test_oracle_is_perfect(world):
    pred = OracleClassifier().predict(world.placements)
    m = evaluate(pred, world.placements, "standard")
    assert m["over_block"] == 0 and m["leak"] == 0 and m["tier_accuracy"] == 1


def test_parse_verdict_handles_wrapped_json():
    v = parse_verdict('Sure! {"tier": 2, "category": "news", "rationale": "violent protest coverage"}')
    assert v.tier == 2
    with pytest.raises(Exception):
        parse_verdict('{"tier": 9, "category": "x", "rationale": "y"}')  # out of range


def test_llm_classifier_caches_and_fails_closed(tmp_path):
    calls = []

    def fake(prompt):
        calls.append(prompt)
        if 'Video title: "DIY bath bomb' in prompt:
            return json.dumps({"tier": 0, "category": "diy", "rationale": "cosmetic product"})
        return "not json"

    llm = LLMClassifier(fake, cache_path=tmp_path / "cache.json", retries=1)
    assert llm.classify("DIY bath bomb tutorial").tier == SAFE
    assert llm.classify("DIY bath bomb tutorial").tier == SAFE      # served from cache
    assert llm.classify("garbled").tier == HIGH                     # fail closed
    assert len(calls) == 1 + 2                                      # 1 success, 2 attempts for failure
    llm.save()
    assert LLMClassifier(fake, cache_path=tmp_path / "cache.json").cache  # cache persisted


def test_hybrid_routes_only_risky_slice(world):
    oracle_lookup = dict(zip(world.placements.title, world.placements.true_tier))

    def fake(prompt):
        title = prompt.split('Video title: "')[1].split('"')[0]
        return json.dumps({"tier": int(oracle_lookup[title]), "category": "x", "rationale": "y"})

    hybrid = HybridClassifier(LLMClassifier(fake))
    pred = hybrid.predict(world.placements)
    assert 0 < hybrid.routed_share < 1
    kw = evaluate(KeywordClassifier().predict(world.placements), world.placements)
    hy = evaluate(pred, world.placements)
    assert hy["over_block"] < kw["over_block"] and hy["leak"] < kw["leak"]


# ---------------------------------------------------------------- auction
def test_shading_table_is_sane():
    rng = np.random.default_rng(0)
    comp = 5 * np.exp(0.6 * rng.standard_normal(10_000))
    v = np.linspace(0, 20, 50)
    b = shading_table(comp, floor=1.0, v_grid=v)
    assert np.all(b <= v + 1e-9)                      # never bid above value
    assert np.all(np.diff(b[v > 2]) >= -0.05)         # (weakly) increasing in value
    assert np.all((b[v > 2] >= 1.0))                  # bids clear the floor when worth it


@pytest.mark.parametrize("auction,bidding", [("second_price", "truthful"), ("first_price", "shaded")])
def test_campaign_respects_constraints(world, auction, bidding):
    cfg = CampaignConfig(budget=40, frequency_cap=3, auction=auction, bidding=bidding)
    pred = KeywordClassifier().predict(world.placements)
    d = run_campaign(world, cfg, pred, seed=1)
    imp = d.impressions
    assert imp.price_cpm.sum() / 1000 <= cfg.budget + 1e-9
    assert imp.groupby("user_id").size().max() <= cfg.frequency_cap
    assert (pred[imp.placement_id] <= TOLERANCE[cfg.suitability]).all()   # blocked inventory never served
    assert not np.isin(imp.user_id, d.holdout_users).any()                 # holdout never served
    assert (imp.price_cpm >= world.placements.floor_cpm.values[imp.placement_id] - 1e-9).all()


def test_gvp_adds_reach_at_equal_budget(world):
    pred = OracleClassifier().predict(world.placements)
    base = CampaignConfig(budget=40)
    yt = summarize(run_campaign(world, replace(base, networks=("youtube",)), pred), world)
    both = summarize(run_campaign(world, base, pred), world)
    assert both["reach"] > yt["reach"]
    assert both["unsafe_impression_rate"] == 0


def test_no_filter_serves_unsafe_inventory(world):
    d = run_campaign(world, CampaignConfig(budget=40), NoFilter().predict(world.placements))
    assert summarize(d, world)["unsafe_impression_rate"] > 0


# ---------------------------------------------------------------- measurement
def test_private_report(world):
    d = run_campaign(world, CampaignConfig(budget=40), OracleClassifier().predict(world.placements))
    cells, s = private_reach_report(d, world, epsilon=1e6, k=0)
    assert s["total_reach_error"] < 1e-3                  # negligible noise -> exact
    assert cells.true_count.sum() == summarize(d, world)["reach"]
    _, strict = private_reach_report(d, world, epsilon=1.0, k=10_000)
    assert strict["suppressed_reach_share"] == 1.0          # everything below threshold


def test_lift_study_outputs(world):
    d = run_campaign(world, CampaignConfig(budget=40), OracleClassifier().predict(world.placements))
    r = lift_study(d, world, survey_share=1.0)
    assert r["ci_low"] <= r["lift_estimate"] <= r["ci_high"]
    assert r["true_lift"] > 0
