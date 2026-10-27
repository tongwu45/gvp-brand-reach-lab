"""Run the full experiment suite and write results/ and docs/ charts.

    python -m gvplab.experiments --seeds 5
    python -m gvplab.experiments --llm gemini        # also evaluate an LLM classifier

Every scenario within a seed shares the same simulated world, and results are
averaged over seeds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .auction import run_campaign
from .brand_safety import (TRANSPORTS, HybridClassifier, KeywordClassifier, LLMClassifier, NoFilter,
                           OracleClassifier, evaluate)
from .config import TOLERANCE, CampaignConfig
from .inventory import build_world
from .measurement import lift_study, private_reach_report, summarize

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

KEY_METRICS = ["reach_pct", "in_target_reach", "vcpm", "cost_per_1k_reached", "avg_frequency",
               "unsafe_impression_rate", "gvp_only_reach", "gvp_share_of_spend", "gvp_publisher_revenue",
               "advertiser_surplus", "blocked_share"]


def _agg(df: pd.DataFrame, by: list[str] | str) -> pd.DataFrame:
    num = df.select_dtypes("number").columns.difference(["seed"])
    return df.groupby(by, sort=False)[list(num)].agg(["mean", "std"])


def channel_scenarios(world, seed: int, base: CampaignConfig) -> list[dict]:
    pl = world.placements
    kw, orc = KeywordClassifier().predict(pl), OracleClassifier().predict(pl)
    runs = {
        "youtube_only": (replace(base, networks=("youtube",)), orc),
        "youtube+gvp, no filter": (base, NoFilter().predict(pl)),
        "youtube+gvp, keyword blocklist": (base, kw),
        "youtube+gvp, context-aware (oracle)": (base, orc),
    }
    rows, deliveries = [], {}
    for name, (cfg, pred) in runs.items():
        d = run_campaign(world, cfg, pred, seed=seed)
        deliveries[name] = d
        rows.append({"seed": seed, "scenario": name, **summarize(d, world)})

    # Publisher-side cost of over-blocking: revenue that genuinely suitable GVP
    # placements earn with accurate labels but lose under the keyword list.
    tol = TOLERANCE[base.suitability]
    wrongly_blocked = (pl.true_tier.values <= tol) & (kw > tol)
    imp = deliveries["youtube+gvp, context-aware (oracle)"].impressions
    gvp = imp.network == "gvp"
    lost = imp.price_cpm[gvp & wrongly_blocked[imp.placement_id.values]].sum() / 1000
    rows[2]["suitable_publisher_revenue_lost"] = lost
    rows[2]["suitable_publisher_revenue_lost_share"] = lost / max(imp.price_cpm[gvp].sum() / 1000, 1e-9)
    return rows, deliveries


def auction_scenarios(world, seed: int, base: CampaignConfig) -> list[dict]:
    orc = OracleClassifier().predict(world.placements)
    rows = []
    for auction, bidding in [("second_price", "truthful"), ("first_price", "truthful"), ("first_price", "shaded")]:
        d = run_campaign(world, replace(base, auction=auction, bidding=bidding), orc, seed=seed)
        rows.append({"seed": seed, "auction": f"{auction} / {bidding}", **summarize(d, world)})
    return rows


def budget_curve(world, seed: int, base: CampaignConfig, budgets=(100, 200, 400, 800, 1600, 3200)) -> list[dict]:
    orc = OracleClassifier().predict(world.placements)
    rows = []
    for b in budgets:
        for label, nets in (("YouTube only", ("youtube",)), ("YouTube + GVP", ("youtube", "gvp"))):
            d = run_campaign(world, replace(base, budget=b, networks=nets), orc, seed=seed)
            s = summarize(d, world)
            rows.append({"seed": seed, "budget": b, "plan": label, "reach_pct": s["reach_pct"], "vcpm": s["vcpm"]})
    return rows


def classifier_eval(world, seed: int) -> list[dict]:
    pl = world.placements
    rows = []
    for clf in (KeywordClassifier(), OracleClassifier()):
        pred = clf.predict(pl)
        for suit in TOLERANCE:
            rows.append({"seed": seed, "classifier": clf.name, "suitability": suit, **evaluate(pred, pl, suit)})
    return rows


def privacy_grid(delivery, world, seed: int) -> list[dict]:
    rows = []
    for dims in (("age", "region"), ("dma", "age")):
        for eps in (0.1, 0.5, 1.0, 2.0):
            for k in (0, 10, 50):
                _, s = private_reach_report(delivery, world, epsilon=eps, k=k, dims=dims, seed=seed)
                rows.append({"seed": seed, **s})
    return rows


def llm_eval(provider: str, model: str | None, limit: int, world) -> pd.DataFrame:
    transport = TRANSPORTS[provider](model) if model else TRANSPORTS[provider]()
    llm = LLMClassifier(transport, cache_path=RESULTS / f"llm_cache_{provider}.json")
    pl = world.placements.drop_duplicates("title").sample(min(limit, world.placements.title.nunique()),
                                                          random_state=0).reset_index(drop=True)
    rows = []
    for name, clf in (("keyword_blocklist", KeywordClassifier()), ("llm", llm), ("hybrid", HybridClassifier(llm))):
        pred = clf.predict(pl)
        for suit in TOLERANCE:
            rows.append({"classifier": name, "suitability": suit, **evaluate(pred, pl, suit)})
    out = pd.DataFrame(rows)
    out.attrs["llm_calls"], out.attrs["llm_failures"] = llm.calls, llm.failures
    return out


def charts(curve: pd.DataFrame, privacy: pd.DataFrame, auction: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DOCS.mkdir(exist_ok=True)
    c = curve.groupby(["plan", "budget"]).reach_pct.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for col, color in zip(c.columns, ("#1a73e8", "#ea4335")):
        ax.plot(c.index, 100 * c[col], marker="o", label=col, color=color)
    ax.set_xscale("log", base=2)
    ax.set_xticks(c.index, [f"${b:,}" for b in c.index])
    ax.minorticks_off()
    ax.set_xlabel("Flight budget (log scale)")
    ax.set_ylabel("Unique viewable reach (% of users)")
    ax.set_title("Same budget, more people: reach curve")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DOCS / "reach_curve.png", dpi=160)
    plt.close(fig)

    p = privacy[privacy.granularity == "dma x age"].groupby(["k", "epsilon"]).total_reach_error.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k in p.columns:
        ax.plot(p.index, 100 * p[k], marker="o", label=f"k-threshold = {k}")
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget epsilon (log scale; lower = more private)")
    ax.set_ylabel("Total reach error (%)")
    ax.set_title("Privacy vs accuracy, DMA x age report (900 cells)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DOCS / "privacy_tradeoff.png", dpi=160)
    plt.close(fig)

    a = auction.groupby("auction", sort=False)[["reach_pct", "advertiser_surplus", "gvp_publisher_revenue"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, col, title in zip(axes, a.columns, ("Reach (% users)", "Advertiser surplus ($)", "GVP publisher revenue ($)")):
        vals = a[col] * (100 if col == "reach_pct" else 1)
        ax.bar(range(len(a)), vals, color=["#1a73e8", "#9aa0a6", "#34a853"])
        ax.set_xticks(range(len(a)), [s.replace(" / ", "\n") for s in a.index], fontsize=7)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Auction format and bidding strategy, equal budget", fontsize=10)
    fig.tight_layout()
    fig.savefig(DOCS / "auction_comparison.png", dpi=160)
    plt.close(fig)


def fmt(df: pd.DataFrame, cols: list[str]) -> str:
    """Render 'mean ± std' columns as a markdown table."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        m, s = df[(c, "mean")], df[(c, "std")]
        if c.endswith(("_pct", "rate", "share", "error", "accuracy", "over_block", "leak")) or c in ("viewability",):
            out[c] = [f"{100 * a:.1f}% ± {100 * b:.1f}" for a, b in zip(m, s.fillna(0))]
        elif m.abs().max() >= 100:
            out[c] = [f"{a:,.0f} ± {b:,.0f}" for a, b in zip(m, s.fillna(0))]
        else:
            out[c] = [f"{a:.2f} ± {b:.2f}" for a, b in zip(m, s.fillna(0))]
    return out.to_markdown()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--llm", choices=sorted(TRANSPORTS), help="also evaluate an LLM classifier (needs API key)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--llm-limit", type=int, default=150, help="unique titles to send to the LLM")
    args = ap.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    base = CampaignConfig()
    chan, auc, curve, clf, priv, lift = [], [], [], [], [], []
    for seed in range(args.seeds):
        world = build_world(seed=seed)
        rows, deliveries = channel_scenarios(world, seed, base)
        chan += rows
        auc += auction_scenarios(world, seed, base)
        curve += budget_curve(world, seed, base)
        clf += classifier_eval(world, seed)
        main_delivery = deliveries["youtube+gvp, context-aware (oracle)"]
        priv += privacy_grid(main_delivery, world, seed)
        lift.append({"seed": seed, **lift_study(main_delivery, world, seed=seed)})
        print(f"seed {seed} done")

    chan, auc, curve = pd.DataFrame(chan), pd.DataFrame(auc), pd.DataFrame(curve)
    clf, priv, lift = pd.DataFrame(clf), pd.DataFrame(priv), pd.DataFrame(lift)
    for name, df in dict(channel_scenarios=chan, auction=auc, budget_curve=curve, classifier_eval=clf,
                         privacy=priv, lift=lift).items():
        df.to_csv(RESULTS / f"{name}.csv", index=False)
    charts(curve, priv, auc)

    # Headline numbers, computed per seed then averaged.
    piv = chan.pivot(index="seed", columns="scenario")
    yt, ctx, kwd = "youtube_only", "youtube+gvp, context-aware (oracle)", "youtube+gvp, keyword blocklist"
    headline = {
        "seeds": args.seeds,
        "reach_lift_vs_youtube_only": float((piv.reach_pct[ctx] / piv.reach_pct[yt] - 1).mean()),
        "vcpm_change_vs_youtube_only": float((piv.vcpm[ctx] / piv.vcpm[yt] - 1).mean()),
        "cost_per_1k_reached_change": float((piv.cost_per_1k_reached[ctx] / piv.cost_per_1k_reached[yt] - 1).mean()),
        "gvp_only_share_of_reach": float((piv.gvp_only_reach[ctx] / piv.reach[ctx]).mean()),
        "unsafe_rate_no_filter": float(piv.unsafe_impression_rate["youtube+gvp, no filter"].mean()),
        "unsafe_rate_keyword": float(piv.unsafe_impression_rate[kwd].mean()),
        "keyword_over_block": float(clf[(clf.classifier == "keyword_blocklist") & (clf.suitability == "standard")].over_block.mean()),
        "keyword_leak": float(clf[(clf.classifier == "keyword_blocklist") & (clf.suitability == "standard")].leak.mean()),
        "keyword_tricky_accuracy": float(clf[(clf.classifier == "keyword_blocklist") & (clf.suitability == "standard")].tricky_decision_accuracy.mean()),
        "suitable_publisher_revenue_lost_share": float(chan[chan.scenario == kwd].suitable_publisher_revenue_lost_share.mean()),
    }
    a = auc.pivot(index="seed", columns="auction")
    headline.update({
        "first_price_naive_reach_change": float((a.reach_pct["first_price / truthful"] / a.reach_pct["second_price / truthful"] - 1).mean()),
        "first_price_shaded_reach_change": float((a.reach_pct["first_price / shaded"] / a.reach_pct["second_price / truthful"] - 1).mean()),
        "shading_reach_recovery_vs_naive": float((a.reach_pct["first_price / shaded"] / a.reach_pct["first_price / truthful"] - 1).mean()),
        "first_price_shaded_publisher_revenue_change": float((a.gvp_publisher_revenue["first_price / shaded"] / a.gvp_publisher_revenue["second_price / truthful"] - 1).mean()),
    })
    pg = priv[(priv.granularity == "dma x age")]
    headline.update({
        "dma_eps1_k10_error": float(pg[(pg.epsilon == 1.0) & (pg.k == 10)].total_reach_error.mean()),
        "dma_eps1_k50_error": float(pg[(pg.epsilon == 1.0) & (pg.k == 50)].total_reach_error.mean()),
        "dma_eps1_k50_suppressed_share": float(pg[(pg.epsilon == 1.0) & (pg.k == 50)].suppressed_reach_share.mean()),
        "lift_estimate_mean": float(lift.lift_estimate.mean()),
        "lift_true_mean": float(lift.true_lift.mean()),
        "lift_significant_runs": int(lift.significant.sum()),
    })
    (RESULTS / "headline.json").write_text(json.dumps(headline, indent=2))

    md = ["# Results", "",
          f"Generated by `python -m gvplab.experiments --seeds {args.seeds}`. Mean ± std across seeds. "
          "All data is simulated; see the README for assumptions.", "",
          "## 1. Channel scenarios (equal budget)", "",
          fmt(_agg(chan, "scenario"), KEY_METRICS), "",
          "## 2. Brand-safety classifier (decision level, traffic-weighted)", "",
          fmt(_agg(clf, ["classifier", "suitability"]), ["tier_accuracy", "over_block", "leak", "tricky_decision_accuracy"]), "",
          "## 3. Auction format (equal budget, oracle brand safety)", "",
          fmt(_agg(auc, "auction"), ["reach_pct", "vcpm", "advertiser_surplus", "gvp_publisher_revenue"]), "",
          "## 4. Privacy-safe reach report", "",
          fmt(_agg(priv.drop(columns=["cells"]), ["granularity", "epsilon", "k"]),
              ["total_reach_error", "incremental_gvp_reach_error", "suppressed_reach_share"]), "",
          "## 5. Brand-lift study (10% holdout, 15% survey sample)", "",
          lift.drop(columns="seed").round(4).to_markdown(index=False), "",
          "## Headline", "", "```json", json.dumps(headline, indent=2), "```", ""]

    if args.llm:
        res = llm_eval(args.llm, args.model, args.llm_limit, build_world(seed=0))
        res.to_csv(RESULTS / f"llm_eval_{args.llm}.csv", index=False)
        md += [f"## 6. LLM classifier ({args.llm})", "",
               res.round(3).to_markdown(index=False), "",
               f"LLM calls: {res.attrs['llm_calls']}, unparsed (failed closed): {res.attrs['llm_failures']}", ""]
    (RESULTS / "RESULTS.md").write_text("\n".join(md))
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
