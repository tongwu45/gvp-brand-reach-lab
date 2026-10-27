"""Synthetic video inventory, users and ad opportunities.

The world is generated once per seed and then shared across scenarios, so every
scenario sees the same users, placements, competing bids and viewability draws
(common random numbers). Differences between scenarios therefore come from the
policy being tested, not from noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FLOOR, HIGH, LOW, MEDIUM, SAFE, MarketConfig

# --------------------------------------------------------------------------
# Title bank. Each entry: (title, true tier, is_tricky).
# "Tricky" titles are where surface keywords and real meaning disagree:
# safe content that contains alarming words, or risky content with none.
# --------------------------------------------------------------------------
STRAIGHTFORWARD = [
    ("10 easy weeknight pasta recipes", SAFE),
    ("Morning yoga flow for beginners", SAFE),
    ("Tiny house tour: 200 sq ft of clever storage", SAFE),
    ("Electric SUV road test and range review", SAFE),
    ("Learn Spanish in 15 minutes a day", SAFE),
    ("Highlights: championship final, last two minutes", SAFE),
    ("Budget travel guide to Lisbon", SAFE),
    ("How compound interest actually works", SAFE),
    ("Indie folk live session in the studio", SAFE),
    ("Kids science experiment with baking soda", SAFE),
    ("Unboxing the new noise-cancelling headphones", SAFE),
    ("Weekly weather outlook for the Pacific Northwest", SAFE),
    ("Stand-up set full of crude jokes and swearing", LOW),
    ("Horror movie trailer reaction", LOW),
    ("Boxing match full fight replay", LOW),
    ("Craft beer tasting and brewery tour", LOW),
    ("Explicit lyrics rap cypher", LOW),
    ("Breaking news: police respond to violent protest downtown", MEDIUM),
    ("Documentary on the opioid crisis and overdose deaths", MEDIUM),
    ("War footage analysis from the front line", MEDIUM),
    ("True crime: the murder case that shocked a town", MEDIUM),
    ("Graphic street fight caught on camera, blood everywhere", HIGH),
    ("Uncensored nude scenes compilation", HIGH),
    ("How to buy weed and pills without a prescription", HIGH),
    ("Racist rant goes viral, slurs and hate speech", HIGH),
    ("Terrorist propaganda video, recruitment message", FLOOR),
    ("Execution footage, extremely graphic gore", FLOOR),
]

TRICKY = [
    # alarming keywords, safe meaning
    ("Portrait photography: shooting in golden hour light", SAFE),
    ("DIY bath bomb tutorial for a relaxing night", SAFE),
    ("Killer deals: best laptops under $500", SAFE),
    ("Pharmacy tour: how drug store loyalty points work", SAFE),
    ("How to fight climate change in your own kitchen", SAFE),
    ("This chili recipe is so hot it's criminal", SAFE),
    ("Breast cancer awareness: early screening explained", SAFE),
    ("Shot-by-shot breakdown of a Wes Anderson film", SAFE),
    ("Gaming: we destroyed the final boss on hard mode", SAFE),
    ("History lecture: causes of the Second World War", SAFE),
    # risky meaning, no obvious keywords
    ("What they did to him in that alley - full uncut video", HIGH),
    ("Leaked clip from her private party, watch before deleted", HIGH),
    ("The pharmacist won't tell you this trick to get more pills", HIGH),
    ("Why those people should not be allowed in our country", HIGH),
    ("Doctors hate this: cure diabetes in 3 days with one herb", MEDIUM),
    ("They faked the election and here is the proof", MEDIUM),
    ("Prank gone wrong, he didn't wake up", MEDIUM),
    ("Challenge: how long can you hold your breath underwater", LOW),
]

SUFFIXES = ["", " | Part 2", " (2026)", " - full video", " #shorts", " | Episode 14", " [HD]", " - reaction"]


@dataclass
class World:
    cfg: MarketConfig
    users: pd.DataFrame
    placements: pd.DataFrame
    opps: pd.DataFrame


def _make_placements(cfg: MarketConfig, rng: np.random.Generator) -> pd.DataFrame:
    banks: dict[tuple[bool, int], list[str]] = {}
    for title, tier in STRAIGHTFORWARD:
        banks.setdefault((False, tier), []).append(title)
    for title, tier in TRICKY:
        banks.setdefault((True, tier), []).append(title)
    rows = []
    pid = 0
    for t in cfg.tiers:
        tiers = rng.choice(5, size=t.n_placements, p=np.array(t.risk_mix) / sum(t.risk_mix))
        use_tricky = rng.random(t.n_placements) < cfg.tricky_share
        view = rng.beta(*t.viewability, size=t.n_placements)
        for i in range(t.n_placements):
            tier = int(tiers[i])
            is_tricky = bool(use_tricky[i]) and (True, tier) in banks  # no tricky FLOOR titles
            pool = banks[(is_tricky, tier)]
            rows.append({
                "placement_id": pid,
                "network": t.network,
                "pub_tier": t.name,
                "title": pool[rng.integers(len(pool))] + SUFFIXES[rng.integers(len(SUFFIXES))],
                "true_tier": tier,
                "tricky": is_tricky,
                "viewability": float(view[i]),
                "floor_cpm": t.floor_cpm,
                "comp_median": t.comp_cpm_median,
                "comp_sigma": t.comp_sigma,
                "traffic": t.traffic_weight / t.n_placements,
            })
            pid += 1
    return pd.DataFrame(rows)


def _make_users(cfg: MarketConfig, rng: np.random.Generator) -> pd.DataFrame:
    shares = np.array([s.share for s in cfg.segments])
    seg_idx = rng.choice(len(cfg.segments), size=cfg.n_users, p=shares / shares.sum())
    dma_p = 1.0 / np.arange(1, cfg.n_dmas + 1) ** 0.9
    dma_p /= dma_p.sum()
    return pd.DataFrame({
        "user_id": np.arange(cfg.n_users),
        "segment": np.array([s.name for s in cfg.segments])[seg_idx],
        "seg_idx": seg_idx,
        "in_target": rng.random(cfg.n_users) < cfg.target_share,
        "age": rng.choice(cfg.age_buckets, size=cfg.n_users),
        "region": rng.choice(cfg.regions, size=cfg.n_users),
        "dma": rng.choice([f"dma_{i:02d}" for i in range(cfg.n_dmas)], size=cfg.n_users, p=dma_p),
    })


def _make_opportunities(cfg, users, placements, rng) -> pd.DataFrame:
    frames = []
    for network, rate_attr in (("youtube", "yt_rate"), ("gvp", "gvp_rate")):
        rates = np.array([getattr(s, rate_attr) for s in cfg.segments])[users.seg_idx.values]
        # Gamma-Poisson mixture: heterogeneous viewing intensity within a segment.
        lam = rng.gamma(shape=2.0, scale=rates / 2.0)
        counts = rng.poisson(lam)
        uid = np.repeat(users.user_id.values, counts)
        pool = placements[placements.network == network]
        p = pool.traffic.values / pool.traffic.values.sum()
        pidx = rng.choice(pool.placement_id.values, size=len(uid), p=p)
        frames.append(pd.DataFrame({"user_id": uid, "placement_id": pidx, "network": network}))
    opps = pd.concat(frames, ignore_index=True)
    pl = placements.set_index("placement_id")
    med = pl.comp_median.values[opps.placement_id.values]
    sig = pl.comp_sigma.values[opps.placement_id.values]
    opps["comp_cpm"] = med * np.exp(sig * rng.standard_normal(len(opps)))   # best competing bid
    opps["view_draw"] = rng.random(len(opps))                               # viewable if < viewability
    opps["t"] = rng.random(len(opps))                                       # arrival time in the flight
    return opps.sort_values("t", kind="stable").reset_index(drop=True)


def build_world(cfg: MarketConfig | None = None, seed: int = 0) -> World:
    cfg = cfg or MarketConfig()
    rng = np.random.default_rng(seed)
    placements = _make_placements(cfg, rng)
    users = _make_users(cfg, rng)
    opps = _make_opportunities(cfg, users, placements, rng)
    return World(cfg, users, placements, opps)
