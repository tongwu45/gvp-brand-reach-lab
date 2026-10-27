"""Campaign measurement: delivery metrics, incremental reach, a privacy-safe
reach report and a randomized brand-lift study."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .auction import Delivery
from .config import TOLERANCE
from .inventory import World


def summarize(d: Delivery, world: World) -> dict:
    imp = d.impressions
    pl = world.placements
    tol = TOLERANCE[d.config.suitability]
    spend = imp.price_cpm.sum() / 1000.0
    viewable = imp[imp.viewable]
    reached = viewable.user_id.unique()
    users = world.users.set_index("user_id")
    nets = viewable.groupby("user_id").network.agg(lambda s: frozenset(s))
    gvp_only = int(sum(1 for n in nets if n == frozenset({"gvp"})))
    eligible_users = len(world.users) - len(d.holdout_users)
    unsafe = pl.true_tier.values[imp.placement_id.values] > tol
    gvp = imp.network == "gvp"
    return {
        "spend": spend,
        "impressions": len(imp),
        "viewable_impressions": len(viewable),
        "viewability": len(viewable) / max(len(imp), 1),
        "ecpm": 1000 * spend / max(len(imp), 1),
        "vcpm": 1000 * spend / max(len(viewable), 1),
        "reach": len(reached),
        "reach_pct": len(reached) / eligible_users,
        "in_target_reach": int(users.in_target.loc[reached].sum()),
        "avg_frequency": len(viewable) / max(len(reached), 1),
        "gvp_only_reach": gvp_only,
        "gvp_share_of_spend": imp.price_cpm[gvp].sum() / 1000.0 / max(spend, 1e-9),
        "unsafe_impression_rate": float(unsafe.mean()) if len(imp) else 0.0,
        "gvp_publisher_revenue": imp.price_cpm[gvp].sum() / 1000.0,
        "advertiser_surplus": (imp.value_cpm - imp.price_cpm).sum() / 1000.0,
        "cost_per_1k_reached": 1000 * spend / max(len(reached), 1),
        "blocked_share": d.blocked_share,
    }


# --------------------------------------------------------------------------
# Privacy-safe reach report
# --------------------------------------------------------------------------
PATHS = ("youtube_only", "gvp_only", "both")


def _domain(world: World, dim: str) -> list[str]:
    if dim == "dma":
        return [f"dma_{i:02d}" for i in range(world.cfg.n_dmas)]
    return list(world.cfg.age_buckets if dim == "age" else world.cfg.regions)


def reach_cells(d: Delivery, world: World, dims: tuple[str, ...] = ("age", "region")) -> pd.DataFrame:
    """True reach counts over a fixed, disjoint cell domain: dims x exposure path.

    Each reached user falls in exactly one cell, so adding or removing one user
    changes one count by one (L1 sensitivity = 1).
    """
    v = d.impressions[d.impressions.viewable]
    nets = v.groupby("user_id").network.agg(lambda s: frozenset(s))
    path = nets.map(lambda n: "both" if len(n) == 2 else ("youtube_only" if "youtube" in n else "gvp_only"))
    u = world.users.set_index("user_id").loc[path.index, list(dims)].assign(path=path.values)
    counts = u.value_counts([*dims, "path"]).rename("true_count")
    domain = pd.MultiIndex.from_tuples(
        itertools.product(*(_domain(world, dim) for dim in dims), PATHS), names=[*dims, "path"])
    return counts.reindex(domain, fill_value=0).reset_index()


def private_reach_report(d: Delivery, world: World, epsilon: float = 1.0, k: int = 50,
                         dims: tuple[str, ...] = ("age", "region"),
                         seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Release reach cells with the Laplace mechanism (epsilon-DP), then suppress
    cells whose *noisy* count is below k. Thresholding after noise keeps the
    guarantee, since it only post-processes the noisy output."""
    rng = np.random.default_rng(seed + 99)
    cells = reach_cells(d, world, dims)
    cells["noisy_count"] = cells.true_count + rng.laplace(0.0, 1.0 / epsilon, len(cells))
    cells["released"] = cells.noisy_count >= k
    cells["reported"] = np.where(cells.released, np.round(cells.noisy_count), 0).astype(int)

    def err(mask) -> float:
        t = cells.true_count[mask].sum()
        return abs(cells.reported[mask].sum() - t) / max(t, 1)

    stats = {
        "epsilon": epsilon,
        "k": k,
        "granularity": " x ".join(dims),
        "cells": len(cells),
        "total_reach_error": err(slice(None)),
        "incremental_gvp_reach_error": err(cells.path == "gvp_only"),
        "suppressed_cells": int((~cells.released).sum()),
        "suppressed_reach_share": cells.true_count[~cells.released].sum() / max(cells.true_count.sum(), 1),
    }
    return cells, stats


# --------------------------------------------------------------------------
# Randomized brand-lift study
# --------------------------------------------------------------------------
def lift_study(d: Delivery, world: World, base_rate: float = 0.30, max_lift: float = 0.06,
               decay: float = 0.5, survey_share: float = 0.15, seed: int = 0) -> dict:
    """Intent-to-treat lift on a survey outcome (e.g. ad recall).

    Users were randomised into treatment and a holdout before delivery. Each
    user's probability of a positive answer rises with viewable frequency f as
    base + max_lift * (1 - exp(-decay * f)). A random share of users is surveyed;
    the ITT estimate compares treated vs holdout response rates.
    """
    rng = np.random.default_rng(seed + 7)
    n = len(world.users)
    freq = np.zeros(n)
    vc = d.impressions[d.impressions.viewable].user_id.value_counts()
    freq[vc.index.values] = vc.values
    p = base_rate + max_lift * (1 - np.exp(-decay * freq))
    control = np.zeros(n, dtype=bool)
    control[d.holdout_users] = True
    surveyed = rng.random(n) < survey_share
    answer = rng.random(n) < p
    t, c = answer[surveyed & ~control], answer[surveyed & control]
    est = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    true_itt = p[~control].mean() - base_rate
    return {
        "lift_estimate": float(est),
        "ci_low": float(est - 1.96 * se),
        "ci_high": float(est + 1.96 * se),
        "true_lift": float(true_itt),
        "significant": bool(est - 1.96 * se > 0),
        "n_treated": int(len(t)),
        "n_control": int(len(c)),
    }
