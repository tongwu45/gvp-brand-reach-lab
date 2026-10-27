"""Auction and delivery simulation for one brand campaign.

For each ad opportunity the campaign computes its value, applies the brand-safety
filter, and bids against the best competing bid and the publisher floor. Wins
are then filtered in arrival order by the frequency cap and the budget.

Both constraints can be applied after the fact without changing any other
outcome, because the competing bids do not react to our campaign; this keeps
the whole run vectorised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FLOOR, TOLERANCE, CampaignConfig
from .inventory import World


@dataclass
class Delivery:
    """Result of one campaign run."""

    config: CampaignConfig
    impressions: pd.DataFrame          # won impressions, one row each
    holdout_users: np.ndarray          # user_ids excluded from delivery (lift control group)
    blocked_share: float               # share of eligible opportunities removed by the safety filter


# --------------------------------------------------------------------------
# Bid shading for first-price auctions
# --------------------------------------------------------------------------
def shading_table(comp_sample: np.ndarray, floor: float, v_grid: np.ndarray) -> np.ndarray:
    """Optimal first-price bid for each value in ``v_grid``.

    Maximises expected surplus (v - b) * P(win | b), where P(win | b) is the
    empirical CDF of the best competing bid, and a bid below the floor never wins.
    """
    comp_sorted = np.sort(comp_sample)
    b_grid = np.linspace(0, v_grid.max(), 600)
    win = np.searchsorted(comp_sorted, b_grid, side="left") / len(comp_sorted)
    win = np.where(b_grid >= floor, win, 0.0)
    surplus = (v_grid[:, None] - b_grid[None, :]) * win[None, :]
    return b_grid[surplus.argmax(axis=1)]


def _shaded_bids(values: np.ndarray, pub_tier: np.ndarray, world: World, rng) -> np.ndarray:
    """Learn a shading curve per publisher tier from *historical* competing bids
    (fresh draws, not the live ones) and apply it."""
    bids = np.zeros_like(values)
    pl = world.placements.drop_duplicates("pub_tier").set_index("pub_tier")
    v_grid = np.linspace(0, max(values.max(), 1e-6), 400)
    for tier in np.unique(pub_tier):
        row = pl.loc[tier]
        history = row.comp_median * np.exp(row.comp_sigma * rng.standard_normal(20_000))
        table = shading_table(history, row.floor_cpm, v_grid)
        mask = pub_tier == tier
        bids[mask] = np.interp(values[mask], v_grid, table)
    return bids


# --------------------------------------------------------------------------
# Campaign run
# --------------------------------------------------------------------------
def run_campaign(world: World, cfg: CampaignConfig, tier_pred: np.ndarray, seed: int = 0) -> Delivery:
    """Simulate one flight.

    tier_pred: predicted brand-safety tier per placement (index = placement_id).
    """
    rng = np.random.default_rng(seed + 10_007)
    users, pl, opps = world.users, world.placements, world.opps

    # Lift-study holdout: same users held out in every scenario for a given seed.
    holdout_rng = np.random.default_rng(seed + 1)
    holdout = holdout_rng.random(len(users)) < cfg.holdout_share
    holdout_users = users.user_id.values[holdout]

    o = opps[opps.network.isin(cfg.networks)]
    o = o[~holdout[o.user_id.values]]
    pid = o.placement_id.values

    # Brand safety: platform floor is always excluded, then the advertiser setting.
    tol = TOLERANCE[cfg.suitability]
    blocked = (pl.true_tier.values[pid] == FLOOR) | (tier_pred[pid] > tol)

    # Value of the impression to the advertiser ($ CPM), using placement viewability
    # as the predicted view rate and audience membership as the targeting signal.
    in_target = users.in_target.values[o.user_id.values]
    values = cfg.value_cpm * pl.viewability.values[pid] * np.where(in_target, 1.0, cfg.off_target_mult)

    if cfg.auction == "first_price" and cfg.bidding == "shaded":
        bids = _shaded_bids(values, pl.pub_tier.values[pid], world, rng)
    else:
        bids = values  # truthful bidding (dominant strategy in second price)

    comp = o.comp_cpm.values
    floor = pl.floor_cpm.values[pid]
    win = (~blocked) & (bids >= floor) & (bids > comp)

    if cfg.auction == "second_price":
        price = np.maximum(comp, floor)
    else:
        price = bids

    won = o[win].copy()
    won["bid_cpm"] = bids[win]
    won["price_cpm"] = price[win]
    won["value_cpm"] = values[win]
    won["viewable"] = won.view_draw.values < pl.viewability.values[won.placement_id.values]

    # Even pacing: participate in a random share p of auctions, with p chosen so
    # the flight spends (close to) its budget over the whole period instead of
    # running out early. The frequency cap is applied in arrival order after
    # throttling, because throttled auctions never serve.
    pace_draw = rng.random(len(won))

    def deliver(p: float) -> pd.DataFrame:
        kept = won[pace_draw < p]
        return kept[kept.groupby("user_id").cumcount() < cfg.frequency_cap]

    full = deliver(1.0)
    if full.price_cpm.sum() / 1000.0 > cfg.budget:
        lo, hi = 0.0, 1.0
        for _ in range(25):
            mid = (lo + hi) / 2
            if deliver(mid).price_cpm.sum() / 1000.0 > cfg.budget:
                hi = mid
            else:
                lo = mid
        full = deliver(lo)
    won = full.reset_index(drop=True)

    eligible = len(o)
    return Delivery(cfg, won, holdout_users, float(blocked.mean()) if eligible else 0.0)
