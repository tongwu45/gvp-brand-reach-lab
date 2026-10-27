"""Configuration objects for the simulation.

Every number here is an *assumption* chosen to be directionally realistic, not
Google data. They are deliberately exposed so a reviewer can change them and see
how the conclusions move.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Brand-safety risk tiers, loosely modelled on the GARM suitability framework.
SAFE, LOW, MEDIUM, HIGH, FLOOR = 0, 1, 2, 3, 4
TIER_NAMES = {SAFE: "safe", LOW: "low", MEDIUM: "medium", HIGH: "high", FLOOR: "floor"}

# Advertiser suitability settings -> highest tier the advertiser accepts.
TOLERANCE = {"limited": SAFE, "standard": LOW, "expanded": MEDIUM}


@dataclass
class UserSegment:
    name: str
    share: float          # share of the user population
    yt_rate: float        # mean weekly ad opportunities on YouTube
    gvp_rate: float       # mean weekly ad opportunities on GVP publishers


@dataclass
class PublisherTier:
    name: str
    network: str              # "youtube" or "gvp"
    n_placements: int
    viewability: tuple[float, float]   # Beta(a, b) for placement viewability
    risk_mix: tuple[float, ...]        # probability of SAFE..FLOOR
    floor_cpm: float                   # publisher reserve price, $ CPM
    comp_cpm_median: float             # median of the best competing bid, $ CPM
    comp_sigma: float = 0.6            # log-normal spread of competing bids
    traffic_weight: float = 1.0        # relative share of the network's traffic


@dataclass
class MarketConfig:
    n_users: int = 200_000
    target_share: float = 0.35         # share of users in the campaign's target audience
    tricky_share: float = 0.25         # share of titles where keywords mislead
    segments: list[UserSegment] = field(default_factory=lambda: [
        UserSegment("youtube_heavy", 0.40, yt_rate=16.0, gvp_rate=2.0),
        UserSegment("mixed",         0.35, yt_rate=7.0,  gvp_rate=6.0),
        UserSegment("off_youtube",   0.25, yt_rate=1.0,  gvp_rate=9.0),
    ])
    tiers: list[PublisherTier] = field(default_factory=lambda: [
        PublisherTier("youtube",        "youtube", 3000, (18, 2.5), (0.82, 0.12, 0.04, 0.015, 0.005), 3.0, 9.0, traffic_weight=1.0),
        PublisherTier("gvp_premium",    "gvp",      600, (14, 3.5), (0.80, 0.12, 0.06, 0.015, 0.005), 3.0, 6.5, traffic_weight=0.45),
        PublisherTier("gvp_long_tail",  "gvp",     1200, (6, 3.0),  (0.55, 0.20, 0.13, 0.09, 0.03),  1.0, 3.0, traffic_weight=0.40),
        PublisherTier("gvp_mfa",        "gvp",      400, (3, 5.0),  (0.50, 0.25, 0.15, 0.08, 0.02),  0.5, 2.0, traffic_weight=0.15),
    ])
    age_buckets: tuple[str, ...] = ("18-24", "25-34", "35-44", "45-54", "55+")
    regions: tuple[str, ...] = ("west", "midwest", "south", "northeast")
    n_dmas: int = 60                   # local TV markets, with Zipf-like population sizes


@dataclass
class CampaignConfig:
    budget: float = 800.0              # $ for the flight (binding: delivery is budget-limited)
    value_cpm: float = 14.0            # advertiser value of a viewable, in-target impression ($ CPM)
    off_target_mult: float = 0.4       # value multiplier for users outside the target audience
    frequency_cap: int = 4             # max impressions per user per flight
    networks: tuple[str, ...] = ("youtube", "gvp")
    suitability: str = "standard"      # limited | standard | expanded
    auction: str = "second_price"      # second_price | first_price
    bidding: str = "truthful"          # truthful | shaded
    holdout_share: float = 0.10        # users held out for the lift study
