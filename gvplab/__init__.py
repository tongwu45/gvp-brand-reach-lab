"""GVP Brand Reach Lab: a simulation sandbox for extending YouTube brand demand
to third-party video publisher inventory."""

from .config import CampaignConfig, MarketConfig
from .inventory import World, build_world
from .auction import run_campaign
from .measurement import summarize, private_reach_report, lift_study

__all__ = ["CampaignConfig", "MarketConfig", "World", "build_world", "run_campaign",
           "summarize", "private_reach_report", "lift_study"]
__version__ = "0.1.0"
