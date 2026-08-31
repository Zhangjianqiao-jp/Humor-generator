"""Latent communication baselines; never imported by strict HOMER text runs."""

from .bridges import LearnedLatentBridge, TypedLatentBridge
from .statebridge import StateBridgeAlignment
from .budget import BudgetedChannels, channel_causal_tail, concatenate_budgeted

__all__ = [
    "LearnedLatentBridge", "TypedLatentBridge", "StateBridgeAlignment",
    "BudgetedChannels", "channel_causal_tail", "concatenate_budgeted",
]
