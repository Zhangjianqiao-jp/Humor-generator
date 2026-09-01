"""Latent communication baselines; never imported by strict HOMER text runs."""

from .bridges import LearnedLatentBridge, TypedLatentBridge
from .statebridge import StateBridgeAlignment

__all__ = ["LearnedLatentBridge", "TypedLatentBridge", "StateBridgeAlignment"]
