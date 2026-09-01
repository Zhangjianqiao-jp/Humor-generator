"""Uniform conditioning contract for the four preregistered baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..homer.contracts import HomerPlan
from ..homer.prompts import caption_messages
from .baselines import Baseline
from .bridges import LearnedLatentBridge, TypedLatentBridge
from .state_capture import AlignedMessageStates
from .statebridge import StateBridgeAlignment


@dataclass(frozen=True)
class ConditioningOutput:
    baseline: Baseline
    text_messages: list[dict[str, Any]] | None = None
    latent_slots: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (self.text_messages is None) == (self.latent_slots is None):
            raise ValueError("conditioning must contain exactly one of text_messages or latent_slots")
        if self.latent_slots is not None:
            if self.latent_slots.ndim != 3 or self.latent_slots.shape[1] < 1:
                raise ValueError("latent slots must be non-empty [B,S,D]")
            if not torch.isfinite(self.latent_slots).all():
                raise ValueError("latent slots contain non-finite values")


def text_homer_condition(
    plan: HomerPlan,
    *,
    conflict: str,
    path: tuple[str, ...],
    options: str = "",
) -> ConditioningOutput:
    return ConditioningOutput(
        baseline=Baseline.TEXT_HOMER,
        text_messages=caption_messages(plan.description, conflict, list(path), options),
    )


def statebridge_condition(
    aligner: StateBridgeAlignment,
    states: AlignedMessageStates,
) -> ConditioningOutput:
    slots = aligner(states.states, states.token_ids)
    return ConditioningOutput(Baseline.STATEBRIDGE, latent_slots=slots)


def learned_condition(
    bridge: LearnedLatentBridge,
    states: dict[str, AlignedMessageStates],
) -> ConditioningOutput:
    required = set(TypedLatentBridge.channel_order)
    if set(states) != required:
        raise ValueError(f"learned baseline requires exactly {sorted(required)}")
    values = [states[name].states for name in TypedLatentBridge.channel_order]
    width = {value.shape[2] for value in values}
    batch = {value.shape[0] for value in values}
    if len(width) != 1 or len(batch) != 1:
        raise ValueError("learned baseline channels must share batch and hidden width")
    return ConditioningOutput(Baseline.LEARNED, latent_slots=bridge(torch.cat(values, dim=1)))


def typed_condition(
    bridge: TypedLatentBridge,
    states: dict[str, AlignedMessageStates],
) -> ConditioningOutput:
    values = {name: value.states for name, value in states.items()}
    return ConditioningOutput(Baseline.TYPED, latent_slots=bridge(values)["all"])
