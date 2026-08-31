"""Deterministic, channel-preserving communication-budget utilities."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .bridges import TypedLatentBridge
from .state_capture import AlignedMessageStates


@dataclass(frozen=True)
class BudgetedChannels:
    channels: dict[str, AlignedMessageStates]
    original_lengths: dict[str, int]
    transmitted_lengths: dict[str, int]

    @property
    def total_tokens(self) -> int:
        return sum(self.transmitted_lengths.values())


def channel_causal_tail(
    states: dict[str, AlignedMessageStates],
    *,
    slots_per_channel: int,
) -> BudgetedChannels:
    """Keep each semantic channel represented instead of truncating after concat."""
    required = set(TypedLatentBridge.channel_order)
    if set(states) != required:
        raise ValueError(f"expected channels {sorted(required)}")
    if slots_per_channel < 2:
        raise ValueError("each channel needs at least two states for StateBridge")
    selected: dict[str, AlignedMessageStates] = {}
    original: dict[str, int] = {}
    transmitted: dict[str, int] = {}
    for name in TypedLatentBridge.channel_order:
        item = states[name]
        length = int(item.token_ids.shape[1])
        keep = min(length, slots_per_channel)
        if keep < 2:
            raise ValueError(f"channel {name} has fewer than two aligned states")
        selected[name] = AlignedMessageStates(
            item.token_ids[:, -keep:], item.states[:, -keep:], item.semantics
        )
        original[name] = length
        transmitted[name] = keep
    return BudgetedChannels(selected, original, transmitted)


def concatenate_budgeted(values: BudgetedChannels) -> AlignedMessageStates:
    return AlignedMessageStates(
        token_ids=torch.cat(
            [values.channels[name].token_ids for name in TypedLatentBridge.channel_order], dim=1
        ),
        states=torch.cat(
            [values.channels[name].states for name in TypedLatentBridge.channel_order], dim=1
        ),
        semantics="\n\n".join(
            values.channels[name].semantics for name in TypedLatentBridge.channel_order
        ),
    )
