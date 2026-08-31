"""Registry for the four preregistered communication baselines."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Baseline(str, Enum):
    TEXT_HOMER = "text_homer"
    STATEBRIDGE = "statebridge"
    LEARNED = "learned_latent"
    TYPED = "typed_learned_latent"


@dataclass(frozen=True)
class BaselineSpec:
    name: Baseline
    trainable: bool
    transmitted_fields: tuple[str, ...]
    receiver_condition: str


BASELINES = {
    Baseline.TEXT_HOMER: BaselineSpec(
        Baseline.TEXT_HOMER, False, ("description", "conflicts", "local", "global", "retrieval"), "text"
    ),
    Baseline.STATEBRIDGE: BaselineSpec(
        Baseline.STATEBRIDGE, False, ("conflicts", "local", "global"), "continuous_prefix"
    ),
    Baseline.LEARNED: BaselineSpec(
        Baseline.LEARNED, True, ("conflicts", "local", "global"), "continuous_prefix"
    ),
    Baseline.TYPED: BaselineSpec(
        Baseline.TYPED, True, ("conflicts", "local", "global"), "typed_continuous_prefix"
    ),
}


def get_baseline(value: str | Baseline) -> BaselineSpec:
    return BASELINES[Baseline(value)]
