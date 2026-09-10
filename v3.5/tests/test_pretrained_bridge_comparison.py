from __future__ import annotations

import pytest

from scripts.generate_pretrained_bridge_comparison import generation_schedule


def test_homer_schedule_is_unique_and_reproducible() -> None:
    expected = [
        {"trial": 0, "candidate_index": 0, "generation_seed": 100},
        {"trial": 0, "candidate_index": 1, "generation_seed": 101},
        {"trial": 1, "candidate_index": 0, "generation_seed": 200},
        {"trial": 1, "candidate_index": 1, "generation_seed": 201},
    ]
    assert generation_schedule(trials=2, candidates=2, base_seed=100) == expected
    assert generation_schedule(trials=2, candidates=2, base_seed=100) == expected


def test_schedule_rejects_colliding_trial_stride() -> None:
    with pytest.raises(ValueError, match="trial_stride"):
        generation_schedule(trials=2, candidates=5, base_seed=100, trial_stride=4)


def test_schedule_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        generation_schedule(trials=0, candidates=5, base_seed=100)
