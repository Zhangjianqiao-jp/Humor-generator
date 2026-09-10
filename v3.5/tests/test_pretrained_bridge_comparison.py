from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_pretrained_bridge_comparison import (
    assert_complete_keys,
    generation_schedule,
    load_completed_keys,
    load_population,
)


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


def test_public_population_split_alias_is_normalized() -> None:
    rows = load_population(
        Path("data/processed/homer_pretrained_7b_public_release_362"),
        "test",
    )
    assert len(rows) == 47
    assert {row["split"] for row in rows} == {"test"}


def _generated_row(*, cluster: str = "c1", seed: int = 100) -> dict:
    return {
        "system_id": "text_homer_context_replay",
        "image_id": cluster,
        "image_sha256": "a" * 64,
        "split": "test",
        "generation_seed": seed,
        "trial": 0,
        "candidate_index": 0,
        "candidate_count": 2,
    }


def test_resumable_generation_rejects_duplicate_keys(tmp_path: Path) -> None:
    rows = [{"cluster_id": "c1", "image_sha256": "a" * 64}]
    schedule = generation_schedule(trials=1, candidates=2, base_seed=100)
    path = tmp_path / "generation.jsonl"
    row = _generated_row()
    path.write_text(
        "\n".join(json.dumps(row) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_completed_keys(
            path, condition="text_homer_context_replay", rows=rows,
            schedule=schedule, split="test",
        )


def test_generation_key_gate_requires_every_scheduled_candidate() -> None:
    rows = [{"cluster_id": "c1"}]
    schedule = generation_schedule(trials=1, candidates=2, base_seed=100)
    with pytest.raises(RuntimeError, match="incomplete"):
        assert_complete_keys({("c1", 100)}, rows=rows, schedule=schedule)
