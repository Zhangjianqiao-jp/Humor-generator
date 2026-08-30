from __future__ import annotations

import pytest

from src.training.dpo_dataset import (
    ImageBalancedPreferenceDataset,
    PreferenceDataset,
    preference_sampling_dataset,
)


def fake_preference_dataset() -> PreferenceDataset:
    dataset = PreferenceDataset.__new__(PreferenceDataset)
    dataset.rows = [
        {"image_id": "a", "pair": 1},
        {"image_id": "a", "pair": 2},
        {"image_id": "b", "pair": 3},
    ]
    return dataset


def test_all_pairs_preserves_every_selected_pair() -> None:
    dataset = fake_preference_dataset()
    sampled = preference_sampling_dataset(dataset, "all_pairs", seed=7, randomize=True)
    assert sampled is dataset
    assert len(sampled) == 3


def test_image_balanced_retains_one_pair_per_image() -> None:
    sampled = preference_sampling_dataset(
        fake_preference_dataset(), "image_balanced", seed=7, randomize=False
    )
    assert isinstance(sampled, ImageBalancedPreferenceDataset)
    assert len(sampled) == 2
    assert sampled[0]["pair"] == 1
    assert sampled[1]["pair"] == 3


def test_unknown_sampling_mode_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unknown preference sampling mode"):
        preference_sampling_dataset(fake_preference_dataset(), "mystery", seed=7, randomize=False)
