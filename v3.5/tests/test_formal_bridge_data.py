from pathlib import Path

import torch

from humor_generator_v35.data.traces import load_trace, save_trace
from humor_generator_v35.latent.state_capture import AlignedMessageStates
from humor_generator_v35.training.formal_bridge import cluster_balanced_rows, shuffled_cluster_map
from scripts.train_bridge import fixed_hash_sample_clusters


def test_trace_roundtrip_is_exact_and_hashed(tmp_path: Path) -> None:
    states = {
        name: AlignedMessageStates(
            torch.tensor([[1, 2]]), torch.randn(1, 2, 4), f"actual {name} output",
        )
        for name in ("conflict", "local", "global")
    }
    path = tmp_path / "trace.pt"
    digest = save_trace(path, states)
    loaded = load_trace(path, expected_sha256=digest)
    assert set(loaded) == set(states)
    for name in states:
        assert torch.equal(loaded[name].token_ids, states[name].token_ids)
        torch.testing.assert_close(loaded[name].states, states[name].states.half())


def test_cluster_balancing_and_derangement() -> None:
    rows = [
        {"cluster_id": "nycc_1", "row_id": "1:a"},
        {"cluster_id": "nycc_1", "row_id": "1:b"},
        {"cluster_id": "nycc_2", "row_id": "2:a"},
        {"cluster_id": "nycc_3", "row_id": "3:a"},
    ]
    selected = cluster_balanced_rows(rows, epoch=1, seed=7)
    assert len(selected) == 3
    assert len({row["cluster_id"] for row in selected}) == 3
    assert next(row for row in selected if row["cluster_id"] == "nycc_1")["row_id"] == "1:b"
    mapping = shuffled_cluster_map(rows, seed=9)
    assert all(source != target for source, target in mapping.items())
    assert set(mapping) == set(mapping.values())


def test_pilot_subset_is_seeded_and_independent_of_input_order() -> None:
    rows = [
        {"cluster_id": f"c{index}", "row_id": f"{index}:a"}
        for index in range(20)
    ]
    first = fixed_hash_sample_clusters(rows, 6, seed=7, split="train")
    second = fixed_hash_sample_clusters(list(reversed(rows)), 6, seed=7, split="train")
    third = fixed_hash_sample_clusters(rows, 6, seed=8, split="train")
    assert {row["cluster_id"] for row in first} == {row["cluster_id"] for row in second}
    assert {row["cluster_id"] for row in first} != {row["cluster_id"] for row in third}
