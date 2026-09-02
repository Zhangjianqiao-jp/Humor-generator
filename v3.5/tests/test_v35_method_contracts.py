from __future__ import annotations

from pathlib import Path

import torch
import yaml

from humor_generator_v35.evaluation.diversity import candidate_set_metrics, summarize_diversity
from humor_generator_v35.latent.budget import channel_causal_tail, concatenate_budgeted
from humor_generator_v35.latent.state_capture import AlignedMessageStates
from humor_generator_v35.latent.bridges import nearest_vocabulary_embeddings
from humor_generator_v35.training.formal_bridge import (
    full_plan_text_messages, hard_negative_cluster_map, latent_messages,
)
from humor_generator_v35.data.clustered import _finalists


ROOT = Path(__file__).resolve().parents[1]


def _states(length: int, offset: int) -> AlignedMessageStates:
    ids = torch.arange(offset, offset + length).reshape(1, -1)
    values = ids.unsqueeze(-1).repeat(1, 1, 4).float()
    return AlignedMessageStates(ids, values, f"semantics-{offset}")


def test_channel_budget_never_erases_an_early_channel() -> None:
    states = {"conflict": _states(4, 10), "local": _states(12, 20), "global": _states(30, 40)}
    budgeted = channel_causal_tail(states, slots_per_channel=3)
    assert budgeted.transmitted_lengths == {"conflict": 3, "local": 3, "global": 3}
    assert budgeted.channels["conflict"].token_ids.tolist() == [[11, 12, 13]]
    joined = concatenate_budgeted(budgeted)
    assert joined.token_ids.shape == (1, 9)
    assert joined.token_ids[0, :3].tolist() == [11, 12, 13]


def test_unknown_electronic_sheep_finalists_are_not_split_into_characters() -> None:
    assert _finalists({"official_newyorker_finalists": "UNKNOWN"}, 3) == []
    assert _finalists({"official_newyorker_finalists": ["nan", "A real caption"]}, 3) == [
        {"caption": "A real caption", "caption_rank": 1, "caption_score": None}
    ]


def test_nearest_vocabulary_control_preserves_shape_and_uses_exact_rows() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    slots = torch.tensor([[[0.9, 0.1], [-0.8, 0.1]]])
    quantized, token_ids = nearest_vocabulary_embeddings(slots, embeddings, chunk_size=2)
    assert token_ids.tolist() == [[0, 2]]
    assert torch.equal(quantized, embeddings[token_ids])


def test_sft_receiver_conditions_always_include_the_image() -> None:
    image = "/tmp/example.jpg"
    latent = latent_messages(image)
    text = full_plan_text_messages(
        image, {"conflict": "c", "local": "l", "global": "g"}
    )
    for messages in (latent, text):
        content = messages[0]["content"]
        assert content[0] == {"type": "image", "image": image}
    assert "Humor plan:" not in latent[0]["content"][1]["text"]
    assert "Humor plan:" in text[0]["content"][1]["text"]


def _plan(left: str, right: str) -> dict:
    return {
        "description": f"A long enough description featuring {left} and {right} in a cartoon.",
        "conflicts": [
            {"left": left, "right": right},
            {"left": f"normal {left}", "right": f"strange {right}"},
        ],
        "local_chains": [{"root": left, "steps": ["a", "b", "c"], "view": "local"}],
        "global_chains": [{"root": right, "steps": ["d", "e", "f"], "view": "global"}],
    }


def test_hard_negatives_are_not_self_and_have_different_conflicts() -> None:
    rows = [
        {"cluster_id": "a", "standard_description": "cat at an office desk", "dataset": "x"},
        {"cluster_id": "b", "standard_description": "dog at an office desk", "dataset": "x"},
        {"cluster_id": "c", "standard_description": "ship at sea", "dataset": "x"},
    ]
    traces = {
        "a": {"plan": _plan("cat", "worker")},
        "b": {"plan": _plan("dog", "manager")},
        "c": {"plan": _plan("ship", "office")},
    }
    mapping, diagnostics = hard_negative_cluster_map(rows, traces)
    assert all(source != target for source, target in mapping.items())
    assert diagnostics["same_source_fraction"] == 1.0
    assert mapping["a"] == "b"


def test_diversity_reports_quality_conditioned_and_angle_endpoints() -> None:
    captions = ["A cat files taxes.", "The feline audits us.", "A cat files taxes."]
    metrics = candidate_set_metrics(captions)
    assert 0 < metrics["unique_caption_rate"] < 1
    assert metrics["official_average_ead_n1_n5"] > 0
    assert metrics["official_sbert_all_mpnet_base_v2_diversity"] is None
    rows = [
        {
            "receiver": "sft", "condition": "typed", "cluster_id": "x",
            "caption": caption, "absolute_label": "good", "angle_label": angle,
        }
        for caption, angle in zip(captions, ("bureaucracy", "role reversal", "bureaucracy"))
    ]
    report = summarize_diversity(rows, min_candidates=3)
    assert report["per_cluster"][0]["good_only"] is not None
    assert report["per_cluster"][0]["human_angle_coverage"] == 2


def test_formal_jobs_use_native_full_gpu_contract_and_current_p1_path() -> None:
    training = (ROOT / "jobs/formal_bridge_train.pjm").read_text()
    generation = (ROOT / "jobs/pilot_validation_generation.pjm").read_text()
    fallback = (ROOT / "scripts/submit_formal_bridge_matrix.sh").read_text()
    for script in (training, generation):
        assert "PYTORCH_ALLOC_CONF=backend:native" in script
        assert "expandable_segments" not in script
    expected = "outputs/pilot/learned_sft_kl_visualcap_v2"
    assert expected in generation
    assert expected in fallback


def test_all_formal_comparison_configs_share_visual_token_budget() -> None:
    configs = [
        *sorted((ROOT / "configs/pilot").glob("*.yaml")),
        *sorted((ROOT / "configs/formal").glob("*.yaml")),
    ]
    budgets = set()
    for path in configs:
        model = yaml.safe_load(path.read_text())["model"]
        budgets.add((model.get("min_visual_tokens"), model.get("max_visual_tokens")))
    assert budgets == {(256, 1280)}


def test_resource_smoke_covers_image_and_full_memory_stress_samples() -> None:
    smoke = (ROOT / "scripts/real_trace_bridge_smoke.py").read_text()
    assert "max_raw_pixels_plus_max_full_latent_tokens_at_most_two_examples" in smoke
    assert "stress = max(selected" in smoke
    assert "memory_stress = max(selected" in smoke
    assert "for row in (stress, memory_stress)" in smoke
    assert 'config["bridge"]["layer_indices"]' in smoke
    assert 'config["bridge"]["receiver_layers"]' not in smoke
    assert "configure_frozen_receiver" in smoke


def test_formal_job_is_fail_closed_in_smoke_data_gpu_train_order() -> None:
    job = (ROOT / "jobs/formal_bridge_train.pjm").read_text()
    preflight = job.index("run_formal_preflight.py")
    resource = job.index("check_cuda_resource.py")
    stress = job.index("real_trace_bridge_smoke.py")
    training = job.index("train_bridge.py")
    assert preflight < resource < stress < training
    checker = (ROOT / "scripts/run_formal_preflight.py").read_text()
    assert checker.index('"scripts/train_bridge.py", "--help"') < checker.index(
        '"scripts/verify_clustered_dataset.py"'
    )
    assert '"scripts/check_trace_completion.py"' in checker


def test_dataset_audit_checks_every_byte_level_dependency() -> None:
    audit = (ROOT / "scripts/audit_dataset_records.py").read_text()
    for contract in (
        "image SHA-256 mismatch",
        "image.verify()",
        'manifest.get("input_sha256", {})',
        "duplicate row_id",
        "cluster leakage",
    ):
        assert contract in audit
