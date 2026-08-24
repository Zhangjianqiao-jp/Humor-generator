from __future__ import annotations

from argparse import Namespace

import pytest

from scripts.preference_diagnostics.best_of_n import analyse
from scripts.preference_diagnostics.build_preference_pairs import classify, close_length, flatten
from scripts.preference_diagnostics.module_gradient_analysis import module_key
from scripts.preference_diagnostics.humor_representation_probe import binary_auc, probe_layers
from scripts.preference_diagnostics.judge_humor_candidates_qwen import normalize_scores, parse_score_payload
from scripts.preference_diagnostics.compare_best_of_n import paired_compare
from scripts.generate_lora_sft import generate_candidates
from scripts.judge_sft_candidates_qwen import normalize_judgment
from src.preference.diagnostics import derangement, module_identity, select_image_diverse_rows, text_features


def test_derangement_is_deterministic_bijective_and_has_no_fixed_points() -> None:
    keys = ["a", "b", "c", "d"]
    first = derangement(keys, 17)
    second = derangement(keys, 17)
    assert first == second
    assert set(first) == set(first.values()) == set(keys)
    assert all(key != donor for key, donor in first.items())


def test_best_of_n_uses_nested_prefixes() -> None:
    rows = [
        {"image_id": "a", "scores": [1, 4, 2, 5]},
        {"image_id": "b", "scores": [2, 3, 4, 1]},
    ]
    result = analyse(rows, (1, 2, 4), "humor", 4)
    assert [row["h_max"] for row in result] == pytest.approx([1.5, 3.5, 4.5])
    assert [row["p_good"] for row in result] == pytest.approx([0.0, 0.5, 1.0])


def test_flatten_reads_sft_score_without_inventing_dimensions() -> None:
    rows = [
        {
            "image": "x.jpg",
            "image_id": "x",
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": "x.jpg"}, {"type": "text", "text": "p"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "A caption."}]},
            ],
            "meta": {"score": 1.8, "source": "ranking"},
        }
    ]
    item = flatten(rows)[0]
    assert item["caption"] == "A caption."
    assert item["score"] == pytest.approx(1.8)
    assert item["humor"] == pytest.approx(1.8)
    assert item["grounding"] is None


def test_h1_requires_humor_gap_and_grounding_control() -> None:
    args = Namespace(
        min_naturalness=3.0,
        high_humor=4.0,
        low_humor=2.0,
        min_grounding=4.0,
        weak_humor=3.0,
        grounding_margin=2.0,
        specificity_margin=2.0,
        min_score_margin=0.35,
    )
    chosen = {"humor": 5, "grounding": 4, "originality": 4, "specificity": 4, "naturalness": 4, "score": 5, "features": text_features("A dry office joke.")}
    rejected = {"humor": 1, "grounding": 4, "originality": 2, "specificity": 4, "naturalness": 4, "score": 1, "features": text_features("A plain office line.")}
    assert classify(chosen, rejected, args) == "H1"
    rejected["grounding"] = 1
    assert classify(chosen, rejected, args) != "H1"


def test_length_and_module_parsing() -> None:
    a = {"features": text_features("one two three")}
    b = {"features": text_features("one two four")}
    assert close_length(a, b, 0.1)
    name = "base_model.model.model.language_model.layers.17.self_attn.o_proj.lora_A.default.weight"
    assert module_key(name).endswith("layers.17.self_attn.o_proj")
    assert module_identity(module_key(name)) == (17, "o_proj")


def test_probe_layer_selection_and_auc() -> None:
    assert probe_layers(29) == {"early": 7, "middle": 14, "late": 21, "final": 28}
    target = __import__("torch").tensor([0, 0, 1, 1], dtype=__import__("torch").bool)
    score = __import__("torch").tensor([0.1, 0.2, 0.8, 0.9])
    assert binary_auc(target, score) == pytest.approx(1.0)


def test_image_shuffle_limit_selects_distinct_images() -> None:
    rows = [{"image_id": "a", "v": 1}, {"image_id": "a", "v": 2}, {"image_id": "b", "v": 3}]
    selected = select_image_diverse_rows(rows, 2)
    assert [(row["image_id"], row["v"]) for row in selected] == [("a", 1), ("b", 3)]


def test_candidate_generation_is_chunked() -> None:
    torch = __import__("torch")

    class Batch(dict):
        def to(self, _device):
            return self

    class Processor:
        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, **_kwargs):
            return Batch(input_ids=torch.tensor([[1, 2, 3]]))

        def batch_decode(self, tokens, **_kwargs):
            return [f"caption {int(row[-1])}" for row in tokens]

    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = []
            self.cursor = 10

        def generate(self, input_ids, num_return_sequences, **_kwargs):
            self.calls.append(num_return_sequences)
            rows = []
            for _ in range(num_return_sequences):
                rows.append([*input_ids[0].tolist(), self.cursor])
                self.cursor += 1
            return torch.tensor(rows)

    model = Model()
    result = generate_candidates(
        model=model,
        processor=Processor(),
        process_vision_info=lambda _messages: ([], None),
        image_path="unused.jpg",
        prompt="instruction",
        generation_config={"do_sample": True, "candidate_batch_size": 2},
        num_candidates=5,
    )
    assert model.calls == [2, 2, 1]
    assert result == ["caption 10", "caption 11", "caption 12", "caption 13", "caption 14"]


def test_candidate_judge_rejects_silently_omitted_scores() -> None:
    judgment = {
        "best_index": 1,
        "candidates": [
            {
                "index": 1,
                "image_specific": 4,
                "naturalness": 4,
                "humor": 4,
                "format": 5,
                "overall": 4,
                "reason": "specific joke",
            }
        ],
    }
    with pytest.raises(ValueError, match="omitted candidate index 2"):
        normalize_judgment(judgment, ["first", "second"])


def test_best_of_n_humor_judge_requires_complete_one_to_five_scores() -> None:
    payload = {"scores": [{"index": 1, "humor": 4, "grounding": 5}]}
    assert normalize_scores(payload, 1) == [{"index": 1, "humor": 4, "grounding": 5}]
    with pytest.raises(ValueError, match="candidate indices must be exactly"):
        normalize_scores(payload, 2)
    with pytest.raises(ValueError, match=r"outside \[1, 5\]"):
        normalize_scores({"scores": [{"index": 1, "humor": 0, "grounding": 5}]}, 1)
    zero_based = {
        "scores": [
            {"index": 0, "humor": 2, "grounding": 3},
            {"index": 1, "humor": 4, "grounding": 5},
        ]
    }
    assert normalize_scores(zero_based, 2) == [
        {"index": 1, "humor": 2, "grounding": 3},
        {"index": 2, "humor": 4, "grounding": 5},
    ]
    offset = {
        "scores": [
            {"index": 2, "humor": 2, "grounding": 3},
            {"index": 3, "humor": 4, "grounding": 4},
        ]
    }
    assert normalize_scores(offset, 2) == [
        {"index": 1, "humor": 2, "grounding": 3},
        {"index": 2, "humor": 4, "grounding": 4},
    ]
    direct_list = """```json
[{"index": 1, "humor": 3, "grounding": 4}]
```"""
    assert parse_score_payload(direct_list) == {
        "scores": [{"index": 1, "humor": 3, "grounding": 4}]
    }


def test_paired_best_of_n_comparison_uses_matched_images() -> None:
    left = [
        {"image_id": "a", "scores": [1, 4]},
        {"image_id": "b", "scores": [2, 3]},
    ]
    right = [
        {"image_id": "a", "scores": [1, 2]},
        {"image_id": "b", "scores": [2, 2]},
    ]
    rows = paired_compare(left, right, (1, 2), "humor", 4, 100, 7)
    assert rows[0]["delta_hmax"] == pytest.approx(0)
    assert rows[1]["delta_hmax"] == pytest.approx(1.5)
    assert rows[1]["delta_pgood"] == pytest.approx(0.5)
