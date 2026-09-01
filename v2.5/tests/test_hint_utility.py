import pytest

from src.preference.hint_utility import build_hint_pairs, summarize_hint_rows


def hint(hint_id: str, values: list[tuple[float, float, float, float]]):
    return {
        "image_id": "image-1",
        "image": "image.jpg",
        "hint_id": hint_id,
        "hint": f"hint {hint_id}",
        "judged_candidates": [
            {"candidate": f"caption-{index}", "humor": h, "grounding": g, "originality": o, "specificity": s}
            for index, (h, g, o, s) in enumerate(values)
        ],
    }


def test_hint_utility_averages_multiple_caption_samples() -> None:
    rows = [hint("a", [(5, 4, 3, 2), (3, 4, 3, 2)])]
    summaries = summarize_hint_rows(rows, {"humor": 0.5, "grounding": 0.5})
    assert summaries[0]["caption_samples"] == 2
    assert summaries[0]["hint_utility"] == pytest.approx(4.0)


def test_hint_pairs_use_margin_and_pareto_gate() -> None:
    rows = [
        hint("strong", [(5, 5, 5, 5), (4, 5, 5, 5)]),
        hint("weak", [(2, 3, 3, 3), (2, 3, 3, 3)]),
    ]
    summaries = summarize_hint_rows(rows, {"humor": 0.4, "grounding": 0.25, "originality": 0.2, "specificity": 0.15})
    pairs = build_hint_pairs(summaries, min_margin=0.5, require_dominance=True)
    assert len(pairs) == 1
    assert pairs[0]["chosen_hint_id"] == "strong"
    assert len(pairs[0]["chosen_generated_captions"]) == 2


def test_single_caption_is_rejected_as_noisy_hint_utility() -> None:
    with pytest.raises(ValueError, match="at least two"):
        summarize_hint_rows([hint("a", [(5, 5, 5, 5)])], {"humor": 1.0})
