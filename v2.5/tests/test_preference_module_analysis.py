from src.preference.module_analysis import (
    cumulative_selection,
    lora_parameter_count,
    matched_uniform_rank,
    rank_statistics,
)


def test_lora_parameter_count() -> None:
    assert lora_parameter_count(8, 12, 4) == 80


def test_rank_and_cumulative_selection_are_deterministic() -> None:
    rows = [
        {"module_path": "b", "adaptation_utility": 2.0},
        {"module_path": "a", "adaptation_utility": 2.0},
        {"module_path": "c", "adaptation_utility": 1.0},
    ]
    ranked = rank_statistics(rows)
    assert [row["module_path"] for row in ranked] == ["a", "b", "c"]
    assert [row["module_path"] for row in cumulative_selection(rows, 0.8)] == ["a", "b"]


def test_budget_matching_selects_closest_integer_rank() -> None:
    rank, parameters = matched_uniform_rank([(10, 20), (10, 20)], 250)
    assert rank == 4
    assert parameters == 240
