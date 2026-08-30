from scripts.select_sft_module_pilot import select


def test_select_uses_lowest_eval_loss() -> None:
    rows = [
        {"name": "attention", "eval_loss": 1.2},
        {"name": "mlp", "eval_loss": 1.1},
        {"name": "all_linear", "eval_loss": 1.3},
    ]
    assert select(rows)["name"] == "mlp"


def test_select_tie_break_is_deterministic() -> None:
    rows = [
        {"name": "mlp", "eval_loss": 1.0},
        {"name": "attention", "eval_loss": 1.0},
    ]
    assert select(rows)["name"] == "attention"
