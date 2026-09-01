from scripts.compare_dpo_validation_gate import compare


def _metrics(loss: float, accuracy: float, margin: float, chosen: float = -42.0) -> dict:
    return {
        "eval_loss": loss,
        "eval_reward_accuracy": accuracy,
        "eval_reward_margin": margin,
        "eval_chosen_logp": chosen,
    }


def test_gate_passes_clear_validation_improvement() -> None:
    report = compare(_metrics(0.6904, 0.58, 0.0057), _metrics(0.6880, 0.60, 0.0070, -42.05))
    assert report["decision"] == "GO_FULL_QUALITY64"
    assert all(report["checks"].values())


def test_gate_rejects_loss_only_tiny_change() -> None:
    report = compare(_metrics(0.6904, 0.58, 0.0057), _metrics(0.6900, 0.60, 0.0070))
    assert report["decision"] == "NO_GO_FULL_QUALITY64"


def test_gate_rejects_chosen_probability_collapse() -> None:
    report = compare(_metrics(0.6904, 0.58, 0.0057), _metrics(0.6880, 0.60, 0.0070, -42.2))
    assert report["decision"] == "NO_GO_FULL_QUALITY64"
