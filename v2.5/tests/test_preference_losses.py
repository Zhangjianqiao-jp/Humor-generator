import pytest
import torch

from src.preference.losses import preference_loss


def tensors():
    return {
        "chosen_logp": torch.tensor([-4.0, -3.0]),
        "rejected_logp": torch.tensor([-6.0, -5.0]),
        "chosen_tokens": torch.tensor([2, 2]),
        "rejected_tokens": torch.tensor([3, 2]),
        "reference_chosen_logp": torch.tensor([-4.5, -3.5]),
        "reference_rejected_logp": torch.tensor([-5.5, -4.5]),
        "beta": 0.1,
    }


@pytest.mark.parametrize("objective", ["dpo", "ipo", "anchored"])
def test_reference_objectives_are_finite(objective: str) -> None:
    output = preference_loss(objective=objective, **tensors())
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.preference_logits.shape == (2,)


def test_simpo_uses_length_normalized_logps_without_reference() -> None:
    values = tensors()
    values.pop("reference_chosen_logp")
    values.pop("reference_rejected_logp")
    output = preference_loss(objective="simpo", simpo_gamma=0.5, **values)
    expected = 0.1 * (torch.tensor([-2.0, -1.5]) - torch.tensor([-2.0, -2.5])) - 0.5
    assert torch.allclose(output.preference_logits, expected)


def test_anchoring_penalizes_low_chosen_probability() -> None:
    baseline = preference_loss(objective="dpo", **tensors()).loss
    anchored = preference_loss(objective="anchored", anchor_weight=0.2, **tensors()).loss
    assert anchored > baseline


def test_reference_objective_rejects_missing_reference() -> None:
    values = tensors()
    values.pop("reference_chosen_logp")
    values.pop("reference_rejected_logp")
    with pytest.raises(ValueError, match="requires frozen reference"):
        preference_loss(objective="dpo", **values)
