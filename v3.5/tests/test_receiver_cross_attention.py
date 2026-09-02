from __future__ import annotations

import torch
from torch import nn

from humor_generator_v35.latent.cross_attention import (
    CHANNELS,
    ReceiverDrivenCrossAttentionBridge,
)
from humor_generator_v35.training.losses import symmetric_info_nce, variance_floor_loss


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.linear(hidden),)


class _Core(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(width) for _ in range(layers)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


def _states(batch: int = 2, width: int = 16) -> dict[str, torch.Tensor]:
    return {
        name: torch.randn(batch, index + 3, width)
        for index, name in enumerate(CHANNELS)
    }


def test_full_typed_memory_is_not_tail_truncated() -> None:
    bridge = ReceiverDrivenCrossAttentionBridge(
        16, 16, layer_indices=[0, 2], bottleneck_dim=8, heads=2
    )
    states = _states()
    memory, mask = bridge.pack_memory(states)
    assert memory.shape == (2, sum(value.shape[1] for value in states.values()), 16)
    assert mask.all()


def test_hook_is_temporary_and_bridge_receives_gradients() -> None:
    torch.manual_seed(3)
    model = _Core(16, 3)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bridge = ReceiverDrivenCrossAttentionBridge(
        16, 16, layer_indices=[0, 2], bottleneck_dim=8, heads=2, gate_init=0.1
    )
    hidden = torch.randn(2, 5, 16)
    baseline = model(hidden)
    with bridge.inject(model, _states()):
        conditioned = model(hidden)
    restored = model(hidden)
    assert not torch.allclose(baseline, conditioned)
    assert torch.allclose(baseline, restored)
    conditioned.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in bridge.parameters()
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert {item.layer for item in bridge.last_diagnostics} == {0, 2}


def test_fp32_bridge_preserves_bfloat16_receiver_interface() -> None:
    model = _Core(16, 2).to(dtype=torch.bfloat16)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bridge = ReceiverDrivenCrossAttentionBridge(
        16, 16, layer_indices=[0], bottleneck_dim=8, heads=2, gate_init=0.1
    )
    hidden = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    with bridge.inject(model, _states()):
        conditioned = model(hidden)
    assert conditioned.dtype == torch.bfloat16
    conditioned.float().square().mean().backward()
    assert any(parameter.grad is not None for parameter in bridge.parameters())


def test_semantic_constraints_reject_degenerate_batch_and_are_finite() -> None:
    student = torch.randn(4, 12, requires_grad=True)
    teacher = torch.randn(4, 12)
    loss, accuracy = symmetric_info_nce(student, teacher)
    anti_collapse = variance_floor_loss(student)
    (loss + anti_collapse).backward()
    assert torch.isfinite(loss)
    assert 0 <= float(accuracy) <= 1
    assert student.grad is not None
