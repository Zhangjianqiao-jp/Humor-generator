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


def test_attention_normalizes_each_channel_before_channel_fusion() -> None:
    torch.manual_seed(7)
    bridge = ReceiverDrivenCrossAttentionBridge(
        16, 16, layer_indices=[0], bottleneck_dim=8, heads=2
    )
    model = _Core(16, 1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    uneven = {
        "conflict": torch.randn(2, 3, 16),
        "local": torch.randn(2, 17, 16),
        "global": torch.randn(2, 41, 16),
    }
    with bridge.inject(model, uneven):
        model(torch.randn(2, 5, 16))
    weights = bridge.last_diagnostics[0].channel_weights
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-5
    assert all(0.0 < value < 1.0 for value in weights)


def test_phase_a3_fixed_fusion_cannot_drop_a_channel() -> None:
    torch.manual_seed(9)
    bridge = ReceiverDrivenCrossAttentionBridge(
        16, 16, layer_indices=[0], bottleneck_dim=8, heads=2,
        channel_fusion="fixed_equal",
    )
    model = _Core(16, 1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with bridge.inject(model, {
        "conflict": torch.randn(2, 2, 16),
        "local": torch.randn(2, 19, 16),
        "global": torch.randn(2, 53, 16),
    }):
        model(torch.randn(2, 5, 16))
    torch.testing.assert_close(
        torch.tensor(bridge.last_diagnostics[0].channel_weights),
        torch.full((3,), 1 / 3), atol=1e-6, rtol=1e-6,
    )
    assert not bridge.layers["0"].channel_score.weight.requires_grad


def test_alignment_representations_preserve_channel_identity_and_gradients() -> None:
    bridge = ReceiverDrivenCrossAttentionBridge(
        16, 16, layer_indices=[0, 1], bottleneck_dim=8, heads=2
    )
    states = _states(batch=4)
    receiver = _states(batch=4)
    student, teacher = bridge.alignment_representations(states, receiver)
    assert student.shape == teacher.shape == (4, 2 * 3 * 8)
    loss, _ = symmetric_info_nce(student, teacher)
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in bridge.parameters()
    )
    channel_pairs = bridge.alignment_representations_by_channel(states, receiver)
    assert set(channel_pairs) == set(CHANNELS)
    assert all(pair[0].shape == pair[1].shape == (4, 2 * 8) for pair in channel_pairs.values())


def test_info_nce_teacher_projection_is_stationary() -> None:
    bridge = ReceiverDrivenCrossAttentionBridge(
        16, 16, layer_indices=[0], bottleneck_dim=8, heads=2
    )
    states = _states(batch=4)
    receiver = _states(batch=4)
    _, teacher_before = bridge.alignment_representations(states, receiver)
    with torch.no_grad():
        bridge.layers["0"].query.weight.add_(1.0)
    _, teacher_after = bridge.alignment_representations(states, receiver)
    torch.testing.assert_close(teacher_before, teacher_after)


def test_hierarchical_attention_is_invariant_to_duplicate_memory_length() -> None:
    """Duplicating identical tokens must not create channel mass by length alone."""
    torch.manual_seed(11)
    bridge = ReceiverDrivenCrossAttentionBridge(
        16, 16, layer_indices=[0], bottleneck_dim=8, heads=2
    )
    model = _Core(16, 1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    token = torch.randn(2, 1, 16)
    states = {
        "conflict": token.expand(-1, 3, -1).clone(),
        "local": token.expand(-1, 17, -1).clone(),
        "global": token.expand(-1, 41, -1).clone(),
    }
    # Remove type offsets for this strict length-only control.
    with torch.no_grad():
        bridge.channel_types.zero_()
    with bridge.inject(model, states):
        model(torch.randn(2, 5, 16))
    weights = bridge.last_diagnostics[0].channel_weights
    torch.testing.assert_close(
        torch.tensor(weights), torch.full((3,), 1 / 3), atol=1e-5, rtol=1e-5
    )


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
