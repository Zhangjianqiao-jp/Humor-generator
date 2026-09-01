from __future__ import annotations

import pytest
import torch

from humor_generator_v35.latent.baselines import BASELINES, Baseline
from humor_generator_v35.latent.bridges import LearnedLatentBridge, TypedLatentBridge, mean_embedding_norm
from humor_generator_v35.latent.conditioning import learned_condition, statebridge_condition, typed_condition
from humor_generator_v35.latent.state_capture import AlignedMessageStates
from humor_generator_v35.latent.statebridge import StateBridgeAlignment
from humor_generator_v35.training.losses import bridge_objective
from humor_generator_v35.training.memory_safe import (
    assert_smoke_budget,
    estimated_logits_bytes,
    inject_position_ids,
)


def test_all_four_baselines_are_registered() -> None:
    assert set(BASELINES) == set(Baseline)


def test_typed_bridge_keeps_three_channels_and_gradients() -> None:
    bridge = TypedLatentBridge(12, 16, bottleneck_dim=8, slots=2, heads=2, target_norm=2.5)
    result = bridge({name: torch.randn(2, 4, 12) for name in ("conflict", "local", "global")})
    assert result["all"].shape == (2, 6, 16)
    torch.testing.assert_close(
        result["all"].float().norm(dim=-1),
        torch.full((2, 6), 2.5),
        atol=1e-5,
        rtol=1e-5,
    )
    result["all"].square().mean().backward()
    assert all(parameter.grad is not None for parameter in bridge.parameters())


def test_learned_and_typed_are_exactly_parameter_and_slot_matched() -> None:
    learned = LearnedLatentBridge(12, 16, bottleneck_dim=8, slots=6, heads=2)
    typed = TypedLatentBridge(12, 16, bottleneck_dim=8, slots=2, heads=2)
    learned_params = sum(value.numel() for value in learned.parameters() if value.requires_grad)
    typed_params = sum(value.numel() for value in typed.parameters() if value.requires_grad)
    assert learned_params == typed_params
    assert learned(torch.randn(1, 9, 12)).shape[1] == typed.total_slots == 6


def test_mean_embedding_norm_is_chunk_invariant() -> None:
    embeddings = torch.tensor([[3.0, 4.0], [0.0, 2.0], [8.0, 6.0]])
    assert mean_embedding_norm(embeddings, chunk_size=1) == pytest.approx(17 / 3)
    assert mean_embedding_norm(embeddings, chunk_size=8) == pytest.approx(17 / 3)


def test_three_latent_baselines_share_one_output_contract() -> None:
    states = {
        name: AlignedMessageStates(
            torch.tensor([[1, 2, 3]]),
            torch.randn(1, 3, 4),
        )
        for name in ("conflict", "local", "global")
    }
    learned = learned_condition(
        LearnedLatentBridge(4, 4, bottleneck_dim=4, slots=2, heads=1), states
    )
    typed = typed_condition(
        TypedLatentBridge(4, 4, bottleneck_dim=4, slots=2, heads=1), states
    )
    receiver_embeddings = torch.randn(8, 4)
    statebridge = statebridge_condition(
        StateBridgeAlignment(receiver_embeddings, snap_ratio=0.0), states["conflict"]
    )
    assert learned.baseline is Baseline.LEARNED and learned.latent_slots.shape == (1, 2, 4)
    assert typed.baseline is Baseline.TYPED and typed.latent_slots.shape == (1, 6, 4)
    assert statebridge.baseline is Baseline.STATEBRIDGE and statebridge.latent_slots.shape == (1, 3, 4)


def test_combined_loss_is_finite_and_trainable() -> None:
    bridge = LearnedLatentBridge(8, 10, bottleneck_dim=8, slots=3, heads=2)
    head = torch.nn.Linear(10, 17)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    matched = head(bridge(torch.randn(2, 5, 8)))
    shuffled = head(bridge(torch.randn(2, 5, 8)))
    targets = torch.randint(0, 17, (2, 3))
    loss = bridge_objective(
        matched_logits=matched,
        shuffled_logits=shuffled,
        text_teacher_logits=torch.randn_like(matched),
        targets=targets,
        kl_mask=torch.ones_like(targets, dtype=torch.bool),
    )
    loss.total.backward()
    assert torch.isfinite(loss.total)


def test_formal_or_large_smoke_is_blocked() -> None:
    with pytest.raises(RuntimeError):
        assert_smoke_budget(examples=3, optimizer_steps=1, max_sequence_tokens=128, formal_training_enabled=False)
    with pytest.raises(RuntimeError):
        assert_smoke_budget(examples=1, optimizer_steps=1, max_sequence_tokens=128, formal_training_enabled=True)
    assert estimated_logits_bytes(batch=1, sequence=1000, vocabulary=150000) > estimated_logits_bytes(
        batch=1, sequence=64, vocabulary=150000
    )


def test_latent_slots_shift_all_three_mrope_axes_at_caption_boundary() -> None:
    original = torch.tensor([
        [[0, 1, 2, 7, 8]],
        [[0, 1, 4, 7, 8]],
        [[0, 1, 6, 7, 8]],
    ])
    inserted = inject_position_ids(original, slot_count=2, insertion_index=3)
    assert inserted.shape == (3, 1, 7)
    torch.testing.assert_close(inserted[:, :, :3], original[:, :, :3])
    torch.testing.assert_close(inserted[:, :, 3:5], torch.tensor([[[7, 8]]] * 3))
    torch.testing.assert_close(inserted[:, :, 5:], original[:, :, 3:] + 2)
