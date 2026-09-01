from __future__ import annotations

import pytest
import torch
from torch import nn

from humor_generator_v35.latent.state_capture import (
    DecodeStateCapture,
    assert_causal_replay_alignment,
    assert_hook_matches_teacher_forcing,
    causal_prediction_positions,
    teacher_forced_prediction_states,
    teacher_forced_token_states,
)
from humor_generator_v35.latent.statebridge import StateBridgeAlignment
from humor_generator_v35.training.memory_safe import inject_latent_slots


def test_causal_positions_have_no_off_by_one() -> None:
    assert causal_prediction_positions(3, 3).tolist() == [2, 3, 4]


def test_post_token_communication_states_include_the_emitted_token() -> None:
    generated = torch.tensor([[4, 5]])
    hidden = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0]]])
    states = teacher_forced_token_states(
        hidden, prompt_length=3, generated_token_ids=generated
    )
    assert states.states.squeeze().tolist() == [3.0, 4.0]


def test_hook_matches_teacher_forced_predictive_states() -> None:
    torch.manual_seed(4)
    embedding = nn.Embedding(20, 6)
    layer = nn.Linear(6, 6, bias=False)
    prompt = torch.tensor([[1, 2, 3]])
    generated = torch.tensor([[4, 5, 6]])
    capture = DecodeStateCapture()
    handle = layer.register_forward_hook(capture)
    layer(embedding(prompt))
    layer(embedding(generated[:, :1]))
    layer(embedding(generated[:, 1:2]))
    handle.remove()
    hook = capture.align(generated)
    teacher = teacher_forced_prediction_states(
        layer(embedding(torch.cat([prompt, generated], 1))),
        prompt_length=3,
        generated_token_ids=generated,
    )
    assert_hook_matches_teacher_forcing(hook, teacher)
    report = assert_causal_replay_alignment(hook, teacher)
    assert report.mean_cosine == pytest.approx(1.0)


def test_replay_alignment_rejects_shifted_states() -> None:
    token_ids = torch.tensor([[1, 2]])
    hook = teacher_forced_prediction_states(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]),
        prompt_length=1,
        generated_token_ids=token_ids,
    )
    shifted = type(hook)(token_ids, torch.flip(hook.states, dims=[1]))
    with pytest.raises(AssertionError):
        assert_causal_replay_alignment(hook, shifted)


def test_mismatch_is_rejected_not_trimmed() -> None:
    capture = DecodeStateCapture()
    capture.states = [torch.zeros(1, 1, 4)]
    with pytest.raises(RuntimeError):
        capture.align(torch.tensor([[1, 2]]))


def test_latent_slots_are_inserted_at_caption_boundary() -> None:
    tokens = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.long)
    slots = torch.full((1, 2, 3), -1.0)
    combined, combined_mask = inject_latent_slots(
        token_embeddings=tokens,
        attention_mask=mask,
        slots=slots,
        insertion_index=2,
    )
    assert combined.shape == (1, 6, 3)
    torch.testing.assert_close(combined[:, :2], tokens[:, :2])
    torch.testing.assert_close(combined[:, 2:4], slots)
    torch.testing.assert_close(combined[:, 4:], tokens[:, 2:])
    assert combined_mask.tolist() == [[1, 1, 1, 1, 1, 1]]


def test_low_rank_statebridge_matches_dense_transform_action() -> None:
    torch.manual_seed(12)
    tokens, width, vocab = 5, 11, 19
    embeddings = torch.randn(vocab, width)
    hidden = torch.randn(1, tokens, width)
    token_ids = torch.tensor([[1, 3, 5, 7, 9]])
    module = StateBridgeAlignment(embeddings, regularization=0.07, snap_ratio=0.0)
    actual = module(hidden, token_ids).float()

    h = hidden.float().reshape(tokens, width)
    e = embeddings[token_ids].float().reshape(tokens, width)
    mean_h, mean_e = h.mean(0, keepdim=True), e.mean(0, keepdim=True)
    hc, ec = h - mean_h, e - mean_e
    eye = torch.eye(width)
    cov_h = hc.T @ hc / tokens + 0.07 * eye
    cov_e = ec.T @ ec / tokens + 0.07 * eye
    inv_h = module._symmetric_power(cov_h, -0.5)
    inv_e = module._symmetric_power(cov_e, -0.5)
    sqrt_e = module._symmetric_power(cov_e, 0.5)
    white_h, white_e = hc @ inv_h, ec @ inv_e
    u, _, vh = torch.linalg.svd(white_h.T @ white_e, full_matrices=False)
    rotation = u @ vh
    if torch.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    expected = white_h @ rotation @ sqrt_e + mean_e
    expected = expected.reshape_as(actual)
    expected = expected * (
        module.target_norm / expected.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    )
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_statebridge_preserves_dense_fallback_for_rank_mismatch() -> None:
    torch.manual_seed(13)
    embeddings = torch.randn(7, 9)
    hidden = torch.randn(1, 5, 9)
    repeated_ids = torch.tensor([[1, 1, 1, 2, 2]])
    output = StateBridgeAlignment(embeddings, snap_ratio=0.0)(hidden, repeated_ids)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
