from __future__ import annotations

import pytest
import torch
from torch import nn

from humor_generator_v3.latent.state_capture import (
    DecodeStateCapture,
    assert_causal_replay_alignment,
    assert_hook_matches_teacher_forcing,
    causal_prediction_positions,
    teacher_forced_prediction_states,
)
from humor_generator_v3.training.memory_safe import inject_latent_slots


def test_causal_positions_have_no_off_by_one() -> None:
    assert causal_prediction_positions(3, 3).tolist() == [2, 3, 4]


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
