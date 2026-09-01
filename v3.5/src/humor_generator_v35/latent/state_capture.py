"""Causally explicit hidden-state/token alignment.

For a generated sequence y_0..y_{N-1}, HF generation captures the state used
to predict each token: prompt[-1], y_0, ..., y_{N-2].  The teacher-forced
reference positions are therefore [prompt_len-1, prompt_len+N-2].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def causal_prediction_positions(prompt_length: int, generated_length: int) -> torch.Tensor:
    if prompt_length < 1 or generated_length < 1:
        raise ValueError("prompt_length and generated_length must be positive")
    return torch.arange(prompt_length - 1, prompt_length + generated_length - 1)


@dataclass(frozen=True)
class AlignedMessageStates:
    token_ids: torch.Tensor
    states: torch.Tensor
    semantics: str = "unspecified"

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 2 or self.states.ndim != 3:
            raise ValueError("token_ids must be [B,T] and states [B,T,D]")
        if self.token_ids.shape[:2] != self.states.shape[:2]:
            raise ValueError("one hidden state is required for every generated token")


@dataclass(frozen=True)
class ReplayAlignmentReport:
    mean_cosine: float
    min_cosine: float
    relative_l2: float


class DecodeStateCapture:
    """Capture exactly one predictive state per generation forward call."""

    def __init__(self, output_device: str | torch.device = "cpu") -> None:
        self.output_device = torch.device(output_device)
        self.states: list[torch.Tensor] = []

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if isinstance(hidden, torch.Tensor) and hidden.ndim == 3:
            self.states.append(hidden[:, -1:, :].detach().to(self.output_device))

    def align(self, token_ids: torch.Tensor) -> AlignedMessageStates:
        if not self.states:
            raise RuntimeError("no decoder states were captured")
        states = torch.cat(self.states, dim=1)
        if states.shape[1] != token_ids.shape[1]:
            raise RuntimeError(
                f"decode-state/token mismatch: states={states.shape[1]} tokens={token_ids.shape[1]}; "
                "trimming is forbidden"
            )
        return AlignedMessageStates(token_ids.detach().cpu(), states)


class SequenceStateCapture:
    """Capture a complete last-layer sequence from one teacher-forced call."""

    def __init__(self) -> None:
        self.value: torch.Tensor | None = None

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("decoder layer did not return [B,T,D] states")
        self.value = hidden.detach()

    def require(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("teacher-forced decoder states were not captured")
        return self.value


def teacher_forced_prediction_states(
    final_hidden: torch.Tensor,
    *,
    prompt_length: int,
    generated_token_ids: torch.Tensor,
) -> AlignedMessageStates:
    positions = causal_prediction_positions(prompt_length, generated_token_ids.shape[1]).to(final_hidden.device)
    if int(positions[-1]) >= final_hidden.shape[1]:
        raise ValueError("final_hidden does not cover every causal prediction position")
    states = final_hidden.index_select(1, positions).detach().cpu()
    return AlignedMessageStates(generated_token_ids.detach().cpu(), states)


def teacher_forced_token_states(
    final_hidden: torch.Tensor,
    *,
    prompt_length: int,
    generated_token_ids: torch.Tensor,
) -> AlignedMessageStates:
    """Return each emitted token's own post-token state, not its predictor."""
    if prompt_length < 1 or generated_token_ids.shape[1] < 1:
        raise ValueError("prompt and generated sequence must be non-empty")
    positions = torch.arange(
        prompt_length,
        prompt_length + generated_token_ids.shape[1],
        device=final_hidden.device,
    )
    if int(positions[-1]) >= final_hidden.shape[1]:
        raise ValueError("final_hidden does not cover every emitted token position")
    states = final_hidden.index_select(1, positions).detach().cpu()
    return AlignedMessageStates(generated_token_ids.detach().cpu(), states)


def assert_hook_matches_teacher_forcing(
    hook: AlignedMessageStates,
    teacher: AlignedMessageStates,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    if not torch.equal(hook.token_ids, teacher.token_ids):
        raise AssertionError("hook and teacher-forced token ids differ")
    torch.testing.assert_close(hook.states.float(), teacher.states.float(), atol=atol, rtol=rtol)


def assert_causal_replay_alignment(
    hook: AlignedMessageStates,
    teacher: AlignedMessageStates,
    *,
    min_token_cosine: float = 0.90,
    min_mean_cosine: float = 0.98,
) -> ReplayAlignmentReport:
    """Check causal positions without demanding kernel-level bit agreement.

    Cached one-token decoding and full-sequence SDPA are different numerical
    execution paths, especially under NF4/BF16.  Directional agreement is the
    appropriate replay invariant; exact token prediction is checked separately
    through the model's final norm and output head.
    """
    if not torch.equal(hook.token_ids, teacher.token_ids):
        raise AssertionError("hook and teacher-forced token ids differ")
    left, right = hook.states.float(), teacher.states.float()
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1)
    mean_cosine = float(cosine.mean())
    min_cosine = float(cosine.min())
    relative_l2 = float(
        (left - right).norm(dim=-1).div(right.norm(dim=-1).clamp_min(1e-8)).mean()
    )
    if min_cosine < min_token_cosine or mean_cosine < min_mean_cosine:
        raise AssertionError(
            "cached decode and teacher replay disagree at causal positions: "
            f"mean_cosine={mean_cosine:.6f}, min_cosine={min_cosine:.6f}, "
            f"relative_l2={relative_l2:.6f}"
        )
    return ReplayAlignmentReport(mean_cosine, min_cosine, relative_l2)
