"""Caption, text-teacher distillation, and shuffled-message losses."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits and targets must agree before the vocabulary dimension")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=ignore_index)


def text_teacher_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape or mask.shape != student_logits.shape[:-1]:
        raise ValueError("KL tensors have incompatible shapes")
    if not mask.any():
        raise ValueError("KL mask has no supervised tokens")
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    teacher_p = teacher_logp.exp()
    per_token = (teacher_p * (teacher_logp - student_logp)).sum(-1) * temperature**2
    return per_token[mask].mean()


def sequence_log_probability(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits and targets must agree before vocabulary")
    mask = targets.ne(ignore_index)
    safe_targets = targets.masked_fill(~mask, 0)
    selected = F.log_softmax(logits.float(), dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def matched_shuffled_margin_loss(
    matched_logits: torch.Tensor,
    shuffled_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    matched = sequence_log_probability(matched_logits, targets)
    shuffled = sequence_log_probability(shuffled_logits, targets)
    return F.softplus(-(matched - shuffled - margin)).mean()


def symmetric_info_nce(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align samples without forcing coordinate equality.

    Returns symmetric InfoNCE and retrieval@1.  A batch of one is rejected:
    without negatives the objective cannot constrain semantic identity.
    """
    if student.ndim != 2 or teacher.ndim != 2 or student.shape != teacher.shape:
        raise ValueError("student and teacher must be matching [B,D] tensors")
    if student.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two matched pairs")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    left = F.normalize(student.float(), dim=-1)
    right = F.normalize(teacher.detach().float(), dim=-1)
    logits = left @ right.T / temperature
    labels = torch.arange(student.shape[0], device=student.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    accuracy = 0.5 * (
        logits.argmax(-1).eq(labels).float().mean()
        + logits.argmax(0).eq(labels).float().mean()
    )
    return loss, accuracy


def variance_floor_loss(representations: torch.Tensor, *, floor: float = 1.0) -> torch.Tensor:
    """VICReg-style anti-collapse term over semantically distinct samples."""
    if representations.ndim != 2 or representations.shape[0] < 2:
        raise ValueError("variance loss requires at least two [B,D] representations")
    if floor <= 0:
        raise ValueError("floor must be positive")
    standard_deviation = torch.sqrt(representations.float().var(dim=0) + 1e-4)
    return torch.relu(floor - standard_deviation).mean()


@dataclass(frozen=True)
class BridgeLoss:
    total: torch.Tensor
    caption_nll: torch.Tensor
    teacher_kl: torch.Tensor
    shuffled_margin: torch.Tensor


def bridge_objective(
    *,
    matched_logits: torch.Tensor,
    shuffled_logits: torch.Tensor,
    text_teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    kl_mask: torch.Tensor,
    caption_weight: float = 1.0,
    kl_weight: float = 0.5,
    shuffled_weight: float = 0.1,
    shuffled_margin: float = 0.2,
    temperature: float = 1.0,
) -> BridgeLoss:
    caption = token_cross_entropy(matched_logits, targets)
    kl = text_teacher_forward_kl(matched_logits, text_teacher_logits, kl_mask, temperature=temperature)
    shuffled = matched_shuffled_margin_loss(
        matched_logits, shuffled_logits, targets, margin=shuffled_margin
    )
    total = caption_weight * caption + kl_weight * kl + shuffled_weight * shuffled
    return BridgeLoss(total, caption, kl, shuffled)
