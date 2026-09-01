"""OOM-safe contracts for frozen-receiver bridge optimization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class FreezeReport:
    policy_trainable: int
    bridge_trainable: int
    gradient_checkpointing: bool
    use_cache: bool


def configure_frozen_receiver(model: Any, bridge: torch.nn.Module) -> FreezeReport:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = False
    policy_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    bridge_trainable = sum(parameter.numel() for parameter in bridge.parameters() if parameter.requires_grad)
    if policy_trainable:
        raise RuntimeError("receiver policy must remain frozen")
    if bridge_trainable == 0:
        raise RuntimeError("bridge has no trainable parameters")
    return FreezeReport(policy_trainable, bridge_trainable, True, False)


def inject_latent_slots(
    *,
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    slots: torch.Tensor,
    insertion_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert bridge slots at a validated prompt/caption boundary."""
    if token_embeddings.ndim != 3 or slots.ndim != 3:
        raise ValueError("embeddings and slots must both be [B,T,D]")
    if token_embeddings.shape[0] != slots.shape[0] or token_embeddings.shape[2] != slots.shape[2]:
        raise ValueError("slot batch/width must match token embeddings")
    if attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("attention mask does not match token embeddings")
    if not 0 < insertion_index < token_embeddings.shape[1]:
        raise ValueError("insertion index must lie strictly inside the token sequence")
    slot_mask = torch.ones(
        slots.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
    )
    return (
        torch.cat(
            [token_embeddings[:, :insertion_index], slots, token_embeddings[:, insertion_index:]],
            dim=1,
        ),
        torch.cat(
            [attention_mask[:, :insertion_index], slot_mask, attention_mask[:, insertion_index:]],
            dim=1,
        ),
    )


def caption_only_logits(
    model: Any,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    caption_tokens: int,
    **kwargs: Any,
) -> torch.Tensor:
    """Avoid allocating vocabulary logits for visual/prompt prefix tokens."""
    if caption_tokens < 1:
        raise ValueError("caption_tokens must be positive")
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=None,
        use_cache=False,
        logits_to_keep=caption_tokens + 1,
        **kwargs,
    )
    logits = outputs.logits
    if logits.shape[1] < caption_tokens + 1:
        raise RuntimeError("receiver did not return enough causal logits")
    return logits[:, -caption_tokens - 1 : -1, :]


@torch.no_grad()
def cache_text_teacher_logits(
    model: Any,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    caption_tokens: int,
    output_dtype: torch.dtype = torch.float16,
    **kwargs: Any,
) -> torch.Tensor:
    return caption_only_logits(
        model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        caption_tokens=caption_tokens,
        **kwargs,
    ).to(output_dtype).cpu()


def estimated_logits_bytes(*, batch: int, sequence: int, vocabulary: int, bytes_per_value: int = 2) -> int:
    return batch * sequence * vocabulary * bytes_per_value


def assert_smoke_budget(
    *,
    examples: int,
    optimizer_steps: int,
    max_sequence_tokens: int,
    formal_training_enabled: bool,
) -> None:
    if formal_training_enabled:
        raise RuntimeError("v3.0 is engineering-smoke-only until the scientific gates pass")
    if examples > 2 or optimizer_steps > 1 or max_sequence_tokens > 1024:
        raise RuntimeError("engineering smoke is limited to <=2 examples, one step, <=1024 tokens")
