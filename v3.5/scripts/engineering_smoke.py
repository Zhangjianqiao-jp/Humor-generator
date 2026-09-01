#!/usr/bin/env python3
"""CPU one-step smoke for state alignment and all trainable bridge losses."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.latent.bridges import LearnedLatentBridge
from humor_generator_v35.latent.state_capture import (
    AlignedMessageStates,
    DecodeStateCapture,
    assert_hook_matches_teacher_forcing,
    teacher_forced_prediction_states,
)
from humor_generator_v35.training.losses import bridge_objective
from humor_generator_v35.training.memory_safe import assert_smoke_budget


class ToyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def alignment_smoke() -> None:
    torch.manual_seed(7)
    embedding = nn.Embedding(32, 8)
    block = ToyBlock(8)
    prompt = torch.tensor([[2, 3, 4]])
    generated = torch.tensor([[5, 6, 7]])
    capture = DecodeStateCapture()
    handle = block.register_forward_hook(capture)
    block(embedding(prompt))
    block(embedding(generated[:, :1]))
    block(embedding(generated[:, 1:2]))
    handle.remove()
    hook = capture.align(generated)
    full = block(embedding(torch.cat([prompt, generated], dim=1)))
    teacher = teacher_forced_prediction_states(full, prompt_length=prompt.shape[1], generated_token_ids=generated)
    assert_hook_matches_teacher_forcing(hook, teacher)


def optimizer_smoke() -> dict[str, float]:
    torch.manual_seed(11)
    bridge = LearnedLatentBridge(12, 16, bottleneck_dim=8, slots=3, heads=2)
    frozen_head = nn.Linear(16, 23)
    for parameter in frozen_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)
    states = torch.randn(2, 5, 12)
    shuffled_states = states.flip(0)
    targets = torch.randint(0, 23, (2, 3))
    matched = frozen_head(bridge(states))
    shuffled = frozen_head(bridge(shuffled_states))
    teacher = torch.randn_like(matched)
    loss = bridge_objective(
        matched_logits=matched,
        shuffled_logits=shuffled,
        text_teacher_logits=teacher,
        targets=targets,
        kl_mask=torch.ones_like(targets, dtype=torch.bool),
    )
    loss.total.backward()
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in bridge.parameters()):
        raise RuntimeError("bridge gradient smoke failed")
    optimizer.step()
    return {
        "total": float(loss.total.detach()),
        "caption_nll": float(loss.caption_nll.detach()),
        "teacher_kl": float(loss.teacher_kl.detach()),
        "shuffled_margin": float(loss.shuffled_margin.detach()),
    }


def main() -> None:
    assert_smoke_budget(examples=2, optimizer_steps=1, max_sequence_tokens=128, formal_training_enabled=False)
    alignment_smoke()
    print(json.dumps({"status": "pass", "losses": optimizer_smoke()}, sort_keys=True))


if __name__ == "__main__":
    main()
