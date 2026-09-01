#!/usr/bin/env python3
"""One-example Qwen-7B bridge smoke; never a scientific training run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.homer.contracts import parse_conflicts
from humor_generator_v35.homer.prompts import caption_messages, conflict_messages, text_part
from humor_generator_v35.latent.bridges import TypedLatentBridge, mean_embedding_norm
from humor_generator_v35.qwen_backend import QwenBackend, model_device
from humor_generator_v35.training.losses import (
    sequence_log_probability,
    text_teacher_forward_kl,
    token_cross_entropy,
)
from humor_generator_v35.training.memory_safe import (
    assert_smoke_budget,
    cache_text_teacher_logits,
    caption_only_logits,
    configure_frozen_receiver,
    inject_latent_slots,
)


DESCRIPTION = (
    "A king sits on a throne while a servant reads a scroll. Above the king, "
    "a large sword hangs precariously by a single thread."
)
CONFLICTS = "1. royal safety vs. imminent danger 2. calm court routine vs. precarious lethal threat"
LOCAL_PATH = ["king", "royal authority", "executive overhead", "overhead costs"]
GLOBAL_PATH = ["hanging sword", "thread", "budget cut", "cost cutting"]
CAPTION = "Your overhead is going to kill you."


def _text_messages(system: str, user: str, *, assistant: str | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [
        {"role": "system", "content": [text_part(system)]},
        {"role": "user", "content": [text_part(user)]},
    ]
    if assistant is not None:
        result.append({"role": "assistant", "content": [text_part(assistant)]})
    return result


def _prompt_and_full(
    backend: QwenBackend,
    prompt_messages: list[dict[str, Any]],
    answer: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    prompt = backend.encode(prompt_messages, add_generation_prompt=True)
    full_messages = [*prompt_messages, {"role": "assistant", "content": [text_part(answer)]}]
    full = backend.encode(full_messages, add_generation_prompt=False)
    prompt_ids, full_ids = prompt["input_ids"], full["input_ids"]
    prompt_length = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_length:
        raise RuntimeError("assistant answer contains no causal target tokens")
    if not torch.equal(full_ids[:, :prompt_length], prompt_ids):
        raise RuntimeError("chat template prompt is not an exact prefix of the supervised sequence")
    targets = full_ids[:, prompt_length:].clone()
    return prompt, full, targets


def _receiver_width(model: Any) -> int:
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    width = getattr(text_config, "hidden_size", None)
    if width is None:
        width = model.get_input_embeddings().weight.shape[1]
    return int(width)


def _base_embeddings(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model.get_input_embeddings()(batch["input_ids"])


def _latent_logits(
    backend: QwenBackend,
    bridge: TypedLatentBridge,
    states: dict[str, torch.Tensor],
    full: dict[str, torch.Tensor],
    *,
    insertion_index: int,
    caption_tokens: int,
) -> torch.Tensor:
    embeddings = _base_embeddings(backend.model, full)
    slots = bridge(states)["all"].to(dtype=embeddings.dtype)
    combined, mask = inject_latent_slots(
        token_embeddings=embeddings,
        attention_mask=full["attention_mask"],
        slots=slots,
        insertion_index=insertion_index,
    )
    return caption_only_logits(
        backend.model,
        inputs_embeds=combined,
        attention_mask=mask,
        caption_tokens=caption_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    smoke = config["smoke"]
    assert_smoke_budget(
        examples=1,
        optimizer_steps=1,
        max_sequence_tokens=int(smoke["max_sequence_tokens"]),
        formal_training_enabled=bool(config["experiment"]["formal_training_enabled"]),
    )
    model_config = config["model"]
    adapter = model_config.get("adapter")
    adapter_path = (ROOT / adapter).resolve() if adapter else None
    if adapter_path is not None and not adapter_path.is_dir():
        raise FileNotFoundError(f"receiver adapter is missing: {adapter_path}")

    torch.manual_seed(args.seed)
    backend = QwenBackend.load(
        model_config["name"],
        revision=model_config.get("revision"),
        adapter=adapter_path,
        load_in_4bit=True,
    )
    device = model_device(backend.model)
    if device.type != "cuda":
        raise RuntimeError("GPU smoke requires a CUDA receiver")

    # Real-model causal alignment.  This is text-only because multimodal replay
    # needs processor-specific image tensors and is a separate contract.
    generated_conflicts, aligned, alignment_report = backend.generate_and_verify_states(
        conflict_messages(DESCRIPTION), max_new_tokens=96, seed=args.seed
    )
    parsed_conflicts = parse_conflicts(generated_conflicts)

    width = _receiver_width(backend.model)
    receiver_embedding_norm = mean_embedding_norm(backend.model.get_input_embeddings().weight)
    bridge_config = config["bridge"]
    bridge = TypedLatentBridge(
        width,
        width,
        bottleneck_dim=int(bridge_config["bottleneck_dim"]),
        slots=int(bridge_config["slots_per_channel"]),
        heads=int(bridge_config["heads"]),
        target_norm=receiver_embedding_norm,
    ).to(device)
    freeze = configure_frozen_receiver(backend.model, bridge)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-4)
    backend.model.train()
    for module in backend.model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    text_prompt = caption_messages(
        DESCRIPTION,
        CONFLICTS,
        [*LOCAL_PATH, *GLOBAL_PATH],
    )
    latent_prompt = _text_messages(
        "Generate one short New Yorker-style caption. A typed latent humor plan is inserted before the answer.",
        f"Cartoon description: {DESCRIPTION}",
    )
    text_prefix, text_full, text_targets = _prompt_and_full(backend, text_prompt, CAPTION)
    latent_prefix, latent_full, latent_targets = _prompt_and_full(backend, latent_prompt, CAPTION)
    if not torch.equal(text_targets, latent_targets):
        raise RuntimeError("text and latent receivers tokenize the target caption differently")
    caption_tokens = int(latent_targets.shape[1])
    max_length = max(int(text_full["input_ids"].shape[1]), int(latent_full["input_ids"].shape[1]))
    if max_length + 3 * int(bridge_config["slots_per_channel"]) > int(smoke["max_sequence_tokens"]):
        raise RuntimeError("actual smoke sequence exceeds preregistered token budget")

    with torch.no_grad():
        teacher_embeddings = _base_embeddings(backend.model, text_full)
        teacher_logits = cache_text_teacher_logits(
            backend.model,
            inputs_embeds=teacher_embeddings,
            attention_mask=text_full["attention_mask"],
            caption_tokens=caption_tokens,
        )

    source_length = 12
    matched_states = {
        name: torch.randn(1, source_length, width, device=device)
        for name in TypedLatentBridge.channel_order
    }
    shuffled_states = {
        name: torch.flip(value, dims=[1]) for name, value in matched_states.items()
    }
    insertion_index = int(latent_prefix["input_ids"].shape[1])
    loss_config = config["loss"]
    margin = float(loss_config["margin"])
    shuffled_weight = float(loss_config["matched_shuffled_margin"])
    targets = latent_targets

    torch.cuda.reset_peak_memory_stats(device)
    # Compute the margin derivative without retaining two 7B receiver graphs.
    # The following two-pass surrogate has exactly the same first derivative at
    # the current parameters as softplus(-matched+shuffled+margin).
    with torch.no_grad():
        matched0 = sequence_log_probability(
            _latent_logits(
                backend, bridge, matched_states, latent_full,
                insertion_index=insertion_index, caption_tokens=caption_tokens,
            ),
            targets,
        )
        shuffled0 = sequence_log_probability(
            _latent_logits(
                backend, bridge, shuffled_states, latent_full,
                insertion_index=insertion_index, caption_tokens=caption_tokens,
            ),
            targets,
        )
        margin_coefficient = torch.sigmoid(-matched0 + shuffled0 + margin).detach()
        exact_margin_value = F.softplus(-matched0 + shuffled0 + margin).mean()

    optimizer.zero_grad(set_to_none=True)
    first_parameter = next(bridge.parameters())
    first_parameter_before = first_parameter.detach().float().clone()
    matched_logits = _latent_logits(
        backend, bridge, matched_states, latent_full,
        insertion_index=insertion_index, caption_tokens=caption_tokens,
    )
    caption_nll = token_cross_entropy(matched_logits, targets)
    teacher_kl = text_teacher_forward_kl(
        matched_logits,
        teacher_logits.to(device=device, dtype=matched_logits.dtype),
        torch.ones_like(targets, dtype=torch.bool),
        temperature=float(loss_config["temperature"]),
    )
    matched_logp = sequence_log_probability(matched_logits, targets)
    matched_surrogate = (
        float(loss_config["caption_nll"]) * caption_nll
        + float(loss_config["text_teacher_forward_kl"]) * teacher_kl
        - shuffled_weight * (margin_coefficient * matched_logp).mean()
    )
    matched_surrogate.backward()
    del matched_logits, matched_surrogate

    shuffled_logits = _latent_logits(
        backend, bridge, shuffled_states, latent_full,
        insertion_index=insertion_index, caption_tokens=caption_tokens,
    )
    shuffled_logp = sequence_log_probability(shuffled_logits, targets)
    shuffled_surrogate = shuffled_weight * (margin_coefficient * shuffled_logp).mean()
    shuffled_surrogate.backward()

    gradient_values = [
        parameter.grad.detach().float().norm()
        for parameter in bridge.parameters()
        if parameter.grad is not None
    ]
    if not gradient_values or not all(torch.isfinite(value) for value in gradient_values):
        raise RuntimeError("bridge gradients are missing or non-finite")
    gradient_norm = torch.linalg.vector_norm(torch.stack(gradient_values)).item()
    torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=1.0)
    optimizer.step()
    parameter_delta = (first_parameter.detach().float() - first_parameter_before).norm().item()
    if not math.isfinite(parameter_delta) or parameter_delta <= 0:
        raise RuntimeError("optimizer step did not update the bridge")
    exact_total = (
        float(loss_config["caption_nll"]) * caption_nll.detach()
        + float(loss_config["text_teacher_forward_kl"]) * teacher_kl.detach()
        + shuffled_weight * exact_margin_value
    )
    scalar_values = [caption_nll, teacher_kl, exact_margin_value, exact_total]
    if not all(torch.isfinite(value).all() for value in scalar_values) or not math.isfinite(gradient_norm):
        raise RuntimeError("loss or gradient diagnostics are non-finite")

    report = {
        "status": "pass",
        "scientific_training": False,
        "synthetic_sender_states": True,
        "receiver": config["experiment"]["receiver"],
        "adapter": None if adapter_path is None else str(adapter_path),
        "alignment": {
            "generated_tokens": int(aligned.token_ids.shape[1]),
            "hidden_width": int(aligned.states.shape[2]),
            "validated_conflict_pairs": len(parsed_conflicts),
            "semantics": aligned.semantics,
            "teacher_replay_mean_cosine": alignment_report.replay.mean_cosine,
            "teacher_replay_min_cosine": alignment_report.replay.min_cosine,
            "teacher_replay_relative_l2": alignment_report.replay.relative_l2,
            "processed_score_token_accuracy": alignment_report.processed_score_token_accuracy,
            "raw_head_token_accuracy_diagnostic": alignment_report.raw_head_token_accuracy,
        },
        "token_contract": {
            "text_prompt_tokens": int(text_prefix["input_ids"].shape[1]),
            "latent_prompt_tokens": int(latent_prefix["input_ids"].shape[1]),
            "caption_tokens": caption_tokens,
            "latent_slots": 3 * int(bridge_config["slots_per_channel"]),
            "receiver_embedding_mean_norm": receiver_embedding_norm,
        },
        "freeze": freeze.__dict__,
        "loss": {
            "caption_nll": float(caption_nll.detach().cpu()),
            "teacher_kl": float(teacher_kl.detach().cpu()),
            "matched_shuffled_margin": float(exact_margin_value.detach().cpu()),
            "total": float(exact_total.detach().cpu()),
            "bridge_gradient_norm": gradient_norm,
            "first_parameter_update_norm": parameter_delta,
        },
        "memory_bytes": {
            "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
        },
        "notes": [
            "The one-step bridge smoke uses synthetic sender states and is not a quality experiment.",
            "The contrastive term is differentiated by an exact-gradient two-pass recomputation to avoid retaining two receiver graphs.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
