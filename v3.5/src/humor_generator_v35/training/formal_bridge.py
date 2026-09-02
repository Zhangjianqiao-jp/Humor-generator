"""Formal, cluster-balanced bridge-only optimization over frozen Planner traces."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from ..data.traces import load_trace, plan_from_record, read_jsonl
from ..homer.prompts import image_part, text_part
from ..latent.bridges import LearnedLatentBridge, TypedLatentBridge
from ..latent.state_capture import AlignedMessageStates
from ..qwen_backend import QwenBackend
from .losses import sequence_log_probability, text_teacher_forward_kl, token_cross_entropy
from .memory_safe import (
    cache_text_teacher_logits,
    caption_only_logits,
    inject_latent_slots,
    inject_position_ids,
)


GENERATOR_INSTRUCTION = (
    "Generate one short, natural, image-specific humorous caption. Do not explain."
)


@dataclass(frozen=True)
class PreparedExample:
    row: dict[str, Any]
    trace_record: dict[str, Any]


def cluster_balanced_rows(rows: Iterable[dict[str, Any]], *, epoch: int, seed: int) -> list[dict[str, Any]]:
    """Select exactly one caption per image cluster and shuffle deterministically."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["cluster_id"]].append(row)
    selected = []
    for cluster in sorted(grouped):
        options = sorted(grouped[cluster], key=lambda item: item["row_id"])
        selected.append(options[epoch % len(options)])
    random.Random(seed + epoch).shuffle(selected)
    return selected


def shuffled_cluster_map(rows: list[dict[str, Any]], *, seed: int) -> dict[str, str]:
    clusters = sorted({row["cluster_id"] for row in rows})
    if len(clusters) < 2:
        raise ValueError("matched/shuffled training requires at least two image clusters")
    shuffled = clusters[:]
    rng = random.Random(seed)
    while True:
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(clusters, shuffled)):
            return dict(zip(clusters, shuffled))


def hard_negative_cluster_map(
    rows: list[dict[str, Any]],
    trace_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, float]]:
    """Choose description-nearest negatives with a different conflict signature."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["cluster_id"]].append(row)
    clusters = sorted(grouped)
    if len(clusters) < 2:
        raise ValueError("hard-negative training requires at least two clusters")
    descriptions = [
        " ".join(sorted({str(row["standard_description"]) for row in grouped[cluster]}))
        for cluster in clusters
    ]
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(descriptions)
    similarity = linear_kernel(matrix, matrix)
    source_sets = [
        {str(row["dataset"]) for row in grouped[cluster]}
        for cluster in clusters
    ]
    conflict_signatures = []
    for cluster in clusters:
        plan = plan_from_record(trace_index[cluster]["plan"])
        conflict_signatures.append(tuple(sorted(item.render().casefold() for item in plan.conflicts)))
    result: dict[str, str] = {}
    chosen_scores: list[float] = []
    same_source = 0
    for left, cluster in enumerate(clusters):
        candidates = [
            right for right in range(len(clusters))
            if right != left and conflict_signatures[right] != conflict_signatures[left]
        ]
        if not candidates:
            raise RuntimeError(f"no different-conflict hard negative for {cluster}")
        shared_source = [right for right in candidates if source_sets[left] & source_sets[right]]
        pool = shared_source or candidates
        right = max(pool, key=lambda value: (float(similarity[left, value]), clusters[value]))
        result[cluster] = clusters[right]
        chosen_scores.append(float(similarity[left, right]))
        same_source += int(bool(source_sets[left] & source_sets[right]))
    return result, {
        "mean_description_tfidf_similarity": sum(chosen_scores) / len(chosen_scores),
        "same_source_fraction": same_source / len(clusters),
        "pairs": float(len(clusters)),
    }


def prepare_example(row: dict[str, Any], trace_record: dict[str, Any], *, seed: int) -> PreparedExample:
    del seed  # Retained in the public call signature for deterministic callers.
    plan = plan_from_record(trace_record["plan"])
    if not plan.local_chains or not plan.global_chains:
        raise ValueError(f"trace has no association chain: {row['cluster_id']}")
    return PreparedExample(row, trace_record)


def _prompt_and_full(
    backend: QwenBackend,
    prompt_messages: list[dict[str, Any]],
    answer: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    prompt = backend.encode(prompt_messages, add_generation_prompt=True)
    full = backend.encode(
        [*prompt_messages, {"role": "assistant", "content": [text_part(answer)]}],
        add_generation_prompt=False,
    )
    prompt_length = prompt["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][:, :prompt_length], prompt["input_ids"]):
        raise RuntimeError("supervised sequence does not preserve the exact chat-template prefix")
    targets = full["input_ids"][:, prompt_length:].clone()
    if targets.shape[1] < 1:
        raise RuntimeError("caption has no supervised tokens")
    return prompt, full, targets


def latent_messages(image: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [image_part(image), text_part(GENERATOR_INSTRUCTION)]},
    ]


def full_plan_text_messages(image: str, semantics: dict[str, str]) -> list[dict[str, Any]]:
    required = set(TypedLatentBridge.channel_order)
    if set(semantics) != required or any(not semantics[name].strip() for name in required):
        raise ValueError("full-plan text requires non-empty conflict/local/global semantics")
    payload = "\n\n".join(
        f"<{name.upper()}>\n{semantics[name]}\n</{name.upper()}>"
        for name in TypedLatentBridge.channel_order
    )
    return [
        {"role": "user", "content": [
            image_part(image),
            text_part(f"{GENERATOR_INSTRUCTION}\n\nHumor plan:\n{payload}"),
        ]},
    ]


class FrozenReceiverBridgeTask:
    def __init__(
        self,
        backend: QwenBackend,
        bridge: nn.Module,
        *,
        root: Path,
        trace_index: dict[str, dict[str, Any]],
        loss_config: dict[str, float],
        max_caption_tokens: int,
    ) -> None:
        self.backend = backend
        self.bridge = bridge
        self.root = root
        self.trace_index = trace_index
        self.loss_config = loss_config
        self.max_caption_tokens = max_caption_tokens
        for parameter in backend.model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in backend.model.parameters()):
            raise RuntimeError("receiver policy is not frozen")

    def _states(self, cluster: str) -> dict[str, AlignedMessageStates]:
        record = self.trace_index[cluster]
        path = self.root / record["trace_path"]
        loaded = load_trace(path, expected_sha256=record["trace_sha256"])
        device = next(self.bridge.parameters()).device
        return {
            name: AlignedMessageStates(
                item.token_ids.to(device),
                item.states.to(device=device, dtype=next(self.bridge.parameters()).dtype),
                item.semantics,
            )
            for name, item in loaded.items()
        }

    def _slots(self, states: dict[str, AlignedMessageStates]) -> torch.Tensor:
        if isinstance(self.bridge, TypedLatentBridge):
            return self.bridge({name: item.states for name, item in states.items()})["all"]
        if isinstance(self.bridge, LearnedLatentBridge):
            joined = torch.cat(
                [states[name].states for name in TypedLatentBridge.channel_order], dim=1
            )
            return self.bridge(joined)
        raise TypeError(f"unsupported bridge: {type(self.bridge).__name__}")

    def _latent_logits(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        insertion_index: int,
        caption_tokens: int,
        states: dict[str, AlignedMessageStates],
    ) -> torch.Tensor:
        slots = self._slots(states).to(dtype=embeddings.dtype)
        combined, mask = inject_latent_slots(
            token_embeddings=embeddings,
            attention_mask=attention_mask,
            slots=slots,
            insertion_index=insertion_index,
        )
        combined_positions = inject_position_ids(
            position_ids, slot_count=slots.shape[1], insertion_index=insertion_index
        )
        return caption_only_logits(
            self.backend.model,
            inputs_embeds=combined,
            attention_mask=mask,
            position_ids=combined_positions,
            caption_tokens=caption_tokens,
        )

    def prepare(self, example: PreparedExample) -> tuple[Any, ...]:
        caption = example.row["caption"]
        matched_states = self._states(example.row["cluster_id"])
        text_prompt = full_plan_text_messages(
            example.row["image"],
            {name: matched_states[name].semantics for name in TypedLatentBridge.channel_order},
        )
        latent_prompt = latent_messages(example.row["image"])
        latent_prefix, latent_full, latent_targets = _prompt_and_full(
            self.backend, latent_prompt, caption
        )
        latent_embeddings, latent_positions = (
            self.backend.multimodal_embeddings_and_positions(latent_full)
        )
        if latent_targets.shape[1] > self.max_caption_tokens:
            raise RuntimeError(
                f"caption exceeds max_caption_tokens={self.max_caption_tokens}: {example.row['row_id']}"
            )
        teacher_logits = None
        if float(self.loss_config["text_teacher_forward_kl"]) > 0:
            _, text_full, text_targets = _prompt_and_full(self.backend, text_prompt, caption)
            if not torch.equal(text_targets, latent_targets):
                raise RuntimeError("text and latent conditions tokenize the caption differently")
            with torch.no_grad():
                teacher_embeddings, teacher_positions = (
                    self.backend.multimodal_embeddings_and_positions(text_full)
                )
                teacher_logits = cache_text_teacher_logits(
                    self.backend.model,
                    inputs_embeds=teacher_embeddings,
                    attention_mask=text_full["attention_mask"],
                    position_ids=teacher_positions,
                    caption_tokens=int(text_targets.shape[1]),
                )
        return (
            latent_embeddings,
            latent_full["attention_mask"],
            latent_positions,
            int(latent_prefix["input_ids"].shape[1]),
            latent_targets,
            teacher_logits,
            matched_states,
        )

    def backward_example(
        self,
        example: PreparedExample,
        shuffled_cluster: str,
        *,
        loss_scale: float = 1.0,
    ) -> dict[str, float]:
        if loss_scale <= 0:
            raise ValueError("loss_scale must be positive")
        embeddings, attention_mask, positions, insertion, targets, teacher_cpu, matched_states = (
            self.prepare(example)
        )
        shuffled_states = self._states(shuffled_cluster)
        caption_tokens = int(targets.shape[1])
        margin = float(self.loss_config["margin"])
        shuffled_weight = float(self.loss_config["matched_shuffled_margin"])

        # Exact two-pass first derivative: never retain matched and shuffled
        # 7B receiver graphs simultaneously.
        with torch.no_grad():
            matched0 = sequence_log_probability(
                self._latent_logits(
                    embeddings, attention_mask, positions,
                    insertion_index=insertion, caption_tokens=caption_tokens,
                    states=matched_states,
                ), targets,
            )
            shuffled0 = sequence_log_probability(
                self._latent_logits(
                    embeddings, attention_mask, positions,
                    insertion_index=insertion, caption_tokens=caption_tokens,
                    states=shuffled_states,
                ), targets,
            )
            coefficient = torch.sigmoid(-matched0 + shuffled0 + margin).detach()
            margin_value = F.softplus(-matched0 + shuffled0 + margin).mean()
            gap_value = (matched0 - shuffled0).mean()
            fraction_gap_positive = (matched0 > shuffled0).float().mean()
            fraction_gap_over_margin = ((matched0 - shuffled0) > margin).float().mean()

        matched_logits = self._latent_logits(
            embeddings, attention_mask, positions,
            insertion_index=insertion, caption_tokens=caption_tokens, states=matched_states
        )
        caption_nll = token_cross_entropy(matched_logits, targets)
        teacher_kl = matched_logits.new_zeros(())
        if teacher_cpu is not None:
            teacher_kl = text_teacher_forward_kl(
                matched_logits,
                teacher_cpu.to(device=matched_logits.device, dtype=matched_logits.dtype),
                torch.ones_like(targets, dtype=torch.bool),
                temperature=float(self.loss_config["temperature"]),
            )
        matched_logp = sequence_log_probability(matched_logits, targets)
        metric_caption_nll = caption_nll.detach()
        metric_teacher_kl = teacher_kl.detach()
        matched_loss = (
            float(self.loss_config["caption_nll"]) * caption_nll
            + float(self.loss_config["text_teacher_forward_kl"]) * teacher_kl
            - shuffled_weight * (coefficient * matched_logp).mean()
        )
        (loss_scale * matched_loss).backward()
        del caption_nll, teacher_kl, matched_logp, matched_logits, matched_loss

        shuffled_logits = self._latent_logits(
            embeddings, attention_mask, positions,
            insertion_index=insertion, caption_tokens=caption_tokens, states=shuffled_states
        )
        shuffled_logp = sequence_log_probability(shuffled_logits, targets)
        (loss_scale * shuffled_weight * (coefficient * shuffled_logp).mean()).backward()
        total = (
            float(self.loss_config["caption_nll"]) * metric_caption_nll
            + float(self.loss_config["text_teacher_forward_kl"]) * metric_teacher_kl
            + shuffled_weight * margin_value
        )
        return {
            "total": float(total.cpu()),
            "caption_nll": float(metric_caption_nll.cpu()),
            "teacher_kl": float(metric_teacher_kl.cpu()),
            "shuffled_margin": float(margin_value.cpu()),
            "matched_logp": float(matched0.mean().cpu()),
            "shuffled_logp": float(shuffled0.mean().cpu()),
            "matched_minus_shuffled_logp": float(gap_value.cpu()),
            "fraction_gap_gt_0": float(fraction_gap_positive.cpu()),
            "fraction_gap_gt_margin": float(fraction_gap_over_margin.cpu()),
        }

    @torch.no_grad()
    def evaluate_example(
        self,
        example: PreparedExample,
        shuffled_cluster: str,
    ) -> dict[str, float]:
        # Gradients are disabled, so the two logits can be evaluated and freed
        # sequentially without the surrogate used in training.
        embeddings, attention_mask, positions, insertion, targets, teacher_cpu, matched_states = (
            self.prepare(example)
        )
        caption_tokens = int(targets.shape[1])
        matched = self._latent_logits(
            embeddings, attention_mask, positions,
            insertion_index=insertion, caption_tokens=caption_tokens,
            states=matched_states,
        )
        nll = token_cross_entropy(matched, targets)
        kl = matched.new_zeros(())
        if teacher_cpu is not None:
            kl = text_teacher_forward_kl(
                matched,
                teacher_cpu.to(device=matched.device, dtype=matched.dtype),
                torch.ones_like(targets, dtype=torch.bool),
                temperature=float(self.loss_config["temperature"]),
            )
        matched_logp = sequence_log_probability(matched, targets)
        del matched
        shuffled = self._latent_logits(
            embeddings, attention_mask, positions,
            insertion_index=insertion, caption_tokens=caption_tokens,
            states=self._states(shuffled_cluster),
        )
        shuffled_logp = sequence_log_probability(shuffled, targets)
        margin_loss = F.softplus(
            -matched_logp + shuffled_logp + float(self.loss_config["margin"])
        ).mean()
        gap = (matched_logp - shuffled_logp).mean()
        total = (
            float(self.loss_config["caption_nll"]) * nll
            + float(self.loss_config["text_teacher_forward_kl"]) * kl
            + float(self.loss_config["matched_shuffled_margin"]) * margin_loss
        )
        return {
            "total": float(total.cpu()),
            "caption_nll": float(nll.cpu()),
            "teacher_kl": float(kl.cpu()),
            "shuffled_margin": float(margin_loss.cpu()),
            "matched_logp": float(matched_logp.mean().cpu()),
            "shuffled_logp": float(shuffled_logp.mean().cpu()),
            "matched_minus_shuffled_logp": float(gap.cpu()),
            "fraction_gap_gt_0": float((matched_logp > shuffled_logp).float().mean().cpu()),
            "fraction_gap_gt_margin": float(
                ((matched_logp - shuffled_logp) > float(self.loss_config["margin"]))
                .float().mean().cpu()
            ),
        }


def mean_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot average empty metrics")
    return {key: sum(item[key] for item in values) / len(values) for key in values[0]}


def load_trace_index(path: Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl(path)
    result = {item["cluster_id"]: item for item in records}
    if len(result) != len(records):
        raise RuntimeError("duplicate cluster in trace index")
    return result
