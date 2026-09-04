"""Two-stage semantic reconstruction and caption training for latent enrichment."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from ..data.traces import load_trace
from ..homer.prompts import image_part, text_part
from ..latent.bridges import TypedLatentBridge
from ..latent.cross_attention import ReceiverDrivenCrossAttentionBridge
from ..latent.state_capture import AlignedMessageStates
from ..qwen_backend import QwenBackend, find_last_decoder_layer
from .formal_bridge import (
    GENERATOR_INSTRUCTION,
    PreparedExample,
    _prompt_and_full,
    full_plan_text_messages,
    mean_metrics,
)
from .losses import sequence_log_probability, text_teacher_forward_kl, token_cross_entropy
from .memory_safe import cache_text_teacher_logits, caption_only_logits


SEMANTIC_RECOVERY_INSTRUCTION = (
    "Recover the exact structured humor plan carried by the external memory. "
    "Preserve every fact, conflict and association; output only the structured plan."
)


def zero_prefix_caption_messages(image: str) -> list[dict[str, Any]]:
    """Normal SFT task prompt with no latent pseudo-token or assistant-side prefix."""
    return [{"role": "user", "content": [image_part(image), text_part(GENERATOR_INSTRUCTION)]}]


def semantic_recovery_messages(
    image: str,
    channel: str | None = None,
    *,
    include_image: bool = True,
) -> list[dict[str, Any]]:
    """Build the semantic receiver prompt.

    A3 keeps the image for historical reproduction.  A4 sets
    ``include_image=False`` so a channel-isolated recovery example cannot
    solve the target from an image/prompt shortcut instead of latent memory.
    """
    instruction = SEMANTIC_RECOVERY_INSTRUCTION
    if channel is not None:
        if channel not in TypedLatentBridge.channel_order:
            raise ValueError(f"unknown semantic channel: {channel}")
        instruction = (
            f"Recover only the exact {channel} field carried by the external memory. "
            "Preserve every word and output only that field, without a heading or explanation."
        )
    content: list[dict[str, Any]] = []
    if include_image:
        content.append(image_part(image))
    content.append(text_part(instruction))
    return [{"role": "user", "content": content}]


def contextual_teacher_messages(channel: str, semantics: str) -> list[dict[str, Any]]:
    """Receiver-native text condition used to cache a contextual teacher."""
    if channel not in TypedLatentBridge.channel_order:
        raise ValueError(f"unknown semantic channel: {channel}")
    return [{
        "role": "user",
        "content": [text_part(
            f"Humor Planner {channel} field:\n<{channel.upper()}>\n"
            f"{semantics}\n</{channel.upper()}>"
        )],
    }]


def semantic_teacher_messages(channel: str, semantics: str) -> list[dict[str, Any]]:
    """Build a receiver-native teacher condition for semantic recovery.

    The teacher exposes exactly one typed field in ordinary receiver text and
    is used only to cache the frozen receiver's target-span hidden state.  The
    latent student sees the same recovery instruction but not this text, so
    the alignment target is a receiver-native semantic representation rather
    than a random projection of the sender trace.
    """
    if channel not in TypedLatentBridge.channel_order:
        raise ValueError(f"unknown semantic channel: {channel}")
    return [{
        "role": "user",
        "content": [text_part(
            "Use the following Humor Planner field to answer exactly. "
            "Return only the field, without explanation.\n"
            f"<{channel.upper()}>\n{semantics}\n</{channel.upper()}>"
        )],
    }]


def exact_typed_semantics(states: dict[str, AlignedMessageStates]) -> str:
    if set(states) != set(TypedLatentBridge.channel_order):
        raise ValueError("semantic target requires conflict/local/global")
    return "\n\n".join(
        f"<{name.upper()}>\n{states[name].semantics}\n</{name.upper()}>"
        for name in TypedLatentBridge.channel_order
    )


class ReceiverCrossAttentionTask:
    """OOM-safe bridge task; sender and receiver remain frozen.

    ``semantic_reconstruction`` is representation-first training.  The exact
    Planner text must be recoverable from full latent memory.  ``caption`` is
    downstream training with no input-prefix latent tokens.
    """

    def __init__(self, backend: QwenBackend, bridge: ReceiverDrivenCrossAttentionBridge, *,
                 root: Path, trace_index: dict[str, dict[str, Any]],
                 loss_config: dict[str, float], max_target_tokens: int,
                 stage: str, semantic_prompt_include_image: bool = True) -> None:
        if stage not in {"semantic_reconstruction", "caption"}:
            raise ValueError("stage must be semantic_reconstruction or caption")
        self.backend = backend
        self.bridge = bridge
        self.root = root
        self.trace_index = trace_index
        self.loss_config = loss_config
        self.max_target_tokens = max_target_tokens
        self.stage = stage
        self.semantic_prompt_include_image = bool(semantic_prompt_include_image)
        self.channel_visibility = str(loss_config.get("channel_visibility", "all"))
        if self.channel_visibility not in {"all", "target_only"}:
            raise ValueError("channel_visibility must be all or target_only")
        self.counterfactual_reconstruction_weight = float(
            loss_config.get("counterfactual_reconstruction", 0.0)
        )
        if self.counterfactual_reconstruction_weight < 0:
            raise ValueError("counterfactual_reconstruction must be non-negative")
        self.semantic_objective = str(
            loss_config.get("semantic_objective", "joint_reconstruction_v2")
        )
        self.alignment_teacher = str(
            loss_config.get("alignment_teacher", "legacy_embedding_projection")
        )
        if self.alignment_teacher not in {
            "legacy_embedding_projection", "receiver_contextual_final_hidden"
        }:
            raise ValueError("unsupported alignment_teacher")
        if self.semantic_objective not in {
            "joint_reconstruction_v2", "channel_balanced_v3", "channel_isolated_v4",
            "channel_isolated_v5",
        }:
            raise ValueError("unsupported semantic_objective")
        self._contextual_teacher_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.semantic_target_alignment_weight = float(
            loss_config.get("semantic_target_alignment", 0.0)
        )
        if self.semantic_target_alignment_weight < 0:
            raise ValueError("semantic_target_alignment must be non-negative")
        self._semantic_target_teacher_cache: dict[str, dict[str, torch.Tensor]] = {}
        for parameter in backend.model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in backend.model.parameters()):
            raise RuntimeError("receiver policy is not frozen")

    def _states(self, cluster: str) -> dict[str, AlignedMessageStates]:
        record = self.trace_index[cluster]
        loaded = load_trace(
            self.root / record["trace_path"], expected_sha256=record["trace_sha256"]
        )
        parameter = next(self.bridge.parameters())
        return {
            name: AlignedMessageStates(
                item.token_ids.to(parameter.device),
                item.states.to(device=parameter.device, dtype=parameter.dtype),
                item.semantics,
            ) for name, item in loaded.items()
        }

    @staticmethod
    def _tensor_states(states: dict[str, AlignedMessageStates]) -> dict[str, torch.Tensor]:
        return {name: states[name].states for name in TypedLatentBridge.channel_order}

    @torch.no_grad()
    def _contextual_teacher(self, channel: str, semantics: str) -> torch.Tensor:
        """Mean contextual state from the frozen receiver's final decoder layer."""
        encoded = self.backend.encode(
            contextual_teacher_messages(channel, semantics), add_generation_prompt=False
        )
        embeddings, positions = self.backend.multimodal_embeddings_and_positions(encoded)
        captured: list[torch.Tensor] = []

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output if torch.is_tensor(output) else output[0]
            captured.append(hidden.detach())

        handle = find_last_decoder_layer(self.backend.model).register_forward_hook(hook)
        try:
            self.backend.model(
                inputs_embeds=embeddings,
                attention_mask=encoded["attention_mask"],
                position_ids=positions,
                labels=None,
                use_cache=False,
                logits_to_keep=1,
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one contextual teacher hook call, got {len(captured)}")
        mask = encoded["attention_mask"].bool().unsqueeze(-1)
        pooled = (captured[0].float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        return pooled.to(torch.float16).cpu()

    def cache_contextual_teachers(self, clusters: list[str] | set[str]) -> None:
        """Cache teachers once; no frozen-receiver graph is retained."""
        if self.alignment_teacher != "receiver_contextual_final_hidden":
            return
        for cluster in sorted(set(clusters)):
            if cluster in self._contextual_teacher_cache:
                continue
            states = self._states(cluster)
            self._contextual_teacher_cache[cluster] = {
                name: self._contextual_teacher(name, states[name].semantics)
                for name in TypedLatentBridge.channel_order
            }

    @torch.no_grad()
    def _semantic_target_teacher(self, channel: str, semantics: str) -> torch.Tensor:
        """Pool the frozen receiver's native hidden state over target tokens.

        This is the semantic-recovery counterpart of BLIP-2's generative
        alignment stage: the receiver remains frozen, while a bridge is
        trained to produce a state the receiver can interpret.  Pooling keeps
        the cache small (one D-dimensional vector per channel) and avoids
        retaining vocabulary-sized teacher logits.
        """
        messages = semantic_teacher_messages(channel, semantics)
        _, full, targets = _prompt_and_full(self.backend, messages, semantics)
        embeddings, positions = self.backend.multimodal_embeddings_and_positions(full)
        captured: list[torch.Tensor] = []

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output if torch.is_tensor(output) else output[0]
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                raise RuntimeError("receiver teacher did not emit [B,T,D] hidden states")
            captured.append(hidden.detach())

        handle = find_last_decoder_layer(self.backend.model).register_forward_hook(hook)
        try:
            self.backend.model(
                inputs_embeds=embeddings,
                attention_mask=full["attention_mask"],
                position_ids=positions,
                labels=None,
                use_cache=False,
                logits_to_keep=1,
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one semantic teacher hook call, got {len(captured)}")
        # The causal logit for each target token is emitted at the preceding
        # position, matching caption_only_logits' indexing contract.
        target_hidden = captured[0][:, -targets.shape[1] - 1:-1, :]
        if target_hidden.shape[1] != targets.shape[1]:
            raise RuntimeError("semantic teacher target-span alignment is off by one")
        return target_hidden.float().mean(dim=1).to(torch.float16).cpu()

    def cache_semantic_target_teachers(self, clusters: list[str] | set[str]) -> None:
        """Cache one receiver-native target-span vector per cluster/channel."""
        if self.semantic_target_alignment_weight <= 0:
            return
        for cluster in sorted(set(clusters)):
            if cluster in self._semantic_target_teacher_cache:
                continue
            states = self._states(cluster)
            self._semantic_target_teacher_cache[cluster] = {
                name: self._semantic_target_teacher(name, states[name].semantics)
                for name in TypedLatentBridge.channel_order
            }

    def semantic_target_teacher(self, cluster: str, channel: str) -> torch.Tensor:
        if self.semantic_target_alignment_weight <= 0:
            raise RuntimeError("semantic target alignment is disabled")
        if cluster not in self._semantic_target_teacher_cache:
            self.cache_semantic_target_teachers([cluster])
        return self._semantic_target_teacher_cache[cluster][channel]

    def semantic_alignment_pairs(
        self, example: PreparedExample,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Represent one matched Planner trace in sender and receiver spaces.

        This deliberately keeps the small alignment graph outside the frozen
        receiver forward, so gradient-accumulation windows can form a genuine
        multi-example InfoNCE batch without retaining VLM activations.
        """
        states = self._states(example.row["cluster_id"])
        parameter = next(self.bridge.parameters())
        cluster = example.row["cluster_id"]
        if self.alignment_teacher == "receiver_contextual_final_hidden":
            if cluster not in self._contextual_teacher_cache:
                self.cache_contextual_teachers([cluster])
            receiver_contexts = {
                name: value.to(device=parameter.device, dtype=parameter.dtype)
                for name, value in self._contextual_teacher_cache[cluster].items()
            }
        else:
            embedding = self.backend.model.get_input_embeddings()
            receiver_contexts = {
                name: embedding(states[name].token_ids.to(parameter.device)).to(parameter.dtype)
                for name in TypedLatentBridge.channel_order
            }
        return self.bridge.alignment_representations_by_channel(
            self._tensor_states(states), receiver_contexts
        )

    def semantic_alignment_pair(
        self, example: PreparedExample,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility view for old smokes; Phase A3 uses channel-wise pairs."""
        pairs = self.semantic_alignment_pairs(example)
        return (
            torch.cat([pairs[name][0] for name in TypedLatentBridge.channel_order], dim=-1),
            torch.cat([pairs[name][1] for name in TypedLatentBridge.channel_order], dim=-1),
        )

    def prepare(self, example: PreparedExample) -> tuple[Any, ...]:
        states = self._states(example.row["cluster_id"])
        semantics = {
            name: states[name].semantics for name in TypedLatentBridge.channel_order
        }
        if self.stage == "caption":
            messages = zero_prefix_caption_messages(example.row["image"])
            target = example.row["caption"]
        else:
            messages = semantic_recovery_messages(example.row["image"])
            target = exact_typed_semantics(states)
        _, full, targets = _prompt_and_full(self.backend, messages, target)
        if targets.shape[1] > self.max_target_tokens:
            raise RuntimeError(
                f"target exceeds max_target_tokens={self.max_target_tokens}: "
                f"{example.row['row_id']} ({targets.shape[1]})"
            )
        embeddings, positions = self.backend.multimodal_embeddings_and_positions(full)
        teacher_logits = None
        if self.stage == "caption" and float(self.loss_config.get("text_teacher_forward_kl", 0)) > 0:
            _, teacher_full, teacher_targets = _prompt_and_full(
                self.backend, full_plan_text_messages(example.row["image"], semantics), target
            )
            if not torch.equal(teacher_targets, targets):
                raise RuntimeError("text and latent conditions tokenize target differently")
            with torch.no_grad():
                teacher_embeddings, teacher_positions = self.backend.multimodal_embeddings_and_positions(
                    teacher_full
                )
                teacher_logits = cache_text_teacher_logits(
                    self.backend.model,
                    inputs_embeds=teacher_embeddings,
                    attention_mask=teacher_full["attention_mask"],
                    position_ids=teacher_positions,
                    caption_tokens=int(targets.shape[1]),
                )
        return embeddings, full["attention_mask"], positions, targets, teacher_logits, states

    def prepare_semantic_channel(self, example: PreparedExample, channel: str) -> tuple[Any, ...]:
        """Prepare one length-normalized semantic channel for Phase A3/A4."""
        if self.stage != "semantic_reconstruction":
            raise RuntimeError("channel reconstruction is only defined for semantic Phase A")
        states = self._states(example.row["cluster_id"])
        target = states[channel].semantics
        messages = semantic_recovery_messages(
            example.row["image"], channel,
            include_image=self.semantic_prompt_include_image,
        )
        _, full, targets = _prompt_and_full(self.backend, messages, target)
        if targets.shape[1] > self.max_target_tokens:
            raise RuntimeError(
                f"{channel} target exceeds max_target_tokens={self.max_target_tokens}: "
                f"{example.row['row_id']} ({targets.shape[1]})"
            )
        embeddings, positions = self.backend.multimodal_embeddings_and_positions(full)
        return embeddings, full["attention_mask"], positions, targets, None, states

    def prepare_semantic_target(
        self, example: PreparedExample, channel: str, target: str,
    ) -> tuple[Any, ...]:
        """Prepare a semantic target under exactly the same receiver prompt.

        A4 uses this for donor-side counterfactual reconstruction: after
        replacing only channel ``c``, the receiver must decode the donor's
        semantics, rather than merely lowering the original target's score.
        """
        if self.stage != "semantic_reconstruction":
            raise RuntimeError("semantic target preparation is only defined for semantic Phase A")
        states = self._states(example.row["cluster_id"])
        messages = semantic_recovery_messages(
            example.row["image"], channel,
            include_image=self.semantic_prompt_include_image,
        )
        _, full, targets = _prompt_and_full(self.backend, messages, target)
        if targets.shape[1] > self.max_target_tokens:
            raise RuntimeError(
                f"{channel} target exceeds max_target_tokens={self.max_target_tokens}: "
                f"{example.row['row_id']}"
            )
        embeddings, positions = self.backend.multimodal_embeddings_and_positions(full)
        return embeddings, full["attention_mask"], positions, targets, None, states

    def _logits_and_hidden(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor,
        positions: torch.Tensor, targets: torch.Tensor,
        states: dict[str, AlignedMessageStates],
        *, active_channels: Sequence[str] | None = None,
        capture_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # No latent pseudo-token is inserted.  This is the stable out-of-band
        # "zero-prefix" receiver interface requested for the new pipeline.
        captured: list[torch.Tensor] = []
        # Register the bridge hooks before the diagnostic hook.  When a
        # selected bridge layer is the final decoder block, PyTorch runs
        # hooks in registration order; capturing first would observe the
        # pre-injection hidden state and make the target-alignment loss
        # silently train against the wrong representation.
        with self.bridge.inject(
            self.backend.model,
            self._tensor_states(states),
            active_channels=active_channels,
        ):
            handle = None
            if capture_hidden:
                def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                    hidden = output if torch.is_tensor(output) else output[0]
                    if not torch.is_tensor(hidden) or hidden.ndim != 3:
                        raise RuntimeError("receiver did not emit [B,T,D] hidden states")
                    captured.append(hidden)
                handle = find_last_decoder_layer(self.backend.model).register_forward_hook(hook)
            try:
                logits = caption_only_logits(
                    self.backend.model,
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    position_ids=positions,
                    caption_tokens=int(targets.shape[1]),
                )
            finally:
                if handle is not None:
                    handle.remove()
        if not capture_hidden:
            return logits, None
        if len(captured) != 1:
            raise RuntimeError(f"expected one receiver hidden hook call, got {len(captured)}")
        hidden = captured[0][:, -targets.shape[1] - 1:-1, :]
        if hidden.shape[1] != targets.shape[1]:
            raise RuntimeError("receiver target-span hidden alignment is off by one")
        return logits, hidden

    def _logits(self, embeddings: torch.Tensor, attention_mask: torch.Tensor,
                positions: torch.Tensor, targets: torch.Tensor,
                states: dict[str, AlignedMessageStates],
                *, active_channels: Sequence[str] | None = None) -> torch.Tensor:
        return self._logits_and_hidden(
            embeddings, attention_mask, positions, targets, states,
            active_channels=active_channels,
        )[0]

    def _semantic_target_alignment_loss(
        self, hidden: torch.Tensor, *, cluster: str, channel: str,
    ) -> torch.Tensor:
        """Align a student's target-span state to a frozen native-text state."""
        teacher = self.semantic_target_teacher(cluster, channel).to(
            device=hidden.device, dtype=hidden.dtype
        )
        student = hidden.float().mean(dim=1)
        # Scale-free cosine distillation is stable across the three channels
        # and does not force the bridge to reproduce arbitrary hidden-state
        # norms.  A separate NLL and counterfactual term still controls task
        # behavior, so this auxiliary loss cannot win by making outputs random.
        return (1.0 - F.cosine_similarity(student, teacher.float(), dim=-1)).mean()

    def _forward_metrics(self, prepared: tuple[Any, ...], shuffled_cluster: str,
                         *, backward: bool, loss_scale: float = 1.0) -> dict[str, float]:
        embeddings, mask, positions, targets, teacher_cpu, matched_states = prepared
        shuffled_states = self._states(shuffled_cluster)
        margin = float(self.loss_config["margin"])
        shuffled_weight = float(self.loss_config["matched_shuffled_margin"])
        with torch.no_grad():
            matched0 = sequence_log_probability(
                self._logits(embeddings, mask, positions, targets, matched_states), targets
            )
            shuffled0 = sequence_log_probability(
                self._logits(embeddings, mask, positions, targets, shuffled_states), targets
            )
            coefficient = torch.sigmoid(-matched0 + shuffled0 + margin).detach()
            margin_loss = F.softplus(-matched0 + shuffled0 + margin).mean()

        matched = self._logits(embeddings, mask, positions, targets, matched_states)
        nll = token_cross_entropy(matched, targets)
        kl = matched.new_zeros(())
        if teacher_cpu is not None:
            kl = text_teacher_forward_kl(
                matched, teacher_cpu.to(device=matched.device, dtype=matched.dtype),
                torch.ones_like(targets, dtype=torch.bool),
                temperature=float(self.loss_config["temperature"]),
            )
        matched_logp = sequence_log_probability(matched, targets)
        nll_value, kl_value = nll.detach(), kl.detach()
        if backward:
            matched_part = (
                float(self.loss_config["caption_nll"]) * nll
                + float(self.loss_config.get("text_teacher_forward_kl", 0)) * kl
                - shuffled_weight * (coefficient * matched_logp).mean()
            )
            (loss_scale * matched_part).backward()
        del matched, nll, kl, matched_logp

        if backward:
            shuffled = self._logits(embeddings, mask, positions, targets, shuffled_states)
            shuffled_logp = sequence_log_probability(shuffled, targets)
            (loss_scale * shuffled_weight * (coefficient * shuffled_logp).mean()).backward()
            del shuffled, shuffled_logp

        gap = matched0 - shuffled0
        total = (
            float(self.loss_config["caption_nll"]) * nll_value
            + float(self.loss_config.get("text_teacher_forward_kl", 0)) * kl_value
            + shuffled_weight * margin_loss
        )
        diagnostics = self.bridge.last_diagnostics
        return {
            "total": float(total.cpu()),
            "caption_nll": float(nll_value.cpu()),
            "teacher_kl": float(kl_value.cpu()),
            "shuffled_margin": float(margin_loss.cpu()),
            "matched_logp": float(matched0.mean().cpu()),
            "shuffled_logp": float(shuffled0.mean().cpu()),
            "matched_minus_shuffled_logp": float(gap.mean().cpu()),
            "fraction_gap_gt_0": float((gap > 0).float().mean().cpu()),
            "fraction_gap_gt_margin": float((gap > margin).float().mean().cpu()),
            "mean_gate": float(sum(item.gate for item in diagnostics) / max(1, len(diagnostics))),
            "mean_attention_entropy": float(
                sum(item.attention_entropy for item in diagnostics) / max(1, len(diagnostics))
            ),
            "mean_relative_update_norm": float(
                sum(item.relative_update_norm for item in diagnostics) / max(1, len(diagnostics))
            ),
            **{
                f"mean_channel_weight_{name}": float(
                    sum(item.channel_weights[index] for item in diagnostics)
                    / max(1, len(diagnostics))
                )
                for index, name in enumerate(TypedLatentBridge.channel_order)
            },
        }

    def backward_example(
        self,
        example: PreparedExample,
        shuffled_cluster: str | None = None,
        *,
        shuffled_clusters: Mapping[str, str] | None = None,
        loss_scale: float = 1.0,
    ) -> dict[str, float]:
        if self.stage == "semantic_reconstruction" and self.semantic_objective in {
            "channel_balanced_v3", "channel_isolated_v4", "channel_isolated_v5"
        }:
            if shuffled_clusters is None and shuffled_cluster is None:
                raise ValueError("semantic backward requires a channel donor mapping")
            results = []
            for channel in TypedLatentBridge.channel_order:
                prepared = self.prepare_semantic_channel(example, channel)
                donor = (
                    shuffled_clusters[channel]
                    if shuffled_clusters is not None
                    else shuffled_cluster
                )
                results.append(self._forward_channel_metrics(
                    example, prepared, donor, channel=channel, backward=True,
                    loss_scale=loss_scale / len(TypedLatentBridge.channel_order),
                ))
            summary = mean_metrics(results)
            for channel, result in zip(TypedLatentBridge.channel_order, results):
                for key, value in result.items():
                    summary[f"{key}_{channel}"] = value
            return summary
        return self._forward_metrics(
            self.prepare(example), shuffled_cluster, backward=True, loss_scale=loss_scale
        )

    @torch.no_grad()
    def evaluate_example(
        self,
        example: PreparedExample,
        shuffled_cluster: str | None = None,
        *,
        shuffled_clusters: Mapping[str, str] | None = None,
    ) -> dict[str, float]:
        if self.stage == "semantic_reconstruction" and self.semantic_objective in {
            "channel_balanced_v3", "channel_isolated_v4", "channel_isolated_v5"
        }:
            if shuffled_clusters is None and shuffled_cluster is None:
                raise ValueError("semantic evaluation requires a channel donor mapping")
            results = [
                self._forward_channel_metrics(
                    example,
                    self.prepare_semantic_channel(example, channel),
                    (
                        shuffled_clusters[channel]
                        if shuffled_clusters is not None
                        else shuffled_cluster
                    ),
                    channel=channel, backward=False,
                )
                for channel in TypedLatentBridge.channel_order
            ]
            summary = mean_metrics(results)
            for channel, result in zip(TypedLatentBridge.channel_order, results):
                for key, value in result.items():
                    summary[f"{key}_{channel}"] = value
            return summary
        return self._forward_metrics(self.prepare(example), shuffled_cluster, backward=False)

    def _forward_channel_metrics(
        self, example: PreparedExample, prepared: tuple[Any, ...], shuffled_cluster: str, *, channel: str,
        backward: bool, loss_scale: float = 1.0,
    ) -> dict[str, float]:
        """Evaluate one channel's matched/counterfactual communication.

        In the historical A3 protocol all three channels are visible.  A4
        sets ``channel_visibility=target_only`` and therefore makes the
        replacement identifiable: only the selected channel can explain the
        semantic target.  The optional donor reconstruction term additionally
        requires the swapped channel to decode the donor semantics.
        """
        embeddings, mask, positions, targets, _teacher_cpu, matched_states = prepared
        donor_states = self._states(shuffled_cluster)
        counterfactual_states = dict(matched_states)
        counterfactual_states[channel] = donor_states[channel]
        active_channels: Sequence[str] | None = None
        if self.channel_visibility == "target_only":
            active_channels = (channel,)
        margin = float(self.loss_config["margin"])
        counterfactual_weight = float(self.loss_config["matched_shuffled_margin"])
        with torch.no_grad():
            matched0 = sequence_log_probability(
                self._logits(
                    embeddings, mask, positions, targets, matched_states,
                    active_channels=active_channels,
                ), targets
            )
            counterfactual0 = sequence_log_probability(
                self._logits(
                    embeddings, mask, positions, targets, counterfactual_states,
                    active_channels=active_channels,
                ), targets
            )
            coefficient = torch.sigmoid(-matched0 + counterfactual0 + margin).detach()
            margin_loss = F.softplus(-matched0 + counterfactual0 + margin).mean()

        matched, matched_hidden = self._logits_and_hidden(
            embeddings, mask, positions, targets, matched_states,
            active_channels=active_channels,
            capture_hidden=self.semantic_target_alignment_weight > 0,
        )
        reconstruction_nll = token_cross_entropy(matched, targets)
        matched_logp = sequence_log_probability(matched, targets)
        nll_value = reconstruction_nll.detach()
        semantic_alignment = matched.new_zeros(())
        if self.semantic_target_alignment_weight > 0:
            if matched_hidden is None:
                raise RuntimeError("semantic target alignment requested without hidden states")
            semantic_alignment = self._semantic_target_alignment_loss(
                matched_hidden, cluster=example.row["cluster_id"], channel=channel
            )
        if backward:
            matched_part = (
                float(self.loss_config["caption_nll"]) * reconstruction_nll
                - counterfactual_weight * (coefficient * matched_logp).mean()
                + self.semantic_target_alignment_weight * semantic_alignment
            )
            (loss_scale * matched_part).backward()
        semantic_alignment_value = semantic_alignment.detach()
        del matched, matched_hidden, reconstruction_nll, matched_logp, semantic_alignment

        if backward:
            counterfactual = self._logits(
                embeddings, mask, positions, targets, counterfactual_states,
                active_channels=active_channels,
            )
            counterfactual_logp = sequence_log_probability(counterfactual, targets)
            (
                loss_scale * counterfactual_weight
                * (coefficient * counterfactual_logp).mean()
            ).backward()
            del counterfactual, counterfactual_logp

        # A4's donor reconstruction closes the loophole where the bridge only
        # learns to make the original target unlikely after a swap.  With the
        # same typed prompt and isolated channel, the swapped memory must also
        # decode the donor semantics.  Weight zero exactly preserves A3.
        donor_reconstruction_value = matched0.new_zeros(())
        if self.counterfactual_reconstruction_weight > 0:
            donor_prepared = self.prepare_semantic_target(
                # The prompt/image is the target example; only the selected
                # channel memory is replaced by the donor below.
                example,
                channel,
                donor_states[channel].semantics,
            )
            donor_embeddings, donor_mask, donor_positions, donor_targets, _, _ = donor_prepared
            donor_logits = self._logits(
                donor_embeddings,
                donor_mask,
                donor_positions,
                donor_targets,
                counterfactual_states,
                active_channels=active_channels,
            )
            donor_nll = token_cross_entropy(donor_logits, donor_targets)
            donor_reconstruction_value = donor_nll.detach()
            if backward:
                (
                    loss_scale * self.counterfactual_reconstruction_weight * donor_nll
                ).backward()
            del donor_logits, donor_nll

        gap = matched0 - counterfactual0
        total = (
            float(self.loss_config["caption_nll"]) * nll_value
            + counterfactual_weight * margin_loss
            + self.counterfactual_reconstruction_weight * donor_reconstruction_value
            + self.semantic_target_alignment_weight * semantic_alignment_value
        )
        diagnostics = self.bridge.last_diagnostics
        return {
            "total": float(total.cpu()),
            "caption_nll": float(nll_value.cpu()),
            "teacher_kl": 0.0,
            "shuffled_margin": float(margin_loss.cpu()),
            "counterfactual_reconstruction_nll": float(donor_reconstruction_value.cpu()),
            "semantic_target_alignment": float(semantic_alignment_value.cpu()),
            "matched_logp": float(matched0.mean().cpu()),
            "shuffled_logp": float(counterfactual0.mean().cpu()),
            "matched_minus_shuffled_logp": float(gap.mean().cpu()),
            "fraction_gap_gt_0": float((gap > 0).float().mean().cpu()),
            "fraction_gap_gt_margin": float((gap > margin).float().mean().cpu()),
            "mean_gate": float(sum(item.gate for item in diagnostics) / max(1, len(diagnostics))),
            "mean_attention_entropy": float(
                sum(item.attention_entropy for item in diagnostics) / max(1, len(diagnostics))
            ),
            "mean_relative_update_norm": float(
                sum(item.relative_update_norm for item in diagnostics) / max(1, len(diagnostics))
            ),
            **{
                f"mean_channel_weight_{name}": float(
                    sum(item.channel_weights[index] for item in diagnostics)
                    / max(1, len(diagnostics))
                )
                for index, name in enumerate(TypedLatentBridge.channel_order)
            },
        }
