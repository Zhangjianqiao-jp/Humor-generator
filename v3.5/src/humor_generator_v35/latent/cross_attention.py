"""Receiver-driven, full-state latent enrichment for frozen Qwen receivers.

Unlike the legacy input-prefix bridges, this module never pretends that a
sender final-layer state is a receiver token embedding.  The receiver queries
the complete typed sender memory from selected decoder layers.  Only this
module is trainable; temporary hooks are removed after every forward call.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch
from torch import nn

from ..qwen_backend import find_decoder_layers


CHANNELS = ("conflict", "local", "global")


@dataclass(frozen=True)
class CrossAttentionDiagnostics:
    layer: int
    gate: float
    attention_entropy: float
    relative_update_norm: float
    channel_weights: tuple[float, float, float]


class _GatedLatentEnrichment(nn.Module):
    def __init__(self, receiver_dim: int, sender_dim: int, bottleneck_dim: int, heads: int,
                 gate_init: float) -> None:
        super().__init__()
        if bottleneck_dim % heads:
            raise ValueError("bottleneck_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = bottleneck_dim // heads
        self.receiver_norm = nn.LayerNorm(receiver_dim)
        self.sender_norm = nn.LayerNorm(sender_dim)
        self.query = nn.Linear(receiver_dim, bottleneck_dim, bias=False)
        # A fixed copy gives InfoNCE a stationary receiver-native target. Using
        # the live query weights under detach() would still let the target drift
        # between optimizer steps as caption/reconstruction gradients update Q.
        self.register_buffer(
            "alignment_teacher_projection",
            self.query.weight.detach().clone(),
            persistent=True,
        )
        self.key = nn.Linear(sender_dim, bottleneck_dim, bias=False)
        self.value = nn.Linear(sender_dim, bottleneck_dim, bias=False)
        self.output = nn.Linear(bottleneck_dim, receiver_dim, bias=False)
        # Hierarchical multi-source attention: normalize positions inside each
        # semantic source first, then let the receiver choose among sources.
        # This prevents a long association channel from receiving more mass
        # merely because it contains more tokens.
        self.channel_score = nn.Linear(bottleneck_dim, 1, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def _query(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, target_len, _ = hidden.shape
        return self.query(self.receiver_norm(hidden.float())).view(
            batch, target_len, self.heads, self.head_dim
        ).transpose(1, 2)

    def _attend_channel(
        self, q: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, source_len, _ = memory.shape
        normalized_memory = self.sender_norm(memory.float())
        k = self.key(normalized_memory).view(
            batch, source_len, self.heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(normalized_memory).view(
            batch, source_len, self.heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / self.head_dim**0.5
        scores = scores.masked_fill(~memory_mask[:, None, None, :].bool(), -torch.inf)
        probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
        attended = torch.matmul(probabilities, v).transpose(1, 2)
        return attended, probabilities

    def forward(
        self,
        hidden: torch.Tensor,
        memories: Mapping[str, torch.Tensor],
        memory_masks: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, float, float, tuple[float, float, float]]:
        batch, target_len, _ = hidden.shape
        # Keep the small trainable bridge in fp32 even when the frozen Qwen
        # receiver runs in bf16. Returning fp32 hidden states would leak the
        # dtype change into the next frozen decoder block, so only the residual
        # is converted back at the interface boundary.
        q = self._query(hidden)
        attended_by_channel = []
        entropy_by_channel = []
        for name in CHANNELS:
            attended, probabilities = self._attend_channel(
                q, memories[name], memory_masks[name]
            )
            attended_by_channel.append(attended)
            entropy_by_channel.append(
                -(probabilities.float().clamp_min(1e-12).log() * probabilities.float())
                .sum(-1).mean()
            )
        contexts = torch.stack(attended_by_channel, dim=2)  # [B,T,C,H,Dh]
        contexts = contexts.reshape(batch, target_len, len(CHANNELS), -1)
        channel_logits = self.channel_score(contexts.float()).squeeze(-1)
        channel_probabilities = torch.softmax(channel_logits, dim=-1)
        attended = (channel_probabilities.unsqueeze(-1) * contexts.float()).sum(2)
        delta = torch.tanh(self.gate) * self.output(attended)
        entropy = torch.stack(entropy_by_channel).mean()
        mean_channel_weights = channel_probabilities.mean(dim=(0, 1))
        relative = delta.float().norm() / hidden.float().norm().clamp_min(1e-6)
        return (
            hidden + delta.to(hidden.dtype),
            float(entropy.detach().cpu()),
            float(relative.detach().cpu()),
            tuple(float(value) for value in mean_channel_weights.detach().cpu()),
        )

    def alignment_representations(
        self,
        sender_states: Mapping[str, torch.Tensor],
        receiver_token_embeddings: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return channel-preserving sender/receiver representations.

        The sender side uses the same key map that the actual attention path
        reads. The frozen receiver's native token embeddings use the query map.
        Concatenating channels prevents a generic global mean from satisfying
        the contrastive objective while discarding conflict identity.
        """
        students, teachers = [], []
        for name in CHANNELS:
            student = self.key(self.sender_norm(sender_states[name].float())).mean(dim=1)
            teacher = torch.nn.functional.linear(
                self.receiver_norm(receiver_token_embeddings[name].float()),
                self.alignment_teacher_projection,
            ).mean(dim=1)
            students.append(student)
            teachers.append(teacher.detach())
        return torch.cat(students, dim=-1), torch.cat(teachers, dim=-1)


class ReceiverDrivenCrossAttentionBridge(nn.Module):
    """Typed full-state memory queried by selected frozen receiver layers."""

    def __init__(self, receiver_dim: int, sender_dim: int, *, layer_indices: list[int],
                 bottleneck_dim: int = 64, heads: int = 4, gate_init: float = 0.1) -> None:
        super().__init__()
        if not layer_indices or len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer_indices must be non-empty and unique")
        if any(index < 0 for index in layer_indices):
            raise ValueError("layer indices must be non-negative")
        self.layer_indices = tuple(layer_indices)
        self.channel_types = nn.Parameter(torch.zeros(len(CHANNELS), 1, sender_dim))
        nn.init.normal_(self.channel_types, std=sender_dim**-0.5)
        self.layers = nn.ModuleDict({
            str(index): _GatedLatentEnrichment(
                receiver_dim, sender_dim, bottleneck_dim, heads, gate_init
            ) for index in self.layer_indices
        })
        self.last_diagnostics: list[CrossAttentionDiagnostics] = []

    def pack_memory(self, states: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if set(states) != set(CHANNELS):
            raise ValueError(f"full-state bridge requires exactly {CHANNELS}")
        batch = None
        packed = []
        for index, name in enumerate(CHANNELS):
            value = states[name]
            if value.ndim != 3 or value.shape[1] < 1:
                raise ValueError(f"{name} states must be non-empty [B,T,D]")
            batch = value.shape[0] if batch is None else batch
            if value.shape[0] != batch:
                raise ValueError("all memory channels must share a batch size")
            packed.append(value + self.channel_types[index].to(value.dtype))
        memory = torch.cat(packed, dim=1)
        mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        return memory, mask

    def typed_memories(
        self, states: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if set(states) != set(CHANNELS):
            raise ValueError(f"full-state bridge requires exactly {CHANNELS}")
        batch = None
        memories: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for index, name in enumerate(CHANNELS):
            value = states[name]
            if value.ndim != 3 or value.shape[1] < 1:
                raise ValueError(f"{name} states must be non-empty [B,T,D]")
            batch = value.shape[0] if batch is None else batch
            if value.shape[0] != batch:
                raise ValueError("all memory channels must share a batch size")
            memories[name] = value + self.channel_types[index].to(value.dtype)
            masks[name] = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
        return memories, masks

    def alignment_representations(
        self,
        states: Mapping[str, torch.Tensor],
        receiver_token_embeddings: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memories, _ = self.typed_memories(states)
        layer_pairs = [
            adapter.alignment_representations(memories, receiver_token_embeddings)
            for adapter in self.layers.values()
        ]
        return (
            torch.cat([pair[0] for pair in layer_pairs], dim=-1),
            torch.cat([pair[1] for pair in layer_pairs], dim=-1),
        )

    @staticmethod
    def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
        if torch.is_tensor(output):
            return hidden
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        if isinstance(output, list):
            return [hidden, *output[1:]]
        raise TypeError(f"unsupported decoder layer output: {type(output).__name__}")

    @contextmanager
    def inject(self, model: Any, states: Mapping[str, torch.Tensor]) -> Iterator[None]:
        decoder_layers = find_decoder_layers(model)
        if max(self.layer_indices) >= len(decoder_layers):
            raise ValueError(
                f"requested layer {max(self.layer_indices)} from {len(decoder_layers)} decoder layers"
            )
        memories, memory_masks = self.typed_memories(states)
        handles = []
        diagnostics: list[CrossAttentionDiagnostics] = []
        for index in self.layer_indices:
            adapter = self.layers[str(index)]

            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *,
                     layer_index: int = index, enrichment: _GatedLatentEnrichment = adapter) -> Any:
                hidden = output if torch.is_tensor(output) else output[0]
                updated, entropy, relative, channel_weights = enrichment(
                    hidden, memories, memory_masks
                )
                diagnostics.append(CrossAttentionDiagnostics(
                    layer=layer_index,
                    gate=float(torch.tanh(enrichment.gate).detach().cpu()),
                    attention_entropy=entropy,
                    relative_update_norm=relative,
                    channel_weights=channel_weights,
                ))
                return self._replace_hidden(output, updated)

            handles.append(decoder_layers[index].register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self.last_diagnostics = diagnostics

    @property
    def approximate_projection_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
