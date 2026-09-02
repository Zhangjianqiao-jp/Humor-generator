"""Compatibility bridge for evaluating the pre-hierarchical v1 checkpoint.

This module is evaluation-only.  It preserves the v1 single-memory softmax
architecture so an old checkpoint is not silently evaluated with the v2/v3
hierarchical architecture.  New training must use ``cross_attention.py``.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import torch
from torch import nn

from ..qwen_backend import find_decoder_layers
from .cross_attention import CHANNELS, CrossAttentionDiagnostics


class _LegacyGatedLatentEnrichment(nn.Module):
    def __init__(self, receiver_dim: int, sender_dim: int, bottleneck_dim: int,
                 heads: int, gate_init: float) -> None:
        super().__init__()
        if bottleneck_dim % heads:
            raise ValueError("bottleneck_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = bottleneck_dim // heads
        self.receiver_norm = nn.LayerNorm(receiver_dim)
        self.sender_norm = nn.LayerNorm(sender_dim)
        self.query = nn.Linear(receiver_dim, bottleneck_dim, bias=False)
        self.key = nn.Linear(sender_dim, bottleneck_dim, bias=False)
        self.value = nn.Linear(sender_dim, bottleneck_dim, bias=False)
        self.output = nn.Linear(bottleneck_dim, receiver_dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        channel_lengths: tuple[int, int, int],
    ) -> tuple[torch.Tensor, float, float, tuple[float, float, float]]:
        batch, target_len, _ = hidden.shape
        source_len = memory.shape[1]
        q = self.query(self.receiver_norm(hidden.float())).view(
            batch, target_len, self.heads, self.head_dim
        ).transpose(1, 2)
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
        attended = torch.matmul(probabilities, v).transpose(1, 2).reshape(
            batch, target_len, -1
        )
        delta = torch.tanh(self.gate) * self.output(attended)
        entropy = -(
            probabilities.float().clamp_min(1e-12).log() * probabilities.float()
        ).sum(-1).mean()
        relative = delta.float().norm() / hidden.float().norm().clamp_min(1e-6)
        weights = []
        start = 0
        for length in channel_lengths:
            weights.append(probabilities[..., start : start + length].sum(-1).mean())
            start += length
        channel_weights = tuple(float(value.detach().cpu()) for value in weights)
        return (
            hidden + delta.to(hidden.dtype),
            float(entropy.detach().cpu()),
            float(relative.detach().cpu()),
            channel_weights,
        )


class LegacyV1ReceiverDrivenCrossAttentionBridge(nn.Module):
    """Exact v1 single-softmax bridge, retained only for checkpoint re-evaluation."""

    def __init__(self, receiver_dim: int, sender_dim: int, *, layer_indices: list[int],
                 bottleneck_dim: int = 64, heads: int = 4, gate_init: float = 0.1) -> None:
        super().__init__()
        if not layer_indices or len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer_indices must be non-empty and unique")
        self.layer_indices = tuple(layer_indices)
        self.channel_types = nn.Parameter(torch.zeros(len(CHANNELS), 1, sender_dim))
        nn.init.normal_(self.channel_types, std=sender_dim**-0.5)
        self.layers = nn.ModuleDict({
            str(index): _LegacyGatedLatentEnrichment(
                receiver_dim, sender_dim, bottleneck_dim, heads, gate_init
            ) for index in self.layer_indices
        })
        self.last_diagnostics: list[CrossAttentionDiagnostics] = []

    def pack_memory(self, states: Mapping[str, torch.Tensor]) -> tuple[
        torch.Tensor, torch.Tensor, tuple[int, int, int]
    ]:
        if set(states) != set(CHANNELS):
            raise ValueError(f"full-state bridge requires exactly {CHANNELS}")
        lengths = tuple(int(states[name].shape[1]) for name in CHANNELS)
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
        return memory, mask, lengths

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
        memory, memory_mask, channel_lengths = self.pack_memory(states)
        handles = []
        diagnostics: list[CrossAttentionDiagnostics] = []
        for index in self.layer_indices:
            adapter = self.layers[str(index)]

            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *,
                     layer_index: int = index,
                     enrichment: _LegacyGatedLatentEnrichment = adapter) -> Any:
                hidden = output if torch.is_tensor(output) else output[0]
                updated, entropy, relative, channel_weights = enrichment(
                    hidden, memory, memory_mask, channel_lengths
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
