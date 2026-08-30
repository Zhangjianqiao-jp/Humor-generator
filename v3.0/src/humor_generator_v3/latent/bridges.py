"""Learned and typed learned latent baselines."""
from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def mean_embedding_norm(embeddings: torch.Tensor, *, chunk_size: int = 4096) -> float:
    """Compute receiver scale without materializing a full fp32 vocab copy."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        raise ValueError("receiver embeddings must be a non-empty [V,D] tensor")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    total = torch.zeros((), dtype=torch.float64, device=embeddings.device)
    for start in range(0, embeddings.shape[0], chunk_size):
        values = embeddings[start : start + chunk_size].float().norm(dim=-1)
        total += values.double().sum()
    return float((total / embeddings.shape[0]).cpu())


class LearnedLatentBridge(nn.Module):
    def __init__(
        self,
        sender_dim: int,
        receiver_dim: int,
        *,
        bottleneck_dim: int = 256,
        slots: int = 8,
        heads: int = 8,
        target_norm: float | None = None,
    ) -> None:
        super().__init__()
        if bottleneck_dim % heads:
            raise ValueError("bottleneck_dim must be divisible by heads")
        self.input_norm = nn.LayerNorm(sender_dim)
        self.key = nn.Linear(sender_dim, bottleneck_dim, bias=False)
        self.value = nn.Linear(sender_dim, bottleneck_dim, bias=False)
        self.queries = nn.Parameter(torch.randn(slots, bottleneck_dim) * bottleneck_dim**-0.5)
        self.attention = nn.MultiheadAttention(bottleneck_dim, heads, batch_first=True)
        self.output = nn.Linear(bottleneck_dim, receiver_dim, bias=False)
        self.output_norm = nn.LayerNorm(receiver_dim)
        self.target_norm = float(target_norm if target_norm is not None else receiver_dim**0.5)
        if self.target_norm <= 0:
            raise ValueError("target_norm must be positive")

    def forward(self, states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError("states must be [B,T,D]")
        normalized = self.input_norm(states)
        query = self.queries.unsqueeze(0).expand(states.shape[0], -1, -1)
        result, _ = self.attention(
            query,
            self.key(normalized),
            self.value(normalized),
            key_padding_mask=None if mask is None else ~mask.bool(),
            need_weights=False,
        )
        output = self.output_norm(self.output(result))
        return output * (
            self.target_norm / output.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ).to(output.dtype)


class TypedLatentBridge(nn.Module):
    """Separate Conflict, Local Association, and Global Association channels."""

    channel_order = ("conflict", "local", "global")

    def __init__(self, sender_dim: int, receiver_dim: int, **kwargs: object) -> None:
        super().__init__()
        target_norm = kwargs.get("target_norm", receiver_dim**0.5)
        if not isinstance(target_norm, (int, float)) or target_norm <= 0:
            raise ValueError("typed bridge target_norm must be positive")
        self.target_norm = float(target_norm)
        self.channels = nn.ModuleDict({
            name: LearnedLatentBridge(sender_dim, receiver_dim, **kwargs)
            for name in self.channel_order
        })
        self.type_embeddings = nn.Parameter(torch.randn(3, 1, receiver_dim) * receiver_dim**-0.5)

    def forward(self, states: dict[str, torch.Tensor], masks: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        if set(states) != set(self.channel_order):
            raise ValueError(f"typed bridge requires exactly {self.channel_order}")
        outputs: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self.channel_order):
            value = self.channels[name](states[name], None if masks is None else masks.get(name))
            value = value + self.type_embeddings[index]
            # Calibrate after type identity is added, not before.
            value = value * (
                self.target_norm / value.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            ).to(value.dtype)
            outputs[name] = value
        outputs["all"] = torch.cat([outputs[name] for name in self.channel_order], dim=1)
        return outputs
