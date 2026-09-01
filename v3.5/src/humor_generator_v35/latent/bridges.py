"""Parameter- and bandwidth-matched learned latent bridges."""
from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def mean_embedding_norm(embeddings: torch.Tensor, *, chunk_size: int = 4096) -> float:
    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        raise ValueError("receiver embeddings must be a non-empty [V,D] tensor")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    total = torch.zeros((), dtype=torch.float64, device=embeddings.device)
    for start in range(0, embeddings.shape[0], chunk_size):
        total += embeddings[start : start + chunk_size].float().norm(dim=-1).double().sum()
    return float((total / embeddings.shape[0]).cpu())


class _SharedPooler(nn.Module):
    def __init__(
        self,
        sender_dim: int,
        receiver_dim: int,
        *,
        bottleneck_dim: int,
        heads: int,
        target_norm: float,
    ) -> None:
        super().__init__()
        if bottleneck_dim % heads:
            raise ValueError("bottleneck_dim must be divisible by heads")
        if target_norm <= 0:
            raise ValueError("target_norm must be positive")
        self.input_norm = nn.LayerNorm(sender_dim)
        self.key = nn.Linear(sender_dim, bottleneck_dim, bias=False)
        self.value = nn.Linear(sender_dim, bottleneck_dim, bias=False)
        self.attention = nn.MultiheadAttention(bottleneck_dim, heads, batch_first=True)
        self.output = nn.Linear(bottleneck_dim, receiver_dim, bias=False)
        self.output_norm = nn.LayerNorm(receiver_dim)
        self.target_norm = float(target_norm)

    def forward(
        self,
        states: torch.Tensor,
        queries: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if states.ndim != 3 or queries.ndim != 2:
            raise ValueError("states must be [B,T,D] and queries [S,Bottleneck]")
        normalized = self.input_norm(states)
        query = queries.unsqueeze(0).expand(states.shape[0], -1, -1)
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


class LearnedLatentBridge(nn.Module):
    """Untyped bridge with one shared pool and a fixed total slot budget."""

    def __init__(
        self,
        sender_dim: int,
        receiver_dim: int,
        *,
        bottleneck_dim: int = 256,
        slots: int = 24,
        heads: int = 8,
        target_norm: float | None = None,
    ) -> None:
        super().__init__()
        if slots < 1:
            raise ValueError("slots must be positive")
        norm = float(target_norm if target_norm is not None else receiver_dim**0.5)
        self.pooler = _SharedPooler(
            sender_dim, receiver_dim,
            bottleneck_dim=bottleneck_dim, heads=heads, target_norm=norm,
        )
        self.queries = nn.Parameter(torch.randn(slots, bottleneck_dim) * bottleneck_dim**-0.5)

    def forward(self, states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.pooler(states, self.queries, mask)


class TypedLatentBridge(nn.Module):
    """Shared-weight Conflict/Local/Global bridge with equal total slots.

    Only query rows are channel-specific, and the total query count equals the
    untyped bridge. Deterministic type vectors are buffers, not trainable
    parameters, so a 24-slot Learned bridge and 8+8+8 Typed bridge have exactly
    the same trainable parameter count.
    """

    channel_order = ("conflict", "local", "global")

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
        if slots < 1:
            raise ValueError("slots per channel must be positive")
        norm = float(target_norm if target_norm is not None else receiver_dim**0.5)
        self.pooler = _SharedPooler(
            sender_dim, receiver_dim,
            bottleneck_dim=bottleneck_dim, heads=heads, target_norm=norm,
        )
        self.queries = nn.ParameterDict({
            name: nn.Parameter(torch.randn(slots, bottleneck_dim) * bottleneck_dim**-0.5)
            for name in self.channel_order
        })
        type_vectors = torch.zeros(len(self.channel_order), 1, receiver_dim)
        for index in range(len(self.channel_order)):
            type_vectors[index, 0, index] = norm * 0.05
        self.register_buffer("type_vectors", type_vectors, persistent=True)
        self.target_norm = norm

    @property
    def total_slots(self) -> int:
        return sum(int(value.shape[0]) for value in self.queries.values())

    def forward(
        self,
        states: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if set(states) != set(self.channel_order):
            raise ValueError(f"typed bridge requires exactly {self.channel_order}")
        outputs: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self.channel_order):
            value = self.pooler(
                states[name], self.queries[name], None if masks is None else masks.get(name)
            )
            value = value + self.type_vectors[index].to(value.dtype)
            value = value * (
                self.target_norm / value.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            ).to(value.dtype)
            outputs[name] = value
        outputs["all"] = torch.cat([outputs[name] for name in self.channel_order], dim=1)
        return outputs
