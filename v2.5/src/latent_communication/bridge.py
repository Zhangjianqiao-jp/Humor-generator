"""Small trainable bridges for planner-to-generator latent communication.

The two 7B policies stay frozen.  The bridge compresses the planner's
generated-token states into a fixed number of vectors in the Generator input
embedding space.  It is deliberately independent of Qwen so its tensor
contract can be tested without loading either checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BridgeOutput:
    latent_slots: torch.Tensor
    attention_weights: torch.Tensor


class LearnedLatentBridge(nn.Module):
    """Query-resampler bridge with a narrow trainable bottleneck.

    Sender states have shape ``[batch, sender_tokens, sender_dim]``.  Learned
    queries attend to those states and emit ``num_slots`` continuous tokens in
    the receiver's embedding space.  Norm calibration keeps the result close
    to the scale the frozen receiver encountered during pretraining.
    """

    def __init__(
        self,
        sender_dim: int,
        receiver_dim: int,
        *,
        bottleneck_dim: int = 512,
        num_slots: int = 16,
        num_heads: int = 8,
        dropout: float = 0.05,
        target_norm: float = 1.0,
    ) -> None:
        super().__init__()
        if sender_dim < 1 or receiver_dim < 1 or bottleneck_dim < 1 or num_slots < 1:
            raise ValueError("Bridge dimensions and num_slots must be positive")
        if bottleneck_dim % num_heads:
            raise ValueError("bottleneck_dim must be divisible by num_heads")
        self.sender_dim = sender_dim
        self.receiver_dim = receiver_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_slots = num_slots
        self.target_norm = float(target_norm)

        self.sender_norm = nn.LayerNorm(sender_dim)
        self.input_projection = nn.Linear(sender_dim, bottleneck_dim)
        self.queries = nn.Parameter(torch.empty(num_slots, bottleneck_dim))
        self.cross_attention = nn.MultiheadAttention(
            bottleneck_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(bottleneck_dim)
        self.ffn_norm = nn.LayerNorm(bottleneck_dim)
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 4, bottleneck_dim),
        )
        self.output_projection = nn.Linear(bottleneck_dim, receiver_dim)
        # A learned global scale prevents uncontrolled prefix norms while still
        # allowing the bridge to become stronger when downstream NLL supports it.
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.queries, mean=0.0, std=self.bottleneck_dim**-0.5)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        sender_states: torch.Tensor,
        sender_attention_mask: torch.Tensor | None = None,
    ) -> BridgeOutput:
        if sender_states.ndim != 3 or sender_states.shape[-1] != self.sender_dim:
            raise ValueError(
                "sender_states must be [batch, tokens, sender_dim], got "
                f"{tuple(sender_states.shape)}"
            )
        batch, token_count, _ = sender_states.shape
        if token_count < 1:
            raise ValueError("sender_states must contain at least one token")
        if sender_attention_mask is None:
            sender_attention_mask = torch.ones(
                batch, token_count, dtype=torch.bool, device=sender_states.device
            )
        if sender_attention_mask.shape != (batch, token_count):
            raise ValueError("sender_attention_mask shape does not match sender_states")
        valid = sender_attention_mask.to(torch.bool)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every sender example must contain at least one valid token")

        memory = self.input_projection(self.sender_norm(sender_states))
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        attended, weights = self.cross_attention(
            self.query_norm(queries),
            memory,
            memory,
            key_padding_mask=~valid,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = queries + attended
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        slots = self.output_projection(hidden)

        norm = slots.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        slots = slots * (self.target_norm / norm).to(slots.dtype)
        slots = slots * self.log_scale.exp().clamp(max=10.0).to(slots.dtype)
        return BridgeOutput(latent_slots=slots, attention_weights=weights)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class TypedHomerLatentBridge(nn.Module):
    """Independent conflict/imagination channels with explicit type identity.

    HOMER assigns different functions to script opposition and associative
    imagination.  Sharing one untyped resampler lets the receiver ignore that
    distinction, so each field gets its own bridge and learned channel marker.
    """

    def __init__(self, sender_dim: int, receiver_dim: int, **bridge_kwargs: object) -> None:
        super().__init__()
        self.conflict = LearnedLatentBridge(sender_dim, receiver_dim, **bridge_kwargs)
        self.imagination = LearnedLatentBridge(sender_dim, receiver_dim, **bridge_kwargs)
        self.type_embeddings = nn.Parameter(torch.empty(2, 1, receiver_dim))
        nn.init.normal_(self.type_embeddings, std=receiver_dim**-0.5)

    def forward(
        self,
        conflict_states: torch.Tensor,
        imagination_states: torch.Tensor,
        conflict_mask: torch.Tensor | None = None,
        imagination_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        conflict = self.conflict(conflict_states, conflict_mask).latent_slots
        imagination = self.imagination(imagination_states, imagination_mask).latent_slots
        conflict = conflict + self.type_embeddings[0]
        imagination = imagination + self.type_embeddings[1]
        return {
            "conflict_slots": conflict,
            "imagination_slots": imagination,
            "latent_slots": torch.cat([conflict, imagination], dim=1),
        }

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def inject_latent_slots(
    input_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_slots: torch.Tensor,
    insertion_indices: torch.Tensor,
    *,
    placeholder_token_id: int,
    labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Insert latent slots before a per-example sequence position.

    Placeholder IDs keep Qwen's multimodal RoPE and visual-token accounting
    well-defined.  Their embeddings are replaced by bridge outputs and their
    labels are always ``-100``.  ``insertion_indices`` normally marks the end
    of the chat generation prompt, immediately before the caption answer.
    """
    if input_ids.ndim != 2 or token_embeddings.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("input_ids/attention_mask must be rank 2 and embeddings rank 3")
    batch, seq_len = input_ids.shape
    if token_embeddings.shape[:2] != (batch, seq_len):
        raise ValueError("token_embeddings shape does not match input_ids")
    if attention_mask.shape != (batch, seq_len):
        raise ValueError("attention_mask shape does not match input_ids")
    if latent_slots.ndim != 3 or latent_slots.shape[0] != batch:
        raise ValueError("latent_slots must be [batch, slots, hidden]")
    if latent_slots.shape[-1] != token_embeddings.shape[-1]:
        raise ValueError("latent slot and receiver embedding dimensions differ")
    if insertion_indices.shape != (batch,):
        raise ValueError("insertion_indices must have one element per batch row")
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("labels shape does not match input_ids")

    slots = latent_slots.shape[1]
    output_ids, output_embeds, output_masks, output_labels = [], [], [], []
    for row in range(batch):
        index = int(insertion_indices[row].item())
        if index < 0 or index > seq_len:
            raise ValueError(f"invalid insertion index {index} for sequence length {seq_len}")
        placeholders = input_ids.new_full((slots,), int(placeholder_token_id))
        output_ids.append(torch.cat([input_ids[row, :index], placeholders, input_ids[row, index:]]))
        output_embeds.append(
            torch.cat(
                [token_embeddings[row, :index], latent_slots[row], token_embeddings[row, index:]],
                dim=0,
            )
        )
        output_masks.append(
            torch.cat(
                [attention_mask[row, :index], attention_mask.new_ones(slots), attention_mask[row, index:]]
            )
        )
        if labels is not None:
            ignored = labels.new_full((slots,), -100)
            output_labels.append(torch.cat([labels[row, :index], ignored, labels[row, index:]]))

    result = {
        "input_ids": torch.stack(output_ids),
        "inputs_embeds": torch.stack(output_embeds),
        "attention_mask": torch.stack(output_masks),
    }
    if labels is not None:
        result["labels"] = torch.stack(output_labels)
    return result


def insert_constant_slots(
    tensor: torch.Tensor,
    insertion_indices: torch.Tensor,
    num_slots: int,
    *,
    value: int | float = 0,
) -> torch.Tensor:
    """Insert constant positions into a batch-first sequence tensor.

    Qwen2.5-VL emits auxiliary sequence tensors such as
    ``mm_token_type_ids``.  They must grow with the attention mask whenever a
    latent prefix is inserted.  Latent slots are ordinary text positions, so
    their multimodal token type is zero.
    """
    if tensor.ndim != 2:
        raise ValueError("constant slot insertion expects a [batch, sequence] tensor")
    if insertion_indices.shape != (tensor.shape[0],):
        raise ValueError("insertion_indices must have one element per batch row")
    if num_slots < 1:
        raise ValueError("num_slots must be positive")
    rows = []
    for row in range(tensor.shape[0]):
        index = int(insertion_indices[row].item())
        if index < 0 or index > tensor.shape[1]:
            raise ValueError(f"invalid insertion index {index}")
        fill = tensor.new_full((num_slots,), value)
        rows.append(torch.cat([tensor[row, :index], fill, tensor[row, index:]]))
    return torch.stack(rows)
