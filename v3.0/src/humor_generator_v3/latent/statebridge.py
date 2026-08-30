"""Training-free StateBridge baseline.

Derived from the public Apache-2.0 StateBridge implementation by Peng et al.
The baseline performs centering, regularized whitening, orthogonal Procrustes,
receiver-norm calibration, and vocabulary anchoring.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StateBridgeAlignment(nn.Module):
    def __init__(self, receiver_embeddings: torch.Tensor, *, regularization: float = 1e-3, snap_ratio: float = 0.3) -> None:
        super().__init__()
        if receiver_embeddings.ndim != 2:
            raise ValueError("receiver_embeddings must be [vocab, hidden]")
        if not 0.0 <= snap_ratio <= 1.0:
            raise ValueError("snap_ratio must be in [0,1]")
        self.register_buffer("receiver_embeddings", receiver_embeddings.detach().float(), persistent=False)
        self.regularization = float(regularization)
        self.snap_ratio = float(snap_ratio)
        self.target_norm = float(receiver_embeddings.detach().float().norm(dim=-1).mean())

    @staticmethod
    def _symmetric_power(matrix: torch.Tensor, power: float, epsilon: float = 1e-6) -> torch.Tensor:
        matrix = (matrix + matrix.T) * 0.5
        values, vectors = torch.linalg.eigh(matrix)
        values = values.clamp_min(epsilon).pow(power)
        return (vectors * values.unsqueeze(0)) @ vectors.T

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or token_ids.ndim != 2:
            raise ValueError("hidden_states must be [B,T,D] and token_ids [B,T]")
        if hidden_states.shape[:2] != token_ids.shape:
            raise ValueError("hidden-state/token alignment is exact; trimming is forbidden")
        embeddings = F.embedding(token_ids.to(self.receiver_embeddings.device), self.receiver_embeddings)
        hidden = hidden_states.to(self.receiver_embeddings.device).float()
        batch, length, width = hidden.shape
        h = hidden.reshape(-1, width)
        e = embeddings.reshape(-1, width)
        if h.shape[0] < 2:
            raise ValueError("StateBridge Procrustes requires at least two aligned tokens")
        mean_h, mean_e = h.mean(0, keepdim=True), e.mean(0, keepdim=True)
        centered_h, centered_e = h - mean_h, e - mean_e
        identity = torch.eye(width, device=h.device, dtype=h.dtype)
        cov_h = centered_h.T @ centered_h / h.shape[0] + self.regularization * identity
        cov_e = centered_e.T @ centered_e / e.shape[0] + self.regularization * identity
        inv_h = self._symmetric_power(cov_h, -0.5)
        sqrt_e = self._symmetric_power(cov_e, 0.5)
        white_h, white_e = centered_h @ inv_h, centered_e @ self._symmetric_power(cov_e, -0.5)
        u, _, vh = torch.linalg.svd(white_h.T @ white_e, full_matrices=False)
        rotation = u @ vh
        if torch.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vh
        aligned = (centered_h @ inv_h) @ rotation @ sqrt_e + mean_e
        aligned = aligned.reshape(batch, length, width)
        aligned = aligned * (self.target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        if self.snap_ratio:
            flat = aligned.reshape(-1, width)
            similarities = F.normalize(flat, dim=-1) @ F.normalize(self.receiver_embeddings, dim=-1).T
            nearest = self.receiver_embeddings[similarities.argmax(-1)].reshape_as(aligned)
            aligned = (1.0 - self.snap_ratio) * aligned + self.snap_ratio * nearest
            aligned = aligned * (self.target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return aligned.to(hidden_states.dtype)
