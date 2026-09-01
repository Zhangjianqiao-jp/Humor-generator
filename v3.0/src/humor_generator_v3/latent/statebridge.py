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
        self.last_diagnostics: dict[str, int | str] = {}

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
        if h.shape[0] >= width:
            raise ValueError(
                "formal StateBridge expects a token-bounded low-rank prefix (tokens < hidden width)"
            )

        # Compute the exact action of the dense whitening/Procrustes/coloring
        # transform inside the transmitted rows' <=T-1 dimensional subspace.
        # With T << D this replaces O(D^3) decompositions by O(T^2 D).
        uh, sh, vhh = torch.linalg.svd(centered_h, full_matrices=False)
        ue, se, vhe = torch.linalg.svd(centered_e, full_matrices=False)
        tolerance_h = max(centered_h.shape) * torch.finfo(sh.dtype).eps * sh.max()
        tolerance_e = max(centered_e.shape) * torch.finfo(se.dtype).eps * se.max()
        rank_h = int((sh > tolerance_h).sum())
        rank_e = int((se > tolerance_e).sum())
        self.last_diagnostics = {
            "sender_rank": rank_h,
            "receiver_rank": rank_e,
            "solver": "low_rank_exact" if rank_h == rank_e else "dense_rank_mismatch_fallback",
        }
        if rank_h < 1 or rank_e < 1:
            raise RuntimeError("StateBridge received a rank-zero aligned message")
        if rank_h != rank_e:
            # Repeated receiver token IDs can make E lower-rank than H.  The
            # orthogonal complement of the Procrustes map is then non-unique;
            # a thin pseudo-inverse would silently select a different method.
            # Preserve the public dense transform in this uncommon case.
            identity = torch.eye(width, device=h.device, dtype=h.dtype)
            cov_h = centered_h.T @ centered_h / h.shape[0] + self.regularization * identity
            cov_e = centered_e.T @ centered_e / e.shape[0] + self.regularization * identity
            inv_h = self._symmetric_power(cov_h, -0.5)
            inv_e = self._symmetric_power(cov_e, -0.5)
            sqrt_e = self._symmetric_power(cov_e, 0.5)
            white_h, white_e = centered_h @ inv_h, centered_e @ inv_e
            u, _, vh = torch.linalg.svd(white_h.T @ white_e, full_matrices=False)
            rotation = u @ vh
            if torch.det(rotation) < 0:
                u[:, -1] *= -1
                rotation = u @ vh
            aligned = white_h @ rotation @ sqrt_e + mean_e
            aligned = aligned.reshape(batch, length, width)
            aligned = aligned * (
                self.target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            )
            return self._snap(aligned).to(hidden_states.dtype)
        uh, sh, vhh = uh[:, :rank_h], sh[:rank_h], vhh[:rank_h]
        ue, se, vhe = ue[:, :rank_e], se[:rank_e], vhe[:rank_e]
        count = h.shape[0]
        scale_h = sh / torch.sqrt(sh.square() / count + self.regularization)
        scale_e = se / torch.sqrt(se.square() / count + self.regularization)
        core = scale_h[:, None] * (uh.T @ ue) * scale_e[None, :]
        core_u, _, core_vh = torch.linalg.svd(core, full_matrices=False)
        colored_e = torch.sqrt(se.square() / count + self.regularization)
        coefficients = ((uh * scale_h) @ core_u) @ core_vh
        aligned = (coefficients * colored_e) @ vhe + mean_e
        aligned = aligned.reshape(batch, length, width)
        aligned = aligned * (self.target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return self._snap(aligned).to(hidden_states.dtype)

    def _snap(self, aligned: torch.Tensor) -> torch.Tensor:
        if self.snap_ratio:
            width = aligned.shape[-1]
            flat = aligned.reshape(-1, width)
            similarities = F.normalize(flat, dim=-1) @ F.normalize(self.receiver_embeddings, dim=-1).T
            nearest = self.receiver_embeddings[similarities.argmax(-1)].reshape_as(aligned)
            aligned = (1.0 - self.snap_ratio) * aligned + self.snap_ratio * nearest
            aligned = aligned * (self.target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return aligned
