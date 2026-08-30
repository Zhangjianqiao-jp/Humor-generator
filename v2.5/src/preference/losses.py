from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PreferenceLossOutput:
    loss: torch.Tensor
    pair_losses: torch.Tensor
    preference_logits: torch.Tensor
    chosen_reward: torch.Tensor
    rejected_reward: torch.Tensor
    chosen_logp: torch.Tensor
    rejected_logp: torch.Tensor


def preference_loss(
    *,
    objective: str,
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_tokens: torch.Tensor,
    rejected_tokens: torch.Tensor,
    beta: float,
    reference_chosen_logp: torch.Tensor | None = None,
    reference_rejected_logp: torch.Tensor | None = None,
    simpo_gamma: float = 0.5,
    anchor_weight: float = 0.1,
    anchor_length_normalize: bool = True,
) -> PreferenceLossOutput:
    """Compute one of the controlled offline pairwise preference objectives.

    `anchored` is explicitly the project's DPO plus positive chosen-likelihood
    regularizer.  It is CPO-style, not claimed to be a line-for-line CPO
    reproduction.
    """
    objective = objective.lower().replace("-", "_")
    if beta <= 0:
        raise ValueError("beta must be positive")
    chosen_tokens = chosen_tokens.clamp_min(1).to(chosen_logp.dtype)
    rejected_tokens = rejected_tokens.clamp_min(1).to(rejected_logp.dtype)
    chosen_average = chosen_logp / chosen_tokens
    rejected_average = rejected_logp / rejected_tokens

    if objective in {"dpo", "ipo", "anchored", "positive_anchored"}:
        if reference_chosen_logp is None or reference_rejected_logp is None:
            raise ValueError(f"{objective} requires frozen reference log probabilities")
        policy_logratio = chosen_logp - rejected_logp
        reference_logratio = reference_chosen_logp - reference_rejected_logp
        logratio_advantage = policy_logratio - reference_logratio
        preference_logits = beta * logratio_advantage
        chosen_reward = beta * (chosen_logp - reference_chosen_logp)
        rejected_reward = beta * (rejected_logp - reference_rejected_logp)
        if objective == "ipo":
            pair_loss = (logratio_advantage - 1.0 / (2.0 * beta)).square()
        else:
            pair_loss = -F.logsigmoid(preference_logits)
        if objective in {"anchored", "positive_anchored"}:
            chosen_anchor = -chosen_average if anchor_length_normalize else -chosen_logp
            pair_loss = pair_loss + anchor_weight * chosen_anchor
    elif objective == "simpo":
        preference_logits = beta * (chosen_average - rejected_average) - simpo_gamma
        pair_loss = -F.logsigmoid(preference_logits)
        chosen_reward = beta * chosen_average
        rejected_reward = beta * rejected_average
    else:
        raise ValueError(f"Unsupported preference objective: {objective}")

    return PreferenceLossOutput(
        loss=pair_loss.mean(),
        pair_losses=pair_loss,
        preference_logits=preference_logits,
        chosen_reward=chosen_reward,
        rejected_reward=rejected_reward,
        chosen_logp=chosen_logp,
        rejected_logp=rejected_logp,
    )
