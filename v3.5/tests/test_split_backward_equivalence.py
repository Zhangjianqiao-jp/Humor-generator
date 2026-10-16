"""Compare production recomputation gradients with an independent full graph."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from humor_generator_v35.training.cross_attention_bridge import ReceiverCrossAttentionTask


@pytest.mark.parametrize('margin_weight', [0.0, 0.5, 2.0])
@pytest.mark.parametrize('loss_scale', [1.0, 0.25])
def test_caption_split_backward_equals_full_objective(margin_weight, loss_scale):
    generator = torch.Generator().manual_seed(817)
    matched_features = torch.randn(2, 3, 4, generator=generator)
    shuffled_features = torch.randn(2, 3, 4, generator=generator)
    initial = torch.randn(4, 5, generator=generator) * 0.3
    targets = torch.tensor([[0, 1, 4], [2, 3, 0]])
    teacher = torch.randn(2, 3, 5, generator=generator, requires_grad=True)
    temperature, margin = 1.7, 0.2

    class Task(ReceiverCrossAttentionTask):
        def _states(self, cluster):
            assert cluster == 'donor'
            return shuffled_features

        def _logits(self, embeddings, mask, positions, targets, states):
            self.bridge.last_diagnostics = []
            return torch.tanh(states @ self.theta)

    task = Task.__new__(Task)
    task.theta = nn.Parameter(initial.clone())
    task.bridge = SimpleNamespace(last_diagnostics=[])
    task.loss_config = dict(caption_nll=1.0, text_teacher_forward_kl=0.5,
                            matched_shuffled_margin=margin_weight,
                            margin=margin, temperature=temperature)
    metrics = task._forward_metrics(
        (None, None, None, targets, teacher, matched_features),
        'donor', backward=True, loss_scale=loss_scale,
    )

    # Independent definition: no detached coefficient, no production loss helpers,
    # and both memory branches coexist in one autograd graph.
    reference = nn.Parameter(initial.clone())
    matched = torch.tanh(matched_features @ reference)
    shuffled = torch.tanh(shuffled_features @ reference)
    nll = F.cross_entropy(matched.reshape(-1, 5), targets.reshape(-1))
    teacher_p = F.softmax(teacher.detach() / temperature, dim=-1)
    kl = F.kl_div(F.log_softmax(matched / temperature, dim=-1),
                  teacher_p, reduction='none').sum(-1).mean() * temperature**2
    matched_score = F.log_softmax(matched, dim=-1).gather(-1, targets[..., None]).mean((1, 2))
    shuffled_score = F.log_softmax(shuffled, dim=-1).gather(-1, targets[..., None]).mean((1, 2))
    ranking = F.softplus(margin - matched_score + shuffled_score).mean()
    total = nll + 0.5 * kl + margin_weight * ranking
    (loss_scale * total).backward()

    torch.testing.assert_close(task.theta.grad, reference.grad, rtol=2e-5, atol=2e-7)
    assert metrics['total'] == pytest.approx(total.item(), rel=2e-6, abs=2e-7)
    assert teacher.grad is None
