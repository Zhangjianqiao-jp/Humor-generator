"""Bridge-local causality and full-sequence versus token-wise equivalence.

This does not test the receiver's actual KV cache or quantized generation.
"""
import pytest
import torch
from humor_generator_v35.latent.cross_attention import _GatedLatentEnrichment, CHANNELS


@pytest.mark.parametrize('projection', ['shared', 'per_channel'])
@pytest.mark.parametrize('fusion', ['learned', 'fixed_equal'])
def test_bridge_cannot_read_future_queries_and_matches_tokenwise_application(projection, fusion):
    torch.manual_seed(710)
    bridge = _GatedLatentEnrichment(8, 8, 4, 2, 0.1, fusion, projection)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    memories = {c: torch.randn(2, 3+i, 8) for i, c in enumerate(CHANNELS)}
    masks = {c: torch.ones(m.shape[:2], dtype=torch.bool) for c, m in memories.items()}
    masks['local'][:, -1] = False
    full = bridge(hidden, memories, masks)[0]
    chunked = torch.cat([bridge(hidden[:, i:i+1], memories, masks)[0]
                         for i in range(hidden.shape[1])], dim=1)
    torch.testing.assert_close(full, chunked, rtol=2e-5, atol=2e-7)
    altered = hidden.detach().clone()
    altered[:, 3:] += 100
    torch.testing.assert_close(full[:, :3], bridge(altered, memories, masks)[0][:, :3],
                               rtol=0, atol=0)
    full[:, :3].square().sum().backward()
    assert torch.count_nonzero(hidden.grad[:, 3:]) == 0
    assert hidden.grad[:, :3].abs().sum() > 0
