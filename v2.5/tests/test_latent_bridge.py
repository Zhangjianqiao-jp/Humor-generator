from __future__ import annotations

import torch
from torch import nn

from src.latent_communication.bridge import LearnedLatentBridge, TypedHomerLatentBridge, inject_latent_slots, insert_constant_slots
from src.latent_communication.homer import HomerPlan, latent_generator_context, text_generator_context
from src.latent_communication.state_capture import GeneratedTokenStateCapture, find_last_decoder_layer
from src.latent_communication.qwen_pipeline import generator_prompt


def test_bridge_shape_mask_and_gradient() -> None:
    bridge = LearnedLatentBridge(12, 16, bottleneck_dim=8, num_slots=3, num_heads=2)
    sender = torch.randn(2, 5, 12)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
    output = bridge(sender, mask)
    assert output.latent_slots.shape == (2, 3, 16)
    assert output.attention_weights.shape == (2, 2, 3, 5)
    output.latent_slots.square().mean().backward()
    assert any(parameter.grad is not None for parameter in bridge.parameters())


def test_inject_latent_slots_masks_prefix_labels() -> None:
    ids = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    embeds = torch.randn(2, 4, 6)
    mask = torch.ones_like(ids)
    labels = ids.clone()
    latent = torch.randn(2, 2, 6)
    result = inject_latent_slots(
        ids,
        embeds,
        mask,
        latent,
        torch.tensor([2, 3]),
        placeholder_token_id=0,
        labels=labels,
    )
    assert result["input_ids"].tolist() == [
        [10, 11, 0, 0, 12, 13],
        [20, 21, 22, 0, 0, 23],
    ]
    assert result["labels"][0, 2:4].tolist() == [-100, -100]
    assert torch.equal(result["inputs_embeds"][1, 3:5], latent[1])


def test_auxiliary_multimodal_token_types_grow_with_latent_prefix() -> None:
    token_types = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]])
    expanded = insert_constant_slots(token_types, torch.tensor([3, 2]), 2, value=0)
    assert expanded.tolist() == [[0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]]


def test_capture_and_layer_discovery() -> None:
    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()

    wrapper = Wrapper()
    assert find_last_decoder_layer(wrapper) is wrapper.model.layers[-1]
    capture = GeneratedTokenStateCapture()
    handle = wrapper.model.layers[-1].register_forward_hook(capture)
    wrapper.model.layers[-1](torch.randn(1, 3, 4))
    handle.remove()
    assert capture.stacked().shape == (1, 1, 4)


def test_communication_modes_do_not_leak_text_into_pure_latent() -> None:
    plan = '{"scene":"cat at keyboard","target":"work"}'
    assert plan in generator_prompt("text", plan)
    assert plan in generator_prompt("hybrid", plan)
    assert plan not in generator_prompt("latent", plan)


def test_homer_typed_bridge_keeps_two_channels() -> None:
    bridge = TypedHomerLatentBridge(12, 16, bottleneck_dim=8, num_slots=3, num_heads=2)
    result = bridge(torch.randn(2, 5, 12), torch.randn(2, 7, 12))
    assert result["conflict_slots"].shape == (2, 3, 16)
    assert result["imagination_slots"].shape == (2, 3, 16)
    assert result["latent_slots"].shape == (2, 6, 16)


def test_homer_latent_prompt_transmits_only_grounding_as_text() -> None:
    plan = HomerPlan("visible cat", "work vs play", '["keyboard","office"]', '["cat","boss"]')
    text = text_generator_context(plan)
    latent = latent_generator_context(plan)
    assert "visible cat" in text and "work vs play" in text
    assert "visible cat" in latent
    assert "work vs play" not in latent
    assert "keyboard" not in latent
