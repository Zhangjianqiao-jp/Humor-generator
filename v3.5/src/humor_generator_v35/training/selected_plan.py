"""Versioned frozen encoding of exactly the selected plan, shared by train/eval."""
from __future__ import annotations

import hashlib
import json
import torch

from ..homer.prompts import text_part
from ..latent.state_capture import AlignedMessageStates
from ..qwen_backend import find_last_decoder_layer
from .public_bridge import assert_context_path_consistent

SELECTED_PLAN_POLICY = "selected_plan_frozen_encoding_v1"
LEGACY_MEMORY_POLICY = "early_planner_trace_v1"
PARTIAL_CAPTION_POLICY = "marked_caption_open_assistant_v1"
LEGACY_TARGET_POLICY = "source_caption_only_no_fabricated_explanation"


def selected_plan_channels(context):
    """local/global are stable slot names here: selected path 0/path 1.

    No early candidate, unselected path, source caption or image enters encoding.
    """
    assert_context_path_consistent(context)
    paths = [f"The free-association chain of {entity} is {' -> '.join(context.selected_paths[entity])}"
             for entity in context.selected_entities]
    if len(paths) != 2:
        raise ValueError("selected-plan encoding requires two selected paths")
    return {"conflict": context.selected_conflict, "local": paths[0], "global": paths[1]}


def selected_plan_sha256(context):
    return hashlib.sha256(json.dumps(selected_plan_channels(context), ensure_ascii=False,
                                    sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def encode_selected_plan(backend, context):
    """Capture final decoder states from frozen, deterministic text-only inputs.

    This is an encoding pass, not generation. It preserves every channel token,
    including fixed chat-template tokens, and returns detached CPU states.
    """
    if backend.model.training or any(p.requires_grad for p in backend.model.parameters()):
        raise ValueError("selected-plan encoder must be frozen and in eval mode")
    result = {}
    for name, text in selected_plan_channels(context).items():
        encoded = backend.encode([{"role": "user", "content": [text_part(text)]}],
                                 add_generation_prompt=False)
        embeddings, positions = backend.multimodal_embeddings_and_positions(encoded)
        captured = []
        def capture(_module, _inputs, output):
            captured.append((output if torch.is_tensor(output) else output[0]).detach())
        handle = find_last_decoder_layer(backend.model).register_forward_hook(capture)
        try:
            backend.model(inputs_embeds=embeddings, attention_mask=encoded["attention_mask"],
                          position_ids=positions, use_cache=False, logits_to_keep=1)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape[:2] != encoded["input_ids"].shape:
            raise RuntimeError("selected-plan token/state alignment failed")
        result[name] = AlignedMessageStates(encoded["input_ids"].detach().cpu(),
                                            captured[0].cpu(), text)
    return result
