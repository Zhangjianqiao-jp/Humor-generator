"""Load one Qwen2.5-VL base with frozen Planner and Generator adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_shared_qwen_vl_adapters(
    model_name: str,
    planner_adapter: str | Path,
    generator_adapter: str | Path,
    *,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> tuple[Any, Any, Any]:
    """Return a single frozen base carrying two switchable LoRA adapters.

    Sharing the base avoids keeping two copies of the 7B weights in GPU memory.
    Adapter names are fixed to ``planner`` and ``generator`` so callers can
    switch roles explicitly and audit that no policy parameter is trainable.
    """
    planner_adapter = Path(planner_adapter)
    generator_adapter = Path(generator_adapter)
    for path in (planner_adapter, generator_adapter):
        if not path.is_dir():
            raise FileNotFoundError(f"LoRA adapter directory does not exist: {path}")
    try:
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError("Qwen/PEFT inference dependencies are unavailable") from exc

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    model = PeftModel.from_pretrained(
        base,
        planner_adapter,
        adapter_name="planner",
        is_trainable=False,
    )
    model.load_adapter(generator_adapter, adapter_name="generator", is_trainable=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    return model, processor, process_vision_info


def assert_only_bridge_trainable(model: Any, bridge: torch.nn.Module) -> dict[str, int]:
    policy_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bridge_trainable = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    if policy_trainable:
        names = [name for name, p in model.named_parameters() if p.requires_grad][:10]
        raise RuntimeError(f"Frozen-policy contract violated; trainable parameters: {names}")
    if bridge_trainable < 1:
        raise RuntimeError("Bridge has no trainable parameters")
    return {"policy_trainable": policy_trainable, "bridge_trainable": bridge_trainable}
