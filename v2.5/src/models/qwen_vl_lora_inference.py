from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_qwen_vl_lora_for_inference(
    model_name: str,
    adapter_dir: Path | None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
    load_in_4bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> tuple[Any, Any, Any]:
    if adapter_dir is not None and not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter_dir}")
    try:
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Missing inference dependencies. Install v1.5/requirements.txt first."
        ) from exc

    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    if adapter_dir is None:
        model = base_model
        print("[model] using base model without a LoRA adapter")
    else:
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        print(f"[model] loaded LoRA adapter: {adapter_dir}")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    return model, processor, process_vision_info
