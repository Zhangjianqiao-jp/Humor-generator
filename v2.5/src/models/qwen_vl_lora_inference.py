from __future__ import annotations

from pathlib import Path
from typing import Any


def load_qwen_vl_lora_for_inference(
    model_name: str,
    adapter_dir: Path,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
) -> tuple[Any, Any, Any]:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter_dir}")
    try:
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Missing inference dependencies. Install v1.5/requirements.txt first."
        ) from exc

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    return model, processor, process_vision_info
