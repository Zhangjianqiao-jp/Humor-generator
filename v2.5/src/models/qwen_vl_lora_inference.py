from __future__ import annotations

from pathlib import Path
from typing import Any


def load_qwen_vl_lora_for_inference(
    model_name: str,
    adapter_dir: Path | None = None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[Any, Any, Any]:
    try:
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
    model = base_model
    if adapter_dir is not None:
        if not adapter_dir.exists():
            raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter_dir}")
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("Missing PEFT dependency required to load a LoRA adapter.") from exc
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    processor_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
    return model, processor, process_vision_info
