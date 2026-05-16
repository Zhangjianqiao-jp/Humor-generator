from __future__ import annotations


def load_qwen_model(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device_map: str = "auto", torch_dtype: str = "auto"):
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Qwen2.5-VL dependencies. Install recent transformers and qwen-vl-utils."
        ) from exc

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, device_map=device_map, torch_dtype=torch_dtype)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor, process_vision_info
    except Exception as exc:
        raise RuntimeError(
            f"Failed loading model '{model_name}'. Ensure model is downloaded and compatible versions are installed."
        ) from exc
