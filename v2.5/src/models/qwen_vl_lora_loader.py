from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_qwen_vl_with_lora(
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    bias: str = "none",
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
    adapter_path: str | Path | None = None,
    is_trainable: bool = True,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    load_in_4bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> tuple[Any, Any]:
    try:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Missing LoRA/Qwen dependencies. Install v1.5/requirements.txt before training."
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    image_processor = getattr(processor, "image_processor", None)
    if image_min_pixels is not None:
        if image_processor is not None and hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = int(image_min_pixels)
        else:
            print(
                "[processor] image_processor.min_pixels is unavailable; "
                "using the per-image message budget instead."
            )
    if image_max_pixels is not None:
        if image_processor is not None and hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = int(image_max_pixels)
        else:
            print(
                "[processor] image_processor.max_pixels is unavailable; "
                "using the per-image message budget instead."
            )
    if image_min_pixels is not None or image_max_pixels is not None:
        print(
            "[processor] image pixel budget: "
            f"min={getattr(image_processor, 'min_pixels', None)}, "
            f"max={getattr(image_processor, 'max_pixels', None)}"
        )

    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
        print(f"[model] loaded LoRA adapter: {adapter_path}")
    else:
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
            print(
                "[model] prepared 4-bit QLoRA base: "
                f"quant_type={bnb_4bit_quant_type}, double_quant={bnb_4bit_use_double_quant}"
            )
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if is_trainable:
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
    else:
        for _, param in model.named_parameters():
            param.requires_grad = False

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_params = sum(param.numel() for _, param in model.named_parameters() if param.requires_grad)
    total_params = sum(param.numel() for _, param in model.named_parameters())
    pct = 100 * trainable_params / max(total_params, 1)
    print(f"[model] trainable parameters: {trainable_params:,}/{total_params:,} ({pct:.4f}%)")
    print(f"[model] trainable parameter name sample: {trainable_names[:20]}")
    unexpected = [name for name in trainable_names if "lora_A" not in name and "lora_B" not in name]
    if unexpected:
        print(f"[model] warning: trainable non-LoRA parameter names: {unexpected[:20]}")
    return model, processor
