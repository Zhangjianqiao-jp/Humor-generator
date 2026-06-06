from __future__ import annotations

from typing import Any


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
) -> tuple[Any, Any]:
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Missing LoRA/Qwen dependencies. Install v1.5/requirements.txt before training."
        ) from exc

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "lora_" not in name:
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
