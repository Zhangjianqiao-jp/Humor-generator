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
    model.print_trainable_parameters()
    return model, processor
