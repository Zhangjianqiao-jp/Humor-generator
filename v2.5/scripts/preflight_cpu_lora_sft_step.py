#!/usr/bin/env python
"""Run one real CPU LoRA SFT step without loading a quantized/GPU base.

This is a pre-submission gate: it exercises the cached Qwen architecture,
processor, image-message budget, collator, LoRA injection, loss, backward and
optimizer interfaces.  QLoRA's CUDA kernels themselves are deliberately left
to the subsequent one-step MIG gate.
"""
from __future__ import annotations

import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.sft_dataset import DEFAULT_SFT_PROMPT, HumorSFTDataset


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_dataset(config: dict[str, Any], processor: Any) -> HumorSFTDataset:
    data = config["data"]
    image_root = data.get("image_root")
    return HumorSFTDataset(
        Path(data["train_path"]),
        processor=processor,
        max_seq_len=int(data["max_seq_len"]),
        image_root=Path(image_root) if image_root else None,
        max_caption_chars=int(data["max_caption_chars"]),
        min_supervised_tokens=int(data["min_supervised_tokens"]),
        normalize_prompt=bool(data.get("normalize_prompt", True)),
        sft_prompt=str(data.get("sft_prompt", DEFAULT_SFT_PROMPT)),
        image_min_pixels=data.get("image_min_pixels"),
        image_max_pixels=data.get("image_max_pixels"),
        validate_images=True,
        max_samples=1,
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if torch.cuda.is_available():
        raise RuntimeError("CPU preflight must not run with a CUDA device visible.")
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    model_config = config["model"]
    lora = model_config["lora"]
    model_name = str(model_config["model_name"])
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    dataset = make_dataset(config, processor)
    batch = dataset.collate_fn([dataset[0]])
    image_grid = batch["image_grid_thw"]
    merge_size = int(processor.image_processor.merge_size)
    visual_tokens = int(image_grid.prod().item() // (merge_size**2))
    image_tokens = int((batch["input_ids"] == processor.image_token_id).sum().item())
    if image_tokens != visual_tokens:
        raise RuntimeError(f"Image-token mismatch before model forward: {image_tokens} != {visual_tokens}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            bias=str(lora.get("bias", "none")),
            task_type="CAUSAL_LM",
        ),
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No LoRA parameters are trainable.")

    model.train()
    inputs = {
        key: value
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"}
    }
    output = model(**inputs)
    loss = output.loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite CPU preflight loss: {loss}")
    loss.backward()
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable):
        raise RuntimeError("A LoRA gradient is missing or non-finite.")
    optimizer = torch.optim.AdamW(trainable, lr=float(config["training"]["learning_rate"]))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    elapsed = time.monotonic() - started
    print(
        "[cpu-preflight] completed one real LoRA optimizer step "
        f"model={model_name} loss={loss.item():.6f} seq={batch['input_ids'].shape[1]} "
        f"visual_tokens={visual_tokens} trainable={sum(p.numel() for p in trainable):,} "
        f"elapsed_seconds={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
