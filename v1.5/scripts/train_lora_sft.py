#!/usr/bin/env python
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from transformers import Trainer, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_loader import load_qwen_vl_with_lora
from src.training.sft_dataset import HumorSFTDataset


def train(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])

    model, processor = load_qwen_vl_with_lora(
        model_name=config["model"]["model_name"],
        lora_rank=config["model"]["lora"]["rank"],
        lora_alpha=config["model"]["lora"]["alpha"],
        lora_dropout=config["model"]["lora"]["dropout"],
        target_modules=config["model"]["lora"]["target_modules"],
        bias=config["model"]["lora"]["bias"],
        device_map=config["model"]["device_map"],
        torch_dtype=config["model"]["torch_dtype"],
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )

    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_dataset = HumorSFTDataset(Path(config["data"]["train_path"]), processor, config["data"]["max_seq_len"])
    val_dataset = HumorSFTDataset(Path(config["data"]["val_path"]), processor, config["data"]["max_seq_len"])

    args = TrainingArguments(
        output_dir=config["output"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],
        logging_steps=config["training"]["logging_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=config["training"]["bf16"] and torch.cuda.is_available(),
        fp16=config["training"]["fp16"] and torch.cuda.is_available(),
        optim=config["training"]["optim"],
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate_fn,
    )
    trainer.train()
    model.save_pretrained(config["output"]["final_adapter_dir"])
    processor.save_pretrained(config["output"]["final_adapter_dir"])


def main() -> None:
    parser = ArgumentParser(description="Train V1.5 LoRA-SFT humor generator.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sft.yaml"))
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
