#!/usr/bin/env python
from __future__ import annotations

import sys
import inspect
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


def train(config_path: Path, resume_from_checkpoint: str | None = None) -> None:
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

    report_dir = Path(config["output"]["output_dir"]) / "missing_images"
    report_dir.mkdir(parents=True, exist_ok=True)
    skip_missing_images = config["data"].get("skip_missing_images", False)
    train_dataset = HumorSFTDataset(
        Path(config["data"]["train_path"]),
        processor,
        config["data"]["max_seq_len"],
        skip_missing_images=skip_missing_images,
        missing_image_report_path=report_dir / "train_missing.jsonl",
    )
    val_dataset = HumorSFTDataset(
        Path(config["data"]["val_path"]),
        processor,
        config["data"]["max_seq_len"],
        skip_missing_images=skip_missing_images,
        missing_image_report_path=report_dir / "val_missing.jsonl",
    )
    print(
        "Dataset sizes after image validation: "
        f"train={len(train_dataset)}/{train_dataset.original_count} "
        f"(missing={train_dataset.missing_image_count}), "
        f"val={len(val_dataset)}/{val_dataset.original_count} "
        f"(missing={val_dataset.missing_image_count})"
    )
    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty after filtering missing images. "
            "Check data.train_path and the image paths inside the JSONL file. "
            "Most likely image_base_dir points to the wrong location on this machine."
        )
    if len(val_dataset) == 0:
        raise ValueError(
            "Validation dataset is empty after filtering missing images. "
            "Check data.val_path and the image paths inside the JSONL file."
        )

    training_args_kwargs = {
        "output_dir": config["output"]["output_dir"],
        "per_device_train_batch_size": config["training"]["batch_size"],
        "per_device_eval_batch_size": config["training"]["batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "num_train_epochs": config["training"]["num_epochs"],
        "learning_rate": config["training"]["learning_rate"],
        "warmup_ratio": config["training"]["warmup_ratio"],
        "weight_decay": config["training"]["weight_decay"],
        "max_grad_norm": config["training"]["max_grad_norm"],
        "logging_steps": config["training"]["logging_steps"],
        "eval_steps": config["training"]["eval_steps"],
        "save_steps": config["training"]["save_steps"],
        "save_total_limit": config["training"]["save_total_limit"],
        "save_strategy": "steps",
        "bf16": config["training"]["bf16"] and torch.cuda.is_available(),
        "fp16": config["training"]["fp16"] and torch.cuda.is_available(),
        "optim": config["training"]["optim"],
        "remove_unused_columns": False,
        "report_to": [],
    }
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_args_params:
        training_args_kwargs["eval_strategy"] = "steps"
    else:
        training_args_kwargs["evaluation_strategy"] = "steps"

    args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate_fn,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(config["output"]["final_adapter_dir"])
    processor.save_pretrained(config["output"]["final_adapter_dir"])


def main() -> None:
    parser = ArgumentParser(description="Train V1.5 LoRA-SFT humor generator.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sft.yaml"))
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a Trainer checkpoint, for example outputs/lora_sft_v1_5/checkpoint-250.",
    )
    args = parser.parse_args()
    train(args.config, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
