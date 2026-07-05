#!/usr/bin/env python
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lora_sft import (  # noqa: E402
    AdapterCheckpointCallback,
    CadenceGuardCallback,
    FixedGenerationCallback,
    HumorSFTTrainer,
    apply_cli_overrides,
    get_model_device,
    load_config as load_base_config,
    move_batch_to_device,
    prepare_model_and_processor,
    save_run_config,
    training_args_from_config,
)
from src.training.feature_sft_dataset import FeatureHumorSFTDataset  # noqa: E402
from src.training.sft_dataset import DEFAULT_SFT_PROMPT  # noqa: E402


def load_config(config_path: Path) -> dict[str, Any]:
    config = load_base_config(config_path)
    config.setdefault("data", {})
    config["data"].setdefault("normalize_prompt", True)
    config["data"].setdefault("sft_prompt", DEFAULT_SFT_PROMPT)
    config["data"].setdefault("train_context_path", "outputs/analysis/vlm_visual_facts_train.jsonl")
    config["data"].setdefault("val_context_path", "outputs/analysis/vlm_visual_facts_val.jsonl")
    config["data"].setdefault("feature_method", "feature-method")
    config["data"].setdefault("require_context", True)
    return config


def resolve_optional_path(value: str | None) -> Path | None:
    if value in (None, "", "null"):
        return None
    return Path(value)


def build_feature_dataset(
    config: dict[str, Any],
    split: str,
    processor: Any | None,
    max_samples: int | None = None,
    validate_images: bool = True,
) -> FeatureHumorSFTDataset:
    data = config["data"]
    output_dir = Path(config["output"]["output_dir"])
    report_dir = output_dir / "missing_images"
    report_dir.mkdir(parents=True, exist_ok=True)
    path_key = f"{split}_path"
    context_key = f"{split}_context_path"
    report_path = report_dir / f"{split}_missing.jsonl"
    return FeatureHumorSFTDataset(
        path=Path(data[path_key]),
        context_jsonl=Path(data[context_key]),
        feature_method=str(data.get("feature_method", "feature-method")),
        require_context=bool(data.get("require_context", True)),
        processor=processor,
        max_seq_len=int(data.get("max_seq_len", 768)),
        image_root=resolve_optional_path(data.get("image_root")),
        max_caption_chars=int(data.get("max_caption_chars", 240)),
        skip_missing_images=bool(data.get("skip_missing_images", False)),
        normalize_prompt=bool(data.get("normalize_prompt", True)),
        sft_prompt=str(data.get("sft_prompt", DEFAULT_SFT_PROMPT)),
        min_supervised_tokens=int(data.get("min_supervised_tokens", 3)),
        missing_image_report_path=report_path,
        max_samples=max_samples,
        validate_images=validate_images,
    )


def run_debug_data(config: dict[str, Any], num_debug_samples: int, max_train_samples: int | None, max_val_samples: int | None) -> None:
    train_dataset = build_feature_dataset(config, "train", processor=None, max_samples=max_train_samples)
    val_dataset = build_feature_dataset(config, "val", processor=None, max_samples=max_val_samples)
    print(f"[debug-data] train={len(train_dataset)}/{train_dataset.original_count}")
    print(f"[debug-data] val={len(val_dataset)}/{val_dataset.original_count}")
    print("[debug-data] first train samples")
    train_dataset.print_debug_samples(num_debug_samples)
    print("[debug-data] first validation samples")
    val_dataset.print_debug_samples(num_debug_samples)


def run_debug_collator(config: dict[str, Any], num_debug_samples: int) -> None:
    set_seed(config["training"].get("seed", 42))
    model, processor = prepare_model_and_processor(config)
    dataset = build_feature_dataset(config, "train", processor=processor, max_samples=num_debug_samples)
    examples = [dataset[i] for i in range(min(num_debug_samples, len(dataset)))]
    batch = dataset.collate_fn(examples)
    dataset.print_debug_batch(batch, examples, n=num_debug_samples)
    model.eval()
    model_inputs = move_batch_to_device(batch, get_model_device(model))
    with torch.no_grad():
        outputs = model(**model_inputs)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
    print(f"[debug-collator] one forward loss={float(loss.detach().cpu()):.6f}")
    print(f"[debug-collator] supervised tokens per sample={batch['supervised_token_counts'].tolist()}")


def train(
    config_path: Path,
    resume_from_checkpoint: str | None = None,
    debug_one_step: bool = False,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> None:
    config = load_config(config_path)
    set_seed(config["training"].get("seed", 42))

    output_dir = Path(config["output"]["output_dir"])
    save_run_config(config, config_path, output_dir)
    model, processor = prepare_model_and_processor(config)

    if debug_one_step:
        max_train_samples = max_train_samples or max(1, int(config["training"].get("batch_size", 1)))
        max_val_samples = max_val_samples or 1
    elif max_val_samples is None:
        max_val_samples = config.get("evaluation", {}).get("max_eval_samples")

    train_dataset = build_feature_dataset(config, "train", processor=processor, max_samples=max_train_samples)
    val_dataset = build_feature_dataset(config, "val", processor=processor, max_samples=max_val_samples)
    print(
        "[data] feature dataset sizes after validation: "
        f"train={len(train_dataset)}/{train_dataset.original_count}, "
        f"val={len(val_dataset)}/{val_dataset.original_count}"
    )
    train_dataset.print_debug_samples(3)

    args = training_args_from_config(config, debug_one_step=debug_one_step)
    callbacks = [CadenceGuardCallback()]
    if not debug_one_step:
        callbacks.extend(
            [
                AdapterCheckpointCallback(
                    processor=processor,
                    output_dir=output_dir,
                    latest_dir=Path(config["output"].get("latest_adapter_dir", output_dir / "latest")),
                    best_dir=Path(config["output"].get("best_adapter_dir", output_dir / "best_val_loss")),
                ),
                FixedGenerationCallback(
                    val_dataset=val_dataset,
                    processor=processor,
                    output_dir=output_dir,
                    generation_config=config.get("generation", {}),
                    num_samples=int(config.get("evaluation", {}).get("fixed_generation_samples", 5)),
                ),
            ]
        )

    trainer = HumorSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate_fn,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if debug_one_step:
        print("[debug-one-step] completed one optimizer step; no adapter checkpoint saved.")
        return

    final_metrics = trainer.evaluate()
    print(f"[eval] final metrics: {final_metrics}")
    final_dir = Path(config["output"].get("final_adapter_dir", output_dir / "final_lora"))
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[checkpoint] saved final LoRA adapter: {final_dir}")


def main() -> None:
    parser = ArgumentParser(description="Train LoRA-SFT with VLM conservative visual fact features.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sft_vlm_features.yaml"))
    parser.add_argument("--resume_from_checkpoint", "--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--debug-data", action="store_true")
    parser.add_argument("--debug-collator", action="store_true")
    parser.add_argument("--debug-one-step", action="store_true")
    parser.add_argument("--num-debug-samples", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-epochs", type=float, default=None)
    args, unknown = parser.parse_known_args()

    config = load_config(args.config)
    apply_cli_overrides(config, unknown)
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
        print(f"[config] override training.num_epochs={args.num_epochs}")

    if args.debug_data:
        run_debug_data(config, args.num_debug_samples, args.max_train_samples, args.max_val_samples)
        return
    if args.debug_collator:
        run_debug_collator(config, args.num_debug_samples)
        return

    tmp_config_path = args.config
    if config != load_config(args.config):
        tmp_config_path = Path(config["output"]["output_dir"]) / "config_cli_resolved.yaml"
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    train(
        tmp_config_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        debug_one_step=args.debug_one_step,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    main()

