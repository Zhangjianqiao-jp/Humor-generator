#!/usr/bin/env python
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

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
    prepare_model_and_processor,
    save_run_config,
    training_args_from_config,
)
from src.training.guided_humor_sft_dataset import GuidedHumorSFTDataset  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(
    config: dict[str, Any],
    split: str,
    processor: Any | None,
) -> GuidedHumorSFTDataset:
    data = config["data"]
    output_dir = Path(config["output"]["output_dir"])
    return GuidedHumorSFTDataset(
        path=Path(data[f"{split}_path"]),
        guidance_jsonl=Path(data[f"{split}_guidance_path"]),
        require_guidance=True,
        processor=processor,
        max_seq_len=int(data.get("max_seq_len", 512)),
        image_root=None,
        max_caption_chars=int(data.get("max_caption_chars", 160)),
        skip_missing_images=True,
        normalize_prompt=True,
        sft_prompt=str(data["sft_prompt"]),
        min_supervised_tokens=int(data.get("min_supervised_tokens", 3)),
        missing_image_report_path=output_dir / "missing_images" / f"{split}.jsonl",
    )


def train(config_path: Path, debug_one_step: bool = False) -> None:
    config = load_config(config_path)
    set_seed(int(config["training"].get("seed", 42)))
    output_dir = Path(config["output"]["output_dir"])
    save_run_config(config, config_path, output_dir)

    # This always constructs a new PEFT adapter from model.model_name. There is
    # intentionally no resume or adapter input in this training entry point.
    model, processor = prepare_model_and_processor(config)
    train_data = build_dataset(config, "train", processor)
    val_data = build_dataset(config, "val", processor)
    train_data.print_debug_samples(3)

    args = training_args_from_config(config, debug_one_step=debug_one_step)
    callbacks = [CadenceGuardCallback()]
    if not debug_one_step:
        callbacks.extend(
            [
                AdapterCheckpointCallback(
                    processor=processor,
                    output_dir=output_dir,
                    latest_dir=Path(config["output"]["latest_adapter_dir"]),
                    best_dir=Path(config["output"]["best_adapter_dir"]),
                ),
                FixedGenerationCallback(
                    val_dataset=val_data,
                    processor=processor,
                    output_dir=output_dir,
                    generation_config=config["generation"],
                    num_samples=int(config["evaluation"].get("fixed_generation_samples", 8)),
                ),
            ]
        )

    trainer = HumorSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=train_data.collate_fn,
        callbacks=callbacks,
    )
    trainer.train()
    if debug_one_step:
        print("[guided-sft] one optimizer-step smoke test passed")
        return

    metrics = trainer.evaluate()
    print(f"[guided-sft] final metrics={metrics}")
    final_dir = Path(config["output"]["final_adapter_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[guided-sft] saved fresh adapter to {final_dir}")


def main() -> None:
    parser = ArgumentParser(description="Train a fresh guided-input humor LoRA.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--debug-one-step", action="store_true")
    args = parser.parse_args()
    train(args.config, debug_one_step=args.debug_one_step)


if __name__ == "__main__":
    main()
