#!/usr/bin/env python
"""Freeze SFT-policy log probabilities for memory-efficient multimodal DPO."""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_loader import load_qwen_vl_with_lora
from src.training.dpo_dataset import DPOCollator, PreferenceDataset, model_inputs_from_batch, sequence_logps
from src.utils.io import read_jsonl


def model_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def main() -> None:
    parser = ArgumentParser(description="Precompute frozen reference log-probabilities for DPO pairs.")
    parser.add_argument("--config", type=Path, default=Path("configs/dpo_reference_newyorker_compact_3b.yaml"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
        help="Splits to process. This permits keeping a held-out test split sealed.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from a validated .partial.jsonl prefix.")
    parser.add_argument(
        "--stress-longest-text",
        action="store_true",
        help="Process one full batch of the longest chosen+rejected rows (GPU pressure smoke).",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config output_dir.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    data_cfg = config["data"]
    output_dir = args.output_dir or Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_qwen_vl_with_lora(
        model_name=model_cfg["model_name"],
        lora_rank=int(model_cfg["lora"]["rank"]),
        lora_alpha=int(model_cfg["lora"]["alpha"]),
        lora_dropout=float(model_cfg["lora"]["dropout"]),
        target_modules=list(model_cfg["lora"]["target_modules"]),
        bias=str(model_cfg["lora"].get("bias", "none")),
        device_map=str(model_cfg.get("device_map", "auto")),
        torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        adapter_path=Path(model_cfg["reference_adapter_dir"]),
        is_trainable=False,
        image_min_pixels=data_cfg.get("image_min_pixels"),
        image_max_pixels=data_cfg.get("image_max_pixels"),
        load_in_4bit=bool(model_cfg.get("quantization", {}).get("load_in_4bit", False)),
        bnb_4bit_quant_type=str(model_cfg.get("quantization", {}).get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(
            model_cfg.get("quantization", {}).get("bnb_4bit_use_double_quant", True)
        ),
    )
    model.eval()
    device = model_device(model)
    collator = DPOCollator(
        processor,
        max_seq_len=int(data_cfg.get("max_seq_len", 768)),
        require_reference=False,
        image_min_pixels=data_cfg.get("image_min_pixels"),
        image_max_pixels=data_cfg.get("image_max_pixels"),
    )
    split_counts: dict[str, int] = {}
    splits = tuple(dict.fromkeys(args.splits))
    for split in splits:
        dataset = PreferenceDataset(
            Path(data_cfg[f"{split}_path"]),
            max_samples=args.max_samples,
            skip_missing_images=bool(data_cfg.get("skip_missing_images", False)),
        )
        if args.stress_longest_text:
            stress_rows = sorted(
                dataset.rows,
                key=lambda row: len(str(row["chosen"])) + len(str(row["rejected"])),
                reverse=True,
            )[: int(data_cfg.get("batch_size", 1))]
            dataset.rows = stress_rows
            print(
                f"[reference] {split}: stress rows={len(stress_rows)} "
                f"max_text_chars={max(len(str(row['chosen'])) + len(str(row['rejected'])) for row in stress_rows)}"
            )
        all_rows = list(dataset.rows)
        output_path = output_dir / f"dpo_{split}.jsonl"
        partial_path = output_dir / f"dpo_{split}.partial.jsonl"
        completed_rows: list[dict[str, Any]] = []
        if args.resume and partial_path.exists():
            completed_rows = read_jsonl(partial_path)
            if len(completed_rows) > len(all_rows):
                raise ValueError(f"Partial {split} output is longer than its input.")
            for index, completed in enumerate(completed_rows):
                if str(completed.get("pair_id")) != str(all_rows[index].get("pair_id")):
                    raise ValueError(f"Partial {split} output is not an exact input prefix at row {index}.")
                if "reference_logps" not in completed:
                    raise ValueError(f"Partial {split} output row {index} lacks reference_logps.")
            print(f"[reference] {split}: resuming after {len(completed_rows)}/{len(all_rows)} pairs")
        else:
            partial_path.unlink(missing_ok=True)
        dataset.rows = all_rows[len(completed_rows) :]
        loader = DataLoader(
            dataset,
            batch_size=int(data_cfg.get("batch_size", 1)),
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        cursor = 0
        mode = "a" if completed_rows else "w"
        with partial_path.open(mode, encoding="utf-8") as output_handle, torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                outputs = model(**model_inputs_from_batch(batch, device))
                logps, token_counts = sequence_logps(outputs.logits, batch["labels"].to(device))
                n = int(batch["num_pairs"])
                for offset, raw in enumerate(dataset.rows[cursor : cursor + n]):
                    row = dict(raw)
                    row["reference_logps"] = {
                        "chosen": float(logps[offset].cpu()),
                        "rejected": float(logps[n + offset].cpu()),
                        "chosen_tokens": int(token_counts[offset].cpu()),
                        "rejected_tokens": int(token_counts[n + offset].cpu()),
                        "reference_adapter_dir": str(model_cfg["reference_adapter_dir"]),
                    }
                    output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                cursor += n
                if batch_index % 50 == 0 or cursor == len(dataset):
                    output_handle.flush()
                processed = len(completed_rows) + cursor
                if batch_index % 100 == 0 or cursor == len(dataset):
                    print(f"[reference] {split}: {processed}/{len(all_rows)} pairs")
        if cursor != len(dataset):
            raise RuntimeError(f"Reference cursor mismatch for {split}: {cursor} != {len(dataset)}")
        partial_path.replace(output_path)
        split_counts[split] = len(all_rows)
    manifest = {
        "input_config": str(args.config),
        "reference_adapter_dir": str(model_cfg["reference_adapter_dir"]),
        "quantization": model_cfg.get("quantization", {}),
        "processed_splits": list(splits),
        "input_paths": {split: data_cfg[f"{split}_path"] for split in splits},
        "pairs_by_split": split_counts,
        "warning": "Reference log probabilities are valid only for this exact base model, SFT adapter, chat template, and max_seq_len.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    processor.save_pretrained(output_dir / "processor")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
