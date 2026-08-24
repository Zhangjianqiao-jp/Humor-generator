#!/usr/bin/env python3
"""Measure whether caption preference margins depend on the correct image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_loader import load_qwen_vl_with_lora
from src.preference.diagnostics import derangement, histogram_png, mean, median, select_image_diverse_rows, sha256, write_csv, write_json
from src.training.dpo_dataset import DPOCollator, PreferenceDataset, model_inputs_from_batch, sequence_logps


def model_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def pair_margin(model: Any, collator: DPOCollator, row: dict[str, Any], device: torch.device) -> float:
    batch = collator([row])
    with torch.inference_mode():
        output = model(**model_inputs_from_batch(batch, device))
    logps, _ = sequence_logps(output.logits, batch["labels"].to(device))
    return float((logps[0] - logps[1]).cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/preference_diagnostics/image_shuffle"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg, data_cfg = config["model"], config["data"]
    dataset = PreferenceDataset(args.pairs)
    dataset.rows = select_image_diverse_rows(dataset.rows, args.max_samples)
    image_ids = sorted({str(row["image_id"]) for row in dataset.rows})
    if len(image_ids) < 2:
        raise ValueError("image shuffle requires at least two unique images")
    image_by_id: dict[str, str] = {}
    for row in dataset.rows:
        image_by_id[str(row["image_id"])] = str(row["image"])
    donor_ids = derangement(image_ids, args.seed)

    quant = model_cfg.get("quantization", {})
    lora = model_cfg["lora"]
    model, processor = load_qwen_vl_with_lora(
        model_name=model_cfg["model_name"],
        lora_rank=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias=str(lora.get("bias", "none")),
        device_map=str(model_cfg.get("device_map", "auto")),
        torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        adapter_path=args.adapter,
        is_trainable=False,
        image_min_pixels=data_cfg.get("image_min_pixels"),
        image_max_pixels=data_cfg.get("image_max_pixels"),
        load_in_4bit=bool(quant.get("load_in_4bit", False)),
        bnb_4bit_quant_type=str(quant.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
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

    output_rows = []
    for index, row in enumerate(dataset.rows, start=1):
        image_id = str(row["image_id"])
        donor_id = donor_ids[image_id]
        wrong = dict(row)
        wrong["image"] = image_by_id[donor_id]
        correct_margin = pair_margin(model, collator, row, device)
        shuffled_margin = pair_margin(model, collator, wrong, device)
        output_rows.append(
            {
                "image_id": image_id,
                "image": row["image"],
                "donor_image_id": donor_id,
                "donor_image": wrong["image"],
                "correct_margin": correct_margin,
                "shuffled_margin": shuffled_margin,
                "delta_margin": correct_margin - shuffled_margin,
                "pair_type": row.get("pair_type") or row.get("meta", {}).get("pair_type", "unknown"),
            }
        )
        if index % 10 == 0 or index == len(dataset):
            print(f"[image-shuffle] {index}/{len(dataset)}")

    deltas = [float(row["delta_margin"]) for row in output_rows]
    summary = {
        "pairs": len(output_rows),
        "unique_images": len(image_ids),
        "mean_delta_margin": mean(deltas),
        "median_delta_margin": median(deltas),
        "fraction_delta_gt_zero": mean([float(value > 0) for value in deltas]),
        "mean_correct_margin": mean([float(row["correct_margin"]) for row in output_rows]),
        "mean_shuffled_margin": mean([float(row["shuffled_margin"]) for row in output_rows]),
        "seed": args.seed,
        "pairs_sha256": sha256(args.pairs),
        "config_sha256": sha256(args.config),
        "adapter": str(args.adapter),
        "interpretation_guard": "Near-zero delta suggests weak image dependence; positive delta alone does not prove humor quality.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "image_shuffle_margin.csv", output_rows)
    write_json(args.output_dir / "summary.json", summary)
    histogram_png(
        args.output_dir / "image_shuffle_distribution.png",
        deltas,
        "Image-shuffle preference-margin diagnostic",
        "M(correct image) - M(shuffled image)",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
