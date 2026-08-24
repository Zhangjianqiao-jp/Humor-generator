#!/usr/bin/env python3
"""Measure per-layer/module gradients of the chosen-minus-rejected log-probability margin.

Only parameters that are actually trainable in the loaded adapter are measured.
Missing module families are reported as not_measured, never as zero importance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_loader import load_qwen_vl_with_lora
from src.preference.diagnostics import module_identity, select_image_diverse_rows, sha256, write_csv, write_json
from src.training.dpo_dataset import DPOCollator, PreferenceDataset, model_inputs_from_batch, sequence_logps

EXPECTED = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_SUFFIX = re.compile(r"\.lora_[AB](?:\.[^.]+)?\.weight$")


def model_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def module_key(parameter_name: str) -> str:
    value = LORA_SUFFIX.sub("", parameter_name)
    if value == parameter_name:
        value = parameter_name.rsplit(".", maxsplit=1)[0]
    return value


def heatmap(path: Path, rows: list[dict[str, Any]], metric: str) -> None:
    measured = [row for row in rows if row["layer"] != "" and row["module"] in EXPECTED]
    layers = sorted({int(row["layer"]) for row in measured})
    width = 180 + 90 * len(EXPECTED)
    height = 130 + 32 * max(len(layers), 1)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((20, 18), f"Preference gradient: {metric}", fill="#10253f", font=font)
    values = [float(row[metric]) for row in measured if math.isfinite(float(row[metric]))]
    maximum = max(values, default=1.0) or 1.0
    lookup = {(int(row["layer"]), row["module"]): float(row[metric]) for row in measured}
    for column, module in enumerate(EXPECTED):
        draw.text((120 + column * 90, 55), module.replace("_proj", ""), fill="#10253f", font=font)
    for row_index, layer in enumerate(layers):
        y = 80 + row_index * 32
        draw.text((30, y + 8), str(layer), fill="#10253f", font=font)
        for column, module in enumerate(EXPECTED):
            x = 110 + column * 90
            value = lookup.get((layer, module))
            if value is None:
                color = "#e5e7eb"
                label = "NA"
            else:
                ratio = min(1.0, value / maximum)
                color = (int(240 - 190 * ratio), int(248 - 105 * ratio), int(255 - 45 * ratio))
                label = f"{value:.1e}"
            draw.rectangle((x, y, x + 82, y + 27), fill=color, outline="#cbd5e1")
            draw.text((x + 4, y + 8), label, fill="#10253f", font=font)
    draw.text((20, height - 28), "Gray/NA = not measured by the loaded trainable adapter", fill="#475569", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def cosine_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    measured = [row for row in rows if row["layer"] != "" and row["module"] in EXPECTED]
    layers = sorted({int(row["layer"]) for row in measured})
    width, height = 180 + 90 * len(EXPECTED), 130 + 32 * max(len(layers), 1)
    image = Image.new("RGB", (width, height), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default()
    draw.text((20, 18), "SFT vs preference gradient cosine", fill="#10253f", font=font)
    lookup = {(int(row["layer"]), row["module"]): float(row["cosine"]) for row in measured}
    for column, module in enumerate(EXPECTED):
        draw.text((120 + column * 90, 55), module.replace("_proj", ""), fill="#10253f", font=font)
    for row_index, layer in enumerate(layers):
        y = 80 + row_index * 32
        draw.text((30, y + 8), str(layer), fill="#10253f", font=font)
        for column, module in enumerate(EXPECTED):
            x = 110 + column * 90
            value = lookup.get((layer, module))
            if value is None:
                color, label = "#e5e7eb", "NA"
            elif value >= 0:
                color = (int(245 - 150 * value), int(250 - 65 * value), int(245 - 120 * value))
                label = f"{value:+.2f}"
            else:
                strength = abs(value)
                color = (int(250 - 25 * strength), int(245 - 175 * strength), int(245 - 175 * strength))
                label = f"{value:+.2f}"
            draw.rectangle((x, y, x + 82, y + 27), fill=color, outline="#cbd5e1")
            draw.text((x + 4, y + 8), label, fill="#10253f", font=font)
    draw.text((20, height - 28), "Green = aligned; red = conflicting; gray = not measured", fill="#475569", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/preference_diagnostics/module_gradients"))
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compute-sft-alignment", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg, data_cfg = config["model"], config["data"]
    lora, quant = model_cfg["lora"], model_cfg.get("quantization", {})
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
        is_trainable=True,
        image_min_pixels=data_cfg.get("image_min_pixels"),
        image_max_pixels=data_cfg.get("image_max_pixels"),
        load_in_4bit=bool(quant.get("load_in_4bit", False)),
        bnb_4bit_quant_type=str(quant.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
    )
    model.eval()
    dataset = PreferenceDataset(args.pairs)
    dataset.rows = select_image_diverse_rows(dataset.rows, args.max_pairs)
    collator = DPOCollator(
        processor,
        max_seq_len=int(data_cfg.get("max_seq_len", 768)),
        require_reference=False,
        image_min_pixels=data_cfg.get("image_min_pixels"),
        image_max_pixels=data_cfg.get("image_max_pixels"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    device = model_device(model)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable:
        raise RuntimeError("loaded model has no trainable parameters")

    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"grad_sq_sum": 0.0, "grad_norm_sum": 0.0})
    alignment: dict[str, dict[str, float]] = defaultdict(lambda: {"dot": 0.0, "pref_sq": 0.0, "sft_sq": 0.0})
    parameter_count: dict[str, int] = defaultdict(int)
    weight_sq: dict[str, float] = defaultdict(float)
    for name, parameter in trainable.items():
        key = module_key(name)
        parameter_count[key] += parameter.numel()
        weight_sq[key] += float(parameter.detach().float().pow(2).sum().cpu())

    batches = 0
    pairs_seen = 0
    for batch_index, batch in enumerate(loader, start=1):
        model.zero_grad(set_to_none=True)
        output = model(**model_inputs_from_batch(batch, device))
        logps, token_counts = sequence_logps(output.logits, batch["labels"].to(device))
        n = int(batch["num_pairs"])
        margin = (logps[:n] - logps[n:]).mean()
        (-margin).backward(retain_graph=args.compute_sft_alignment)
        batch_sq: dict[str, float] = defaultdict(float)
        pref_grads: dict[str, torch.Tensor] = {}
        for name, parameter in trainable.items():
            if parameter.grad is None:
                continue
            key = module_key(name)
            grad = parameter.grad.detach().float()
            batch_sq[key] += float(grad.pow(2).sum().cpu())
            if args.compute_sft_alignment:
                pref_grads[name] = grad.clone()
        for key, value in batch_sq.items():
            groups[key]["grad_sq_sum"] += value
            groups[key]["grad_norm_sum"] += math.sqrt(value)
        if args.compute_sft_alignment:
            model.zero_grad(set_to_none=True)
            chosen_nll = -(logps[:n] / token_counts[:n].clamp_min(1)).mean()
            chosen_nll.backward()
            for name, pref_grad in pref_grads.items():
                parameter = trainable[name]
                if parameter.grad is None:
                    continue
                sft_grad = parameter.grad.detach().float()
                key = module_key(name)
                alignment[key]["dot"] += float((pref_grad * sft_grad).sum().cpu())
                alignment[key]["pref_sq"] += float(pref_grad.pow(2).sum().cpu())
                alignment[key]["sft_sq"] += float(sft_grad.pow(2).sum().cpu())
        batches += 1
        pairs_seen += n
        print(f"[module-gradient] batch={batch_index}/{len(loader)} pairs={pairs_seen}")

    rows = []
    for key in sorted(parameter_count):
        layer, module = module_identity(key)
        raw = math.sqrt(groups[key]["grad_sq_sum"] / max(batches, 1))
        count = parameter_count[key]
        weight_norm = math.sqrt(weight_sq[key])
        rows.append(
            {
                "layer": "" if layer is None else layer,
                "module": module,
                "module_path": key,
                "raw_grad_norm": raw,
                "mean_batch_grad_norm": groups[key]["grad_norm_sum"] / max(batches, 1),
                "normalized_grad_norm": raw / math.sqrt(max(count, 1)),
                "relative_grad_norm": raw / (weight_norm + 1e-12),
                "parameter_count": count,
                "weight_norm": weight_norm,
                "batches": batches,
            }
        )

    observed = sorted({row["module"] for row in rows if row["module"] in EXPECTED})
    summary = {
        "pairs": pairs_seen,
        "batches": batches,
        "trainable_parameters": sum(parameter_count.values()),
        "observed_module_families": observed,
        "not_measured_module_families": sorted(set(EXPECTED) - set(observed)),
        "pairs_sha256": sha256(args.pairs),
        "config_sha256": sha256(args.config),
        "adapter": str(args.adapter),
        "sft_alignment_computed": args.compute_sft_alignment,
        "gradient_definition": "gradient of negative mean [log pi(chosen|image,prompt) - log pi(rejected|image,prompt)]",
        "normalization": "raw=RMS batch Frobenius norm; normalized=raw/sqrt(parameter_count); relative=raw/trainable_adapter_weight_norm",
        "interpretation_guard": "Not-measured modules cannot be ranked. Relative norm uses adapter weights, not frozen base Fisher.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "module_gradient_scores.csv", rows)
    aggregate: dict[str, dict[str, float]] = defaultdict(lambda: {"grad_sq": 0.0, "params": 0.0, "weight_sq": 0.0})
    for row in rows:
        group = aggregate[str(row["module"])]
        group["grad_sq"] += float(row["raw_grad_norm"]) ** 2
        group["params"] += int(row["parameter_count"])
        group["weight_sq"] += float(row["weight_norm"]) ** 2
    group_rows = []
    for module, values in sorted(aggregate.items()):
        raw = math.sqrt(values["grad_sq"])
        group_rows.append(
            {
                "module": module,
                "raw_grad_norm": raw,
                "normalized_grad_norm": raw / math.sqrt(max(values["params"], 1)),
                "relative_grad_norm": raw / (math.sqrt(values["weight_sq"]) + 1e-12),
                "parameter_count": int(values["params"]),
            }
        )
    write_csv(args.output_dir / "module_group_summary.csv", group_rows)
    write_json(args.output_dir / "summary.json", summary)
    heatmap(args.output_dir / "layer_module_gradient_heatmap.png", rows, "normalized_grad_norm")
    if args.compute_sft_alignment:
        alignment_rows = []
        for key in sorted(parameter_count):
            layer, module = module_identity(key)
            values = alignment[key]
            denominator = math.sqrt(values["pref_sq"] * values["sft_sq"])
            alignment_rows.append(
                {
                    "layer": "" if layer is None else layer,
                    "module": module,
                    "module_path": key,
                    "cosine": values["dot"] / denominator if denominator else float("nan"),
                    "dot_product": values["dot"],
                    "preference_grad_norm": math.sqrt(values["pref_sq"]),
                    "sft_grad_norm": math.sqrt(values["sft_sq"]),
                    "pairs": pairs_seen,
                }
            )
        write_csv(args.output_dir / "sft_dpo_gradient_cosine.csv", alignment_rows)
        cosine_heatmap(args.output_dir / "sft_dpo_gradient_alignment_heatmap.png", alignment_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
