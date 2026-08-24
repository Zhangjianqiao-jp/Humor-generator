#!/usr/bin/env python3
"""Fit linear probes to response-token hidden states at several VLM layers."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_loader import load_qwen_vl_with_lora
from src.preference.diagnostics import line_plot_png, read_jsonl, sha256, write_csv, write_json
from src.training.dpo_dataset import DPOCollator, model_inputs_from_batch

LABELS = {"literal": 0, "weak": 1, "funny": 2}


def model_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def validate_examples(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for index, row in enumerate(rows):
        missing = [key for key in ("image", "image_id", "prompt", "caption", "label") if not str(row.get(key) or "").strip()]
        if missing:
            raise ValueError(f"row {index} missing {missing}")
        if row["label"] not in LABELS:
            raise ValueError(f"row {index}: label must be one of {sorted(LABELS)}")
    observed = {row["label"] for row in rows}
    if observed != set(LABELS):
        raise ValueError(f"probe requires funny/literal/weak; observed={sorted(observed)}")
    return rows


def probe_layers(hidden_count: int) -> dict[str, int]:
    final = hidden_count - 1
    return {
        "early": max(1, round(final * 0.25)),
        "middle": max(1, round(final * 0.50)),
        "late": max(1, round(final * 0.75)),
        "final": final,
    }


def split_images(rows: list[dict[str, Any]], ratio: float, seed: int) -> tuple[set[str], set[str]]:
    image_ids = sorted({str(row["image_id"]) for row in rows})
    if len(image_ids) < 6:
        raise ValueError("probe requires at least six unique images")
    for attempt in range(100):
        shuffled = image_ids[:]
        random.Random(seed + attempt).shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * ratio))
        test_ids, train_ids = set(shuffled[:test_count]), set(shuffled[test_count:])
        train_labels = {row["label"] for row in rows if str(row["image_id"]) in train_ids}
        test_labels = {row["label"] for row in rows if str(row["image_id"]) in test_ids}
        if train_labels == test_labels == set(LABELS):
            return train_ids, test_ids
    raise ValueError("could not create image-disjoint split containing all labels; add more labeled images")


def macro_f1(labels: torch.Tensor, predicted: torch.Tensor, num_classes: int) -> float:
    values = []
    for label in range(num_classes):
        tp = int(((labels == label) & (predicted == label)).sum())
        fp = int(((labels != label) & (predicted == label)).sum())
        fn = int(((labels == label) & (predicted != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values)


def binary_auc(target: torch.Tensor, score: torch.Tensor) -> float:
    positive = score[target.bool()]
    negative = score[~target.bool()]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).float()
    ties = (positive[:, None] == negative[None, :]).float()
    return float((comparisons + 0.5 * ties).mean())


def train_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
    epochs: int,
    num_classes: int = len(LABELS),
) -> dict[str, float]:
    torch.manual_seed(seed)
    train_x, test_x = features[train_mask].float(), features[test_mask].float()
    train_y, test_y = labels[train_mask], labels[test_mask]
    location = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-5)
    train_x, test_x = (train_x - location) / scale, (test_x - location) / scale
    classifier = nn.Linear(train_x.shape[-1], num_classes)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(epochs):
        logits = classifier(train_x)
        loss = nn.functional.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = classifier(test_x)
        probabilities = logits.softmax(-1)
        predicted = logits.argmax(-1)
    aucs = [binary_auc(test_y == label, probabilities[:, label]) for label in range(num_classes)]
    return {
        "accuracy": float((predicted == test_y).float().mean()),
        "macro_f1": macro_f1(test_y, predicted, num_classes),
        "macro_auroc": sum(aucs) / len(aucs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/preference_diagnostics/humor_probe"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    rows = validate_examples(args.examples)
    if args.max_images is not None:
        keep = sorted({str(row["image_id"]) for row in rows})[: args.max_images]
        rows = [row for row in rows if str(row["image_id"]) in set(keep)]
    train_ids, test_ids = split_images(rows, args.test_ratio, args.seed)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg, data_cfg = config["model"], config["data"]
    lora, quant = model_cfg["lora"], model_cfg.get("quantization", {})
    model, processor = load_qwen_vl_with_lora(
        model_name=model_cfg["model_name"], lora_rank=int(lora["rank"]), lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]), target_modules=list(lora["target_modules"]),
        bias=str(lora.get("bias", "none")), device_map=str(model_cfg.get("device_map", "auto")),
        torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")), trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        adapter_path=args.adapter, is_trainable=False, image_min_pixels=data_cfg.get("image_min_pixels"),
        image_max_pixels=data_cfg.get("image_max_pixels"), load_in_4bit=bool(quant.get("load_in_4bit", False)),
        bnb_4bit_quant_type=str(quant.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
    )
    model.eval()
    device = model_device(model)
    collator = DPOCollator(processor, int(data_cfg.get("max_seq_len", 768)), require_reference=False,
                           image_min_pixels=data_cfg.get("image_min_pixels"), image_max_pixels=data_cfg.get("image_max_pixels"))
    fake_pairs = [{"image": row["image"], "image_id": row["image_id"], "prompt": row["prompt"],
                   "chosen": row["caption"], "rejected": row["caption"]} for row in rows]
    loader = DataLoader(fake_pairs, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    features: dict[str, list[torch.Tensor]] = {}
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            n = int(batch["num_pairs"])
            output = model(**model_inputs_from_batch(batch, device), output_hidden_states=True, use_cache=False)
            layers = probe_layers(len(output.hidden_states))
            response_mask = batch["labels"][:n].to(device).ne(-100)
            for name, index in layers.items():
                hidden = output.hidden_states[index][:n].float()
                pooled = (hidden * response_mask.unsqueeze(-1)).sum(1) / response_mask.sum(1, keepdim=True).clamp_min(1)
                features.setdefault(name, []).append(pooled.cpu())
            cursor += n
            print(f"[humor-probe] extracted {cursor}/{len(rows)}")

    labels = torch.tensor([LABELS[row["label"]] for row in rows], dtype=torch.long)
    train_mask = torch.tensor([str(row["image_id"]) in train_ids for row in rows])
    test_mask = ~train_mask
    result_rows = []
    for layer_name, chunks in features.items():
        layer_features = torch.cat(chunks)
        metrics = train_probe(layer_features, labels, train_mask, test_mask, args.seed, args.epochs)
        funny_weak = labels.ne(LABELS["literal"])
        binary_labels = labels.eq(LABELS["funny"]).long()
        binary = train_probe(
            layer_features,
            binary_labels,
            train_mask & funny_weak,
            test_mask & funny_weak,
            args.seed,
            args.epochs,
            num_classes=2,
        )
        result_rows.append(
            {
                "layer": layer_name,
                **metrics,
                "funny_vs_weak_accuracy": binary["accuracy"],
                "funny_vs_weak_f1": binary["macro_f1"],
                "funny_vs_weak_auroc": binary["macro_auroc"],
                "train_examples": int(train_mask.sum()),
                "test_examples": int(test_mask.sum()),
            }
        )
    order = {name: index for index, name in enumerate(("early", "middle", "late", "final"))}
    result_rows.sort(key=lambda row: order[row["layer"]])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "humor_probe_by_layer.csv", result_rows)
    line_plot_png(args.output_dir / "humor_probe_by_layer.png", list(range(len(result_rows))),
                  {"accuracy": [row["accuracy"] for row in result_rows], "F1": [row["macro_f1"] for row in result_rows],
                   "AUROC": [row["macro_auroc"] for row in result_rows]},
                  "Linear humor probe by layer (early, middle, late, final)", "layer index", "score")
    summary = {"examples": len(rows), "train_images": len(train_ids), "test_images": len(test_ids),
               "labels": LABELS, "examples_sha256": sha256(args.examples), "config_sha256": sha256(args.config),
               "adapter": str(args.adapter), "results": result_rows,
               "interpretation_guard": "Probe separability is correlational and does not establish a causal generation mechanism."}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
