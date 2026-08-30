#!/usr/bin/env python3
"""Match uniform LoRA ranks across module placements by trainable parameters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preference.module_analysis import matched_uniform_rank


def text_config(config: Any) -> Any:
    for name in ("text_config", "llm_config"):
        value = getattr(config, name, None)
        if value is not None:
            return value
    return config


def dimensions(config: Any) -> tuple[int, int, int, int]:
    config = text_config(config)
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    return hidden, intermediate, layers, kv_heads * head_dim


def placement_shapes(hidden: int, intermediate: int, layers: int, kv_hidden: int) -> dict[str, list[tuple[int, int]]]:
    per_layer = {
        "q_proj": (hidden, hidden),
        "k_proj": (hidden, kv_hidden),
        "v_proj": (hidden, kv_hidden),
        "o_proj": (hidden, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }
    groups = {
        "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "mlp": ("gate_proj", "up_proj", "down_proj"),
        "all_linear": tuple(per_layer),
    }
    return {name: [per_layer[module] for _ in range(layers) for module in modules] for name, modules in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--target-budget", type=int, default=7_372_800)
    parser.add_argument("--alpha-ratio", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("results/preference_analysis/lora_budget_match.csv"))
    args = parser.parse_args()
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hidden, intermediate, layers, kv_hidden = dimensions(cfg)
    placements = placement_shapes(hidden, intermediate, layers, kv_hidden)
    rows = []
    for name, shapes in placements.items():
        rank, parameters = matched_uniform_rank(shapes, args.target_budget)
        rows.append(
            {
                "placement": name,
                "recommended_rank": rank,
                "lora_alpha": max(1, round(rank * args.alpha_ratio)),
                "trainable_parameter_count": parameters,
                "budget_difference": parameters - args.target_budget,
                "budget_relative_error": (parameters - args.target_budget) / args.target_budget,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    import csv

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "model": args.model,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "kv_projection_size": kv_hidden,
        "target_budget": args.target_budget,
        "rows": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(yaml.safe_dump(manifest, sort_keys=False))


if __name__ == "__main__":
    main()
