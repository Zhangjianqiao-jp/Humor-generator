#!/usr/bin/env python3
"""Build the five low-cost, budget-matched module-placement pilot specs.

Gradient-selected uses the already measured attention-LoRA tangent gradients;
it does not pretend that unmeasured MLP matrices had zero importance. Random
selection is stratified to match the selected module-family counts, making its
LoRA denominator and recommended rank directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preference.module_analysis import matched_uniform_rank


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def placement(name: str, modules: list[str], rank: int, parameters: int, evidence: str) -> dict[str, Any]:
    return {
        "placement": name,
        "adapter_strategy": "separate_preference",
        "lora": {
            "target_modules": modules,
            "rank": rank,
            "alpha": rank * 2,
            "dropout": 0.05,
            "trainable_parameter_count": parameters,
        },
        "selection_evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-csv", type=Path, required=True)
    parser.add_argument("--target-budget", type=int, default=7_372_800)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/preference/go_no_go"))
    args = parser.parse_args()
    rows = read_rows(args.gradient_csv)
    if args.top_k < 1 or args.top_k >= len(rows):
        raise ValueError("top-k must be positive and smaller than the measured pool")
    ranked = sorted(rows, key=lambda row: (-float(row["normalized_grad_norm"]), row["module_path"]))
    selected = ranked[: args.top_k]
    family_counts = Counter(row["module"] for row in selected)
    selected_paths = {row["module_path"] for row in selected}
    rng = random.Random(args.seed)
    random_rows = []
    for family, count in sorted(family_counts.items()):
        candidates = [row for row in rows if row["module"] == family and row["module_path"] not in selected_paths]
        if len(candidates) < count:
            raise ValueError(
                f"Cannot build a disjoint random control for {family}: need {count}, have {len(candidates)}. "
                "Reduce --top-k."
            )
        random_rows.extend(rng.sample(candidates, count))
    # The diagnostic adapter used rank 16, so count/rank recovers d_in+d_out.
    selected_denominators = [int(row["parameter_count"]) // 16 for row in selected]
    random_denominators = [int(row["parameter_count"]) // 16 for row in random_rows]
    selected_shapes = [(value - 1, 1) for value in selected_denominators]
    random_shapes = [(value - 1, 1) for value in random_denominators]
    selected_rank, selected_params = matched_uniform_rank(selected_shapes, args.target_budget)
    random_rank, random_params = matched_uniform_rank(random_shapes, args.target_budget)
    if selected_params != random_params or selected_rank != random_rank:
        raise AssertionError("stratified random placement must exactly match selected budget")

    # Qwen2.5-VL-3B dimensions, already resolved by match_lora_budget.py.
    specs = [
        placement("attention", ["q_proj", "k_proj", "v_proj", "o_proj"], 16, 7_372_800, "complete attention placement"),
        placement("mlp", ["gate_proj", "up_proj", "down_proj"], 5, 7_050_240, "complete MLP placement"),
        placement(
            "all_linear",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            4,
            7_483_392,
            "complete language-backbone linear placement",
        ),
        placement(
            "gradient_selected",
            [row["module_path"] for row in selected],
            selected_rank,
            selected_params,
            f"top {args.top_k} existing attention-LoRA tangent gradients by normalized_grad_norm",
        ),
        placement(
            "random_selected",
            [row["module_path"] for row in random_rows],
            random_rank,
            random_params,
            f"seeded random, matched counts by module family to gradient-selected (seed={args.seed})",
        ),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        (args.output_dir / f"{spec['placement']}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    manifest = {
        "source": str(args.gradient_csv),
        "source_scope": "current SFT attention LoRA only; MLP was not measured",
        "selection_metric": "normalized_grad_norm",
        "target_budget": args.target_budget,
        "top_k": args.top_k,
        "seed": args.seed,
        "family_counts": dict(family_counts),
        "placements": specs,
        "go_condition": "gradient-selected must clearly and stably beat random-selected and show parameter-efficiency advantage",
        "no_go_action": "stop Fisher/SVD/dynamic-rank/large layer-wise search and retain the best simple placement",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
