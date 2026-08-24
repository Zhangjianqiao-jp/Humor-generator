#!/usr/bin/env python3
"""Paired bootstrap comparison of two Best-of-N score pools."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preference_diagnostics.best_of_n import candidate_scores, parse_n_values
from src.preference.diagnostics import mean, read_jsonl, sha256, write_csv, write_json


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of an empty list")
    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def paired_compare(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    n_values: tuple[int, ...],
    score_field: str,
    threshold: float,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    left = {str(row.get("image_id")): candidate_scores(row, score_field) for row in left_rows}
    right = {str(row.get("image_id")): candidate_scores(row, score_field) for row in right_rows}
    image_ids = sorted(set(left) & set(right))
    if set(left) != set(right):
        raise ValueError("systems do not contain exactly the same image IDs")
    if not image_ids:
        raise ValueError("no paired images")
    maximum_n = max(n_values)
    if any(len(left[key]) < maximum_n or len(right[key]) < maximum_n for key in image_ids):
        raise ValueError(f"every image needs at least {maximum_n} scores in both systems")
    rng = random.Random(seed)
    output = []
    for n in n_values:
        per_image = []
        for key in image_ids:
            left_prefix, right_prefix = left[key][:n], right[key][:n]
            left_max, right_max = max(left_prefix), max(right_prefix)
            per_image.append(
                (
                    left_max - right_max,
                    mean(left_prefix) - mean(right_prefix),
                    float(left_max >= threshold) - float(right_max >= threshold),
                    left_max,
                    right_max,
                )
            )
        boot = [[], [], []]
        for _ in range(bootstrap_samples):
            sample = [per_image[rng.randrange(len(per_image))] for _ in per_image]
            for metric in range(3):
                boot[metric].append(mean([row[metric] for row in sample]))
        row = {
            "n": n,
            "images": len(image_ids),
            "left_hmax": mean([item[3] for item in per_image]),
            "right_hmax": mean([item[4] for item in per_image]),
            "delta_hmax": mean([item[0] for item in per_image]),
            "delta_hmax_ci_low": percentile(boot[0], 0.025),
            "delta_hmax_ci_high": percentile(boot[0], 0.975),
            "delta_hmean": mean([item[1] for item in per_image]),
            "delta_hmean_ci_low": percentile(boot[1], 0.025),
            "delta_hmean_ci_high": percentile(boot[1], 0.975),
            "delta_pgood": mean([item[2] for item in per_image]),
            "delta_pgood_ci_low": percentile(boot[2], 0.025),
            "delta_pgood_ci_high": percentile(boot[2], 0.975),
            "left_hmax_win_rate": mean([float(item[0] > 0) for item in per_image]),
            "hmax_tie_rate": mean([float(item[0] == 0) for item in per_image]),
        }
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-field", default="humor")
    parser.add_argument("--good-threshold", type=float, default=4)
    parser.add_argument("--n-values", default="1,2,4,8,16,32")
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    rows = paired_compare(
        read_jsonl(args.left), read_jsonl(args.right), parse_n_values(args.n_values),
        args.score_field, args.good_threshold, args.bootstrap_samples, args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_comparison.csv", rows)
    write_json(
        args.output_dir / "paired_comparison_manifest.json",
        {
            "left": str(args.left), "right": str(args.right),
            "left_sha256": sha256(args.left), "right_sha256": sha256(args.right),
            "score_field": args.score_field, "good_threshold": args.good_threshold,
            "bootstrap_samples": args.bootstrap_samples, "seed": args.seed, "results": rows,
        },
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
