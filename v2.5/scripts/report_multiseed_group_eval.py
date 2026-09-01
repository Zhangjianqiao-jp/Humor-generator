#!/usr/bin/env python3
"""Unblind a multi-seed group evaluation with image-clustered statistics."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ABSOLUTE_LABELS = {"bad": 0, "weak": 1, "good": 2}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson(wins: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = wins / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [centre - margin, centre + margin]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def clustered_bootstrap_ci(
    image_scores: list[float], samples: int, seed: int
) -> list[float]:
    rng = random.Random(seed)
    n = len(image_scores)
    draws = [
        sum(image_scores[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def parse_policy_and_seed(system: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_s(\d+)", system)
    if not match:
        raise ValueError(f"system name must end in _s<seed>: {system!r}")
    return match.group(1), int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--preferred-policy", default="dpo")
    parser.add_argument("--baseline-policy", default="sft")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public = {row["pair_id"]: row for row in read_jsonl(args.public)}
    key_doc = json.load(args.key.open(encoding="utf-8"))
    keys = {row["pair_id"]: row for row in key_doc["key"]}
    decision_doc = json.load(args.decisions.open(encoding="utf-8"))
    decisions = decision_doc["decisions"]
    if set(public) != set(keys) or set(public) != set(decisions):
        raise ValueError("public, key, and decision pair IDs differ")

    rows: list[dict[str, Any]] = []
    position = Counter()
    for pair_id, decision in decisions.items():
        if decision.get("overall") not in {"A", "B", "Tie"}:
            raise ValueError(f"{pair_id}: overall must be A, B, or Tie")
        if decision.get("best_pick") not in {"A", "B", "Tie"}:
            raise ValueError(f"{pair_id}: best_pick must be A, B, or Tie")
        for side in ("A", "B"):
            if decision.get(f"absolute_{side}") not in ABSOLUTE_LABELS:
                raise ValueError(f"{pair_id}: absolute_{side} must be good, weak, or bad")
            if decision.get(f"best_{side}_index") not in {1, 2, 3}:
                raise ValueError(f"{pair_id}: best_{side}_index must be 1, 2, or 3")
        key = keys[pair_id]
        left_policy, left_seed = parse_policy_and_seed(key["group_A_system"])
        right_policy, right_seed = parse_policy_and_seed(key["group_B_system"])
        if left_seed != right_seed:
            raise ValueError(f"{pair_id}: compared systems use different generation seeds")
        if {left_policy, right_policy} != {args.preferred_policy, args.baseline_policy}:
            raise ValueError(f"{pair_id}: unexpected policies {left_policy}, {right_policy}")
        winner_side = decision["overall"]
        best_side = decision["best_pick"]
        position[f"overall_{winner_side}"] += 1
        position[f"best_pick_{best_side}"] += 1
        absolute = {
            left_policy: decision["absolute_A"],
            right_policy: decision["absolute_B"],
        }
        rows.append(
            {
                **public[pair_id],
                **key,
                **decision,
                "generation_seed": left_seed,
                "overall_winner": (
                    "tie" if winner_side == "Tie" else left_policy if winner_side == "A" else right_policy
                ),
                "best_pick_winner": (
                    "tie" if best_side == "Tie" else left_policy if best_side == "A" else right_policy
                ),
                "absolute_quality": absolute,
            }
        )

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[row["generation_seed"]].append(row)
        by_image[row["image_id"]].append(row)

    seeds = sorted(by_seed)
    expected_images = set(by_image)
    if len(seeds) < 3:
        raise ValueError("multi-seed report requires at least three generation seeds")
    if any({row["image_id"] for row in by_seed[seed]} != expected_images for seed in seeds):
        raise ValueError("every seed must contain the same image IDs")

    per_seed: dict[str, Any] = {}
    seed_win_rates: list[float] = []
    for seed in seeds:
        seed_rows = by_seed[seed]
        wins = sum(row["overall_winner"] == args.preferred_policy for row in seed_rows)
        ties = sum(row["overall_winner"] == "tie" for row in seed_rows)
        losses = len(seed_rows) - wins - ties
        win_rate = (wins + 0.5 * ties) / len(seed_rows)
        seed_win_rates.append(win_rate)
        absolute = {}
        for policy in (args.preferred_policy, args.baseline_policy):
            counts = Counter(row["absolute_quality"][policy] for row in seed_rows)
            absolute[policy] = {
                "counts": {label: counts[label] for label in ABSOLUTE_LABELS},
                "rates": {label: counts[label] / len(seed_rows) for label in ABSOLUTE_LABELS},
            }
        per_seed[str(seed)] = {
            "images": len(seed_rows),
            "preferred_wins": wins,
            "ties": ties,
            "preferred_losses": losses,
            "preferred_win_rate": win_rate,
            "absolute_quality": absolute,
        }

    image_scores: list[float] = []
    majority_wins = 0
    majority_ties = 0
    image_summary: dict[str, Any] = {}
    for image_id, image_rows in sorted(by_image.items()):
        wins = sum(row["overall_winner"] == args.preferred_policy for row in image_rows)
        ties = sum(row["overall_winner"] == "tie" for row in image_rows)
        score = (wins + 0.5 * ties) / len(image_rows)
        image_scores.append(score)
        majority = score > 0.5
        majority_wins += int(majority)
        majority_ties += int(score == 0.5)
        qualities = {}
        for policy in (args.preferred_policy, args.baseline_policy):
            labels = [row["absolute_quality"][policy] for row in image_rows]
            qualities[policy] = {
                "labels": labels,
                "majority_good": labels.count("good") > len(labels) / 2,
                "median_ordinal": sorted(ABSOLUTE_LABELS[label] for label in labels)[len(labels) // 2],
            }
        image_summary[image_id] = {
            "preferred_seed_wins": wins,
            "seed_ties": ties,
            "preferred_win_fraction": score,
            "preferred_majority_win": majority,
            "absolute_quality": qualities,
        }

    absolute_image_level = {}
    for policy in (args.preferred_policy, args.baseline_policy):
        majority_good = sum(
            row["absolute_quality"][policy]["majority_good"] for row in image_summary.values()
        )
        median_counts = Counter(
            row["absolute_quality"][policy]["median_ordinal"] for row in image_summary.values()
        )
        absolute_image_level[policy] = {
            "majority_good_images": majority_good,
            "majority_good_rate": majority_good / len(image_summary),
            "majority_good_rate_95ci_wilson": wilson(majority_good, len(image_summary)),
            "median_label_counts": {
                label: median_counts[value] for label, value in ABSOLUTE_LABELS.items()
            },
        }

    report = {
        "protocol": decision_doc.get("protocol"),
        "judge": decision_doc.get("judge"),
        "preferred_policy": args.preferred_policy,
        "baseline_policy": args.baseline_policy,
        "images": len(image_summary),
        "generation_seeds": seeds,
        "trials": len(rows),
        "position_choices": dict(position),
        "per_seed": per_seed,
        "seed_summary": {
            "mean_preferred_win_rate": statistics.mean(seed_win_rates),
            "sample_std_preferred_win_rate": statistics.stdev(seed_win_rates),
            "sample_variance_preferred_win_rate": statistics.variance(seed_win_rates),
        },
        "image_clustered": {
            "mean_preferred_win_fraction": statistics.mean(image_scores),
            "bootstrap_95ci": clustered_bootstrap_ci(
                image_scores, args.bootstrap_samples, args.bootstrap_seed
            ),
            "majority_win_images": majority_wins,
            "majority_tied_images": majority_ties,
            "majority_loss_images": len(image_summary) - majority_wins - majority_ties,
            "majority_win_rate": majority_wins / len(image_summary),
            "majority_win_rate_95ci_wilson": wilson(majority_wins, len(image_summary)),
        },
        "absolute_quality_image_level": absolute_image_level,
        "per_image": image_summary,
        "unblinded_trials": sorted(rows, key=lambda row: (row["image_id"], row["generation_seed"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    compact = {key: value for key, value in report.items() if key not in {"per_image", "unblinded_trials"}}
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
