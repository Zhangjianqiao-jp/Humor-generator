#!/usr/bin/env python
"""Unblind fixed caption scores and report paired image-level bootstrap intervals."""

from __future__ import annotations

import json
import random
import statistics
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def bootstrap_difference(
    image_ids: list[str], metric: Callable[[list[str]], float], repeats: int = 20000
) -> list[float]:
    rng = random.Random(20260812)
    return [metric([rng.choice(image_ids) for _ in image_ids]) for _ in range(repeats)]


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public = {row["blind_id"]: row for row in read_jsonl(args.blind)}
    score_doc = json.loads(args.scores.read_text(encoding="utf-8"))
    scores = {key: int(value) for key, value in score_doc["scores"].items()}
    key_doc = json.loads(args.key.read_text(encoding="utf-8"))
    key = {row["blind_id"]: row["system"] for row in key_doc["key"]}
    if set(public) != set(scores) or set(public) != set(key):
        raise ValueError("blind candidates, scores, and key do not have identical blind IDs")
    threshold = int(score_doc["good_threshold"])

    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_image: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    adjudicated: list[dict[str, Any]] = []
    for blind_id, row in public.items():
        item = {**row, "system": key[blind_id], "score": scores[blind_id]}
        item["good"] = item["score"] >= threshold
        by_system[item["system"]].append(item)
        by_image[item["image_id"]][item["system"]].append(item)
        adjudicated.append(item)

    systems = ["joint", "direct"]
    summary: dict[str, Any] = {}
    for system in systems:
        rows = by_system[system]
        image_groups = [by_image[image_id][system] for image_id in sorted(by_image)]
        summary[system] = {
            "candidates": len(rows),
            "good_candidates": sum(row["good"] for row in rows),
            "good_candidate_rate": sum(row["good"] for row in rows) / len(rows),
            "mean_score": statistics.mean(row["score"] for row in rows),
            "images": len(image_groups),
            "images_with_at_least_one_good": sum(any(row["good"] for row in group) for group in image_groups),
            "image_hit_rate": sum(any(row["good"] for row in group) for group in image_groups) / len(image_groups),
            "strong_candidates": sum(row["score"] == 3 for row in rows),
        }

    image_ids = sorted(by_image)
    def candidate_rate_diff(sample: list[str]) -> float:
        def rate(system: str) -> float:
            rows = [row for image_id in sample for row in by_image[image_id][system]]
            return sum(row["good"] for row in rows) / len(rows)
        return rate("joint") - rate("direct")

    def image_hit_diff(sample: list[str]) -> float:
        def rate(system: str) -> float:
            return sum(any(row["good"] for row in by_image[image_id][system]) for image_id in sample) / len(sample)
        return rate("joint") - rate("direct")

    candidate_boot = bootstrap_difference(image_ids, candidate_rate_diff)
    image_boot = bootstrap_difference(image_ids, image_hit_diff)
    comparison = {
        "candidate_good_rate_difference_joint_minus_direct": summary["joint"]["good_candidate_rate"] - summary["direct"]["good_candidate_rate"],
        "candidate_difference_bootstrap_95_ci": [percentile(candidate_boot, 0.025), percentile(candidate_boot, 0.975)],
        "image_hit_rate_difference_joint_minus_direct": summary["joint"]["image_hit_rate"] - summary["direct"]["image_hit_rate"],
        "image_difference_bootstrap_95_ci": [percentile(image_boot, 0.025), percentile(image_boot, 0.975)],
        "per_image_wins_by_best_score": {"joint": 0, "direct": 0, "tie": 0},
    }
    for image_id in image_ids:
        joint_best = max(row["score"] for row in by_image[image_id]["joint"])
        direct_best = max(row["score"] for row in by_image[image_id]["direct"])
        winner = "joint" if joint_best > direct_best else "direct" if direct_best > joint_best else "tie"
        comparison["per_image_wins_by_best_score"][winner] += 1

    report = {
        "evaluator": "GPT/Codex blind single-rater evaluation",
        "images": len(image_ids),
        "candidates_per_system_per_image": 3,
        "rubric": score_doc["rubric"],
        "good_threshold": threshold,
        "summary": summary,
        "comparison": comparison,
        "adjudicated": adjudicated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "comparison": comparison}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
