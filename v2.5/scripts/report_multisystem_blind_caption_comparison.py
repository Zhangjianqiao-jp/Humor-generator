#!/usr/bin/env python3
"""Unblind a completed multi-system caption evaluation and summarize it."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    candidates = {
        row["blind_id"]: row
        for row in (json.loads(line) for line in args.candidates.open())
    }
    score_doc = json.load(args.scores.open())
    scores = score_doc["scores"]
    key_doc = json.load(args.key.open())
    keys = {row["blind_id"]: row for row in key_doc["key"]}
    if set(candidates) != set(scores) or set(candidates) != set(keys):
        raise ValueError("candidate, score, and key ID sets differ")

    rows = []
    by_system: dict[str, list[dict]] = defaultdict(list)
    by_image_system: dict[tuple[str, str], list[int]] = defaultdict(list)
    good_threshold = score_doc["good_threshold"]
    strong_threshold = score_doc["strong_threshold"]
    for blind_id, candidate in candidates.items():
        system = keys[blind_id]["system"]
        score = int(scores[blind_id])
        row = {**candidate, "system": system, "score": score}
        rows.append(row)
        by_system[system].append(row)
        by_image_system[(candidate["image_id"], system)].append(score)

    systems = key_doc["systems"]
    image_ids = sorted({row["image_id"] for row in rows})
    summary = {}
    for system in systems:
        vals = np.array([row["score"] for row in by_system[system]], dtype=float)
        per_image = [by_image_system[(image_id, system)] for image_id in image_ids]
        summary[system] = {
            "candidates": int(vals.size),
            "good_count": int((vals >= good_threshold).sum()),
            "good_rate": float((vals >= good_threshold).mean()),
            "strong_count": int((vals >= strong_threshold).sum()),
            "strong_rate": float((vals >= strong_threshold).mean()),
            "mean_score": float(vals.mean()),
            "images_with_any_good": int(sum(max(v) >= good_threshold for v in per_image)),
            "image_hit_rate": float(np.mean([max(v) >= good_threshold for v in per_image])),
            "best_of_3_mean": float(np.mean([max(v) for v in per_image])),
        }

    # Paired bootstrap across images; each image is the sampling unit.
    rng = np.random.default_rng(args.seed)
    pairwise = {}
    image_good = {
        system: np.array([
            np.mean(np.array(by_image_system[(image_id, system)]) >= good_threshold)
            for image_id in image_ids
        ])
        for system in systems
    }
    image_mean = {
        system: np.array([
            np.mean(by_image_system[(image_id, system)]) for image_id in image_ids
        ])
        for system in systems
    }
    for a in systems:
        for b in systems:
            if systems.index(a) >= systems.index(b):
                continue
            idx = rng.integers(0, len(image_ids), size=(args.bootstrap, len(image_ids)))
            good_boot = (image_good[a][idx] - image_good[b][idx]).mean(axis=1)
            mean_boot = (image_mean[a][idx] - image_mean[b][idx]).mean(axis=1)
            pairwise[f"{a}_minus_{b}"] = {
                "good_rate_difference": float(image_good[a].mean() - image_good[b].mean()),
                "good_rate_difference_95ci": [float(x) for x in np.quantile(good_boot, [0.025, 0.975])],
                "mean_score_difference": float(image_mean[a].mean() - image_mean[b].mean()),
                "mean_score_difference_95ci": [float(x) for x in np.quantile(mean_boot, [0.025, 0.975])],
            }

    # Winner uses the best of the three samples on each image; ties are retained.
    per_image_winners = []
    win_counts = {system: 0 for system in systems}
    unique_win_counts = {system: 0 for system in systems}
    for image_id in image_ids:
        best = {system: max(by_image_system[(image_id, system)]) for system in systems}
        top = max(best.values())
        winners = [system for system, value in best.items() if value == top]
        for system in winners:
            win_counts[system] += 1
        if len(winners) == 1:
            unique_win_counts[winners[0]] += 1
        per_image_winners.append({"image_id": image_id, "best_score": best, "winners": winners})

    report = {
        "protocol": {
            "images": len(image_ids),
            "candidates_per_system_per_image": 3,
            "total_candidates": len(rows),
            "blind_seed": key_doc["seed"],
            "evaluator": score_doc["evaluator"],
            "good_threshold": good_threshold,
            "bootstrap_resamples": args.bootstrap,
            "bootstrap_unit": "image",
        },
        "systems": summary,
        "pairwise": pairwise,
        "best_of_3_wins_including_ties": win_counts,
        "best_of_3_unique_wins": unique_win_counts,
        "per_image_winners": per_image_winners,
        "unblinded_rows": sorted(rows, key=lambda x: (x["image_id"], x["system"], x["blind_id"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_image_winners", "unblinded_rows"}}, indent=2))


if __name__ == "__main__":
    main()
