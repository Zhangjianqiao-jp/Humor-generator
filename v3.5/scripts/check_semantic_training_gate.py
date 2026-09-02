#!/usr/bin/env python3
"""Fail closed unless a semantic bridge demonstrably learns and uses memory."""
from __future__ import annotations

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import random
import statistics

import yaml


CHANNELS = ("conflict", "local", "global")


def bootstrap_mean_interval(values: list[float], *, seed: int, draws: int = 10000) -> list[float]:
    if len(values) < 2:
        raise ValueError("cluster bootstrap requires at least two values")
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    )
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report-only", action="store_true",
        help="Write a scientific No-Go report without making the scheduler job fail.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("metrics contain no completed epoch")
    required = {
        "matched_minus_shuffled_logp", "fraction_gap_gt_0", "caption_nll",
        "mean_gate", "mean_relative_update_norm", "info_nce",
        "info_nce_retrieval_at_1",
    }
    for row in rows:
        missing = required - set(row["validation"])
        if missing:
            raise RuntimeError(f"validation metrics missing {sorted(missing)}")
        if any(not math.isfinite(float(value)) for value in row["validation"].values()):
            raise RuntimeError("non-finite validation metric")
    gate = config["gate"]
    first_nll = float(rows[0]["validation"]["caption_nll"])
    phase_a3 = str(config["loss"].get("semantic_objective")) == "channel_balanced_v3"

    if phase_a3:
        required_channel_metrics = {
            f"{metric}_{channel}"
            for channel in CHANNELS
            for metric in (
                "matched_minus_shuffled_logp", "fraction_gap_gt_0",
                "caption_nll", "info_nce_retrieval_at_1",
            )
        }
        for row in rows:
            missing = required_channel_metrics - set(row["validation"])
            if missing:
                raise RuntimeError(f"Phase A3 metrics missing {sorted(missing)}")

    def checks_for(row: dict) -> dict[str, bool]:
        values = row["validation"]
        if phase_a3:
            result: dict[str, bool] = {}
            for channel in CHANNELS:
                result[f"positive_gap_{channel}"] = (
                    float(values[f"matched_minus_shuffled_logp_{channel}"])
                    >= float(gate["min_channel_validation_gap"])
                )
                result[f"majority_positive_{channel}"] = (
                    float(values[f"fraction_gap_gt_0_{channel}"])
                    >= float(gate["min_channel_fraction_gap_gt_0"])
                )
                result[f"retrieval_{channel}"] = (
                    float(values[f"info_nce_retrieval_at_1_{channel}"])
                    >= float(gate["min_channel_retrieval_at_1"])
                )
            result.update({
                "nll_improved": first_nll - float(values["caption_nll"])
                >= float(gate["min_nll_improvement"]),
                "bounded_residual_update": float(values["mean_relative_update_norm"])
                <= float(gate["max_relative_update_norm"]),
                "fixed_equal_channel_mass": all(
                    abs(float(values[f"mean_channel_weight_{channel}"]) - 1 / 3)
                    <= float(gate.get("equal_channel_weight_tolerance", 1e-4))
                    for channel in CHANNELS
                ),
            })
            return result
        return {
            "validation_gap": float(values["matched_minus_shuffled_logp"]) >= float(gate["min_validation_gap"]),
            "majority_positive": float(values["fraction_gap_gt_0"]) >= float(gate["min_fraction_gap_gt_0"]),
            "nll_improved": first_nll - float(values["caption_nll"]) >= float(gate["min_nll_improvement"]),
            "bounded_residual_update": float(values["mean_relative_update_norm"]) <= float(gate["max_relative_update_norm"]),
            "nonzero_gate": abs(float(values["mean_gate"])) > 1e-4,
            "retrieval_above_chance": (
                float(values["info_nce_retrieval_at_1"])
                >= float(gate["min_validation_info_nce_retrieval_at_1"])
            ),
        }

    # Select a real jointly-valid epoch when one exists.  Picking the minimum
    # NLL epoch first could hide that a different checkpoint was the only one
    # that actually used matched memory.
    passing = [row for row in rows if all(checks_for(row).values())]
    candidates = passing or rows
    best = min(candidates, key=lambda row: float(row["validation"]["total"]))
    values = best["validation"]
    checks = checks_for(best)
    intervals = {}
    ci_checks = {}
    if phase_a3:
        detail_path = args.metrics.parent / f"validation_epoch_{best['epoch']}.jsonl"
        if not detail_path.is_file():
            raise RuntimeError(f"missing per-cluster validation details: {detail_path}")
        details = [json.loads(line) for line in detail_path.read_text().splitlines() if line.strip()]
        if len({row["cluster_id"] for row in details}) != len(details):
            raise RuntimeError("validation details are not one row per image cluster")
        for index, channel in enumerate(CHANNELS):
            metric = f"matched_minus_shuffled_logp_{channel}"
            channel_values = [float(row[metric]) for row in details]
            intervals[channel] = bootstrap_mean_interval(
                channel_values, seed=int(config["training"]["seed"]) + index
            )
            ci_checks[channel] = intervals[channel][0] > 0
    point_pass = all(checks.values())
    if phase_a3 and point_pass:
        status = "strong_go" if all(ci_checks.values()) else "go_to_outer_semantic_validation"
    else:
        status = "go" if point_pass else "no_go"
    report = {
        "status": status,
        "selected_epoch": best["epoch"],
        "checks": checks,
        "thresholds": gate,
        "validation": values,
        "image_clustered_bootstrap_95ci": intervals,
        "ci_lower_bound_above_zero": ci_checks,
        "interpretation": (
            "This 24-cluster gate is a low-cost mechanism screen, not evidence of "
            "humorous-caption superiority. A point-estimate pass without three positive "
            "cluster-bootstrap lower bounds proceeds only to a larger sealed semantic validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] not in {"go", "strong_go", "go_to_outer_semantic_validation"} and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
