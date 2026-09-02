#!/usr/bin/env python3
"""Fail closed unless a semantic bridge demonstrably learns and uses memory."""
from __future__ import annotations

from argparse import ArgumentParser
import json
import math
from pathlib import Path

import yaml


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    def checks_for(row: dict) -> dict[str, bool]:
        values = row["validation"]
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
    report = {
        "status": "go" if all(checks.values()) else "no_go",
        "selected_epoch": best["epoch"],
        "checks": checks,
        "thresholds": gate,
        "validation": values,
        "interpretation": (
            "This gate establishes decodability/sensitivity, not humorous-caption superiority."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "go":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
