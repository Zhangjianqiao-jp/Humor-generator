#!/usr/bin/env python
"""Apply a preregistered validation-only gate to a step-matched DPO pilot."""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any


def final_validation(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [row for row in rows if row.get("split") == "validation_final"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one validation_final row in {path}, found {len(matches)}.")
    return matches[0]


def compare(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    delta_loss = float(old["eval_loss"]) - float(new["eval_loss"])
    delta_accuracy = float(new["eval_reward_accuracy"]) - float(old["eval_reward_accuracy"])
    delta_margin = float(new["eval_reward_margin"]) - float(old["eval_reward_margin"])
    chosen_delta = float(new["eval_chosen_logp"]) - float(old["eval_chosen_logp"])
    checks = {
        "loss_improvement_at_least_0.001": delta_loss >= 0.001,
        "secondary_support_accuracy_or_margin": delta_accuracy >= 0.0 or delta_margin >= 0.0,
        "reward_accuracy_not_down_more_than_0.01": delta_accuracy >= -0.01,
        "reward_margin_not_down_more_than_0.001": delta_margin >= -0.001,
        "chosen_logp_not_down_more_than_0.1": chosen_delta >= -0.1,
    }
    return {
        "decision": "GO_FULL_QUALITY64" if all(checks.values()) else "NO_GO_FULL_QUALITY64",
        "scope": "validation-only; test47 remains sealed",
        "old": old,
        "new": new,
        "delta_new_minus_old": {
            "eval_loss": -delta_loss,
            "eval_reward_accuracy": delta_accuracy,
            "eval_reward_margin": delta_margin,
            "eval_chosen_logp": chosen_delta,
        },
        "checks": checks,
    }


def main() -> None:
    parser = ArgumentParser(description="Compare Pilot16 and Quality64 step-matched DPO validation.")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(final_validation(args.old), final_validation(args.new))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
