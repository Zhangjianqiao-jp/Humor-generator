#!/usr/bin/env python3
"""Report lexical and formatting shortcuts in chosen/rejected pairs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preference.diagnostics import mean, read_jsonl, sha256, text_features


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = [text_features(str(row["chosen"])) for row in rows]
    rejected = [text_features(str(row["rejected"])) for row in rows]
    names = list(features[0]) if features else []
    result: dict[str, Any] = {}
    for name in names:
        chosen_mean = mean([float(item[name]) for item in features])
        rejected_mean = mean([float(item[name]) for item in rejected])
        result[name] = {
            "chosen_mean": chosen_mean,
            "rejected_mean": rejected_mean,
            "difference": chosen_mean - rejected_mean,
        }
    return result


def warning_lines(stats: dict[str, Any], threshold: float) -> list[str]:
    warnings = []
    for name, values in stats.items():
        difference = float(values["difference"])
        scale = max(abs(float(values["chosen_mean"])), abs(float(values["rejected_mean"])), 1.0)
        standardized = difference / scale
        if abs(standardized) >= threshold:
            direction = "chosen" if difference > 0 else "rejected"
            warnings.append(f"- `{name}` is higher for {direction} (relative difference {standardized:+.3f}).")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/preference_diagnostics/pair_bias_report.md"))
    parser.add_argument("--warning-threshold", type=float, default=0.10)
    args = parser.parse_args()
    rows = read_jsonl(args.pairs)
    required = ("chosen", "rejected")
    for index, row in enumerate(rows):
        missing = [key for key in required if not str(row.get(key) or "").strip()]
        if missing:
            raise ValueError(f"row {index} missing {missing}")
    stats = summarize(rows)
    pair_types = Counter(str(row.get("pair_type") or "unknown") for row in rows)
    lines = [
        "# Preference Pair Bias Report",
        "",
        f"- Input: `{args.pairs}`",
        f"- SHA-256: `{sha256(args.pairs)}`",
        f"- Pairs: {len(rows)}",
        f"- Pair types: `{json.dumps(dict(sorted(pair_types.items())), sort_keys=True)}`",
        "",
        "## Chosen versus rejected features",
        "",
        "| Feature | Chosen mean | Rejected mean | Difference |",
        "|---|---:|---:|---:|",
    ]
    for name, values in stats.items():
        lines.append(
            f"| {name} | {values['chosen_mean']:.4f} | {values['rejected_mean']:.4f} | {values['difference']:+.4f} |"
        )
    warnings = warning_lines(stats, args.warning_threshold)
    lines.extend(["", "## Potential shortcuts", ""])
    lines.extend(warnings or ["No heuristic feature crossed the configured relative-difference threshold."])
    lines.extend(
        [
            "",
            "These checks diagnose surface confounders; they do not establish that a pair is visually grounded or humorously valid.",
            "",
            "## Authoritative methodological references",
            "",
            "- Rafailov et al. (2023), Direct Preference Optimization, NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html",
            "- Wang et al. (2024), mDPO: Conditional Preference Optimization for Multimodal Large Language Models: https://arxiv.org/abs/2406.11839",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[pair-bias] wrote {args.output} for {len(rows)} pairs")


if __name__ == "__main__":
    main()
