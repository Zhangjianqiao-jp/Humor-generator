#!/usr/bin/env python3
"""Summarize controlled objective-screen training and generation metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preference.diagnostics import write_csv
from src.utils.io import read_jsonl

GENERIC = re.compile(r"\b(?:pov|bro|meanwhile|lol|lmao|bruh)\b|[💀😂]", re.IGNORECASE)
TOKENS = re.compile(r"\b\w+(?:['’]\w+)?\b")


def final_metrics(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    baseline = next((row for row in rows if row.get("split") == "validation_baseline"), None)
    final = next((row for row in reversed(rows) if row.get("split") == "validation_final"), None)
    if baseline is None or final is None:
        raise ValueError(f"{path} lacks baseline or final validation metrics")
    return {
        "objective": final["objective"],
        "baseline_loss": baseline["eval_loss"],
        "final_loss": final["eval_loss"],
        "delta_chosen_logp": final["eval_chosen_logp"] - baseline["eval_chosen_logp"],
        "delta_rejected_logp": final["eval_rejected_logp"] - baseline["eval_rejected_logp"],
        "final_reward_margin": final["eval_reward_margin"],
        "final_preference_accuracy": final["eval_reward_accuracy"],
    }


def generation_metrics(path: Path) -> dict[str, float]:
    rows = read_jsonl(path)
    captions = [str(value).strip() for row in rows for value in row.get("candidates", []) if str(value).strip()]
    unigrams = [token.lower() for text in captions for token in TOKENS.findall(text)]
    bigrams = [tuple(tokens[index : index + 2]) for text in captions for tokens in [[x.lower() for x in TOKENS.findall(text)]] for index in range(len(tokens) - 1)]
    return {
        "mean_words": sum(len(TOKENS.findall(text)) for text in captions) / max(len(captions), 1),
        "distinct_1": len(set(unigrams)) / max(len(unigrams), 1),
        "distinct_2": len(set(bigrams)) / max(len(bigrams), 1),
        "generic_template_rate": sum(bool(GENERIC.search(text)) for text in captions) / max(len(captions), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, default=Path("outputs/preference_screen"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/preference_learning/objective_screen"))
    args = parser.parse_args()
    rows = []
    for directory in sorted(args.screen_root.iterdir()):
        metrics = directory / "train_metrics.jsonl"
        if not metrics.exists():
            continue
        row = final_metrics(metrics)
        generations = directory / "eval" / "generations.jsonl"
        if generations.exists():
            row.update(generation_metrics(generations))
        judge = directory / "eval" / "judge_summary.json"
        if judge.exists():
            payload = json.loads(judge.read_text(encoding="utf-8"))
            row.update({f"judge_{key}": value for key, value in payload.get("avg_scores", {}).items()})
            row["judge_hallucination_rate"] = payload.get("hallucination_rate")
            row["judge_qualified_rate"] = payload.get("qualified_rate")
        rows.append(row)
    if not rows:
        raise ValueError(f"No completed objective metrics below {args.screen_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "objective_screen.csv", rows)
    (args.output_dir / "objective_screen.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
