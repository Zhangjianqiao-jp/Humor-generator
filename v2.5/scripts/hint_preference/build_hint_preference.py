#!/usr/bin/env python3
"""Aggregate repeated-caption rewards and construct high-confidence hint pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preference.hint_utility import build_hint_pairs, summarize_hint_rows
from src.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scored-hints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/hint_preference"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    utility = config["hint_utility"]
    weights = {str(key): float(value) for key, value in utility["weights"].items()}
    summaries = summarize_hint_rows(read_jsonl(args.scored_hints), weights)
    pairs = build_hint_pairs(
        summaries,
        min_margin=float(utility.get("min_utility_margin", 0.5)),
        require_dominance=bool(utility.get("require_pareto_dominance", True)),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "hint_utilities.jsonl", summaries)
    write_jsonl(args.output_dir / "hint_preference_pairs.jsonl", pairs)
    manifest = {
        "scored_hints": str(args.scored_hints),
        "weights": weights,
        "caption_samples_required": int(utility["caption_samples_per_hint"]),
        "min_utility_margin": float(utility.get("min_utility_margin", 0.5)),
        "require_pareto_dominance": bool(utility.get("require_pareto_dominance", True)),
        "hints": len(summaries),
        "pairs": len(pairs),
        "warning": "Preference is downstream utility under the frozen Generator, not intrinsic Hint prose quality.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
