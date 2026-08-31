#!/usr/bin/env python3
"""Aggregate one or more independent Group-of-3 judge JSON files."""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import read_jsonl
from humor_generator_v35.evaluation.formal import aggregate_group3


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--private-mapping", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ratings = [json.loads(path.read_text()) for path in args.ratings]
    report = aggregate_group3(read_jsonl(args.private_mapping), ratings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
