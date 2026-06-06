#!/usr/bin/env python
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl


def inspect_outputs(input_jsonl: Path, num_samples: int) -> None:
    rows = read_jsonl(input_jsonl)
    for index, row in enumerate(rows[:num_samples]):
        print("=" * 80)
        print(f"Sample {index + 1}")
        print(f"Image: {row.get('image', '')}")
        gold = row.get('gold_caption') or row.get('gold') or row.get('gold_captions') or ''
        if isinstance(gold, list):
            gold = ' | '.join(str(item) for item in gold[:3])
        print(f"Gold: {gold}")
        print(f"Prompt: {row.get('prompt', '')}")
        candidates = row.get("candidates") or []
        if not candidates and row.get("generated_caption"):
            candidates = [row["generated_caption"]]
        for candidate_index, candidate in enumerate(candidates, start=1):
            print(f"Candidate {candidate_index}: {candidate}")


def main() -> None:
    parser = ArgumentParser(description="Print generated SFT captions for manual bad-case inspection.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=30)
    args = parser.parse_args()
    inspect_outputs(args.input_jsonl, args.num_samples)


if __name__ == "__main__":
    main()
