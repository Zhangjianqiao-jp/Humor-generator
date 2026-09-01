#!/usr/bin/env python3
"""Derive internally consistent pair decisions from blinded group-level ratings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def group_id(group: list[str]) -> str:
    payload = json.dumps(group, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.ratings.open(encoding="utf-8", newline="") as handle:
        ratings = {
            row["group_id"]: row
            for row in csv.DictReader(
                (line for line in handle if not line.startswith("#")), delimiter="\t"
            )
            if int(row["seed"]) == args.seed
        }
    rows = [json.loads(line) for line in args.public.read_text(encoding="utf-8").splitlines() if line]
    decisions = {}
    for row in rows:
        left = ratings.get(group_id(row["group_A"]))
        right = ratings.get(group_id(row["group_B"]))
        if left is None or right is None:
            raise ValueError(f"{row['pair_id']}: missing group-level rating")

        def compare(field: str) -> str:
            a, b = float(left[field]), float(right[field])
            return "A" if a > b else "B" if b > a else "Tie"

        decisions[row["pair_id"]] = {
            "overall": compare("overall_score"),
            "best_pick": compare("best_score"),
            "best_A_index": int(left["best_index"]),
            "best_B_index": int(right["best_index"]),
            "absolute_A": left["absolute"],
            "absolute_B": right["absolute"],
        }
    args.output.write_text(
        json.dumps(
            {
                "protocol": "NeurIPS-2024-style Group-of-3 blinded comparison",
                "judge": "Codex independent blinded group-level rating; pair decisions derived consistently",
                "generation_seed": args.seed,
                "decisions": decisions,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
