#!/usr/bin/env python
from __future__ import annotations

import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        out = rows[:]
        rng.shuffle(out)
        return out
    return rng.sample(rows, count)


def main() -> None:
    parser = ArgumentParser(description="Mix strong/weak/hard-negative reranker pair JSONL files by ratio.")
    parser.add_argument("--strong-jsonl", type=Path, default=Path("data/processed/reranker_score_pools_strict/strong_pairs.jsonl"))
    parser.add_argument("--weak-jsonl", type=Path, default=Path("data/processed/reranker_score_pools_strict/weak_pairs.jsonl"))
    parser.add_argument("--literal-jsonl", type=Path, default=Path("data/processed/reranker_hard_negatives/literal_pairs.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/processed/reranker_hard_negatives/mixed_strong_weak_literal_pairs.jsonl"))
    parser.add_argument("--target-size", type=int, default=300000)
    parser.add_argument("--strong-ratio", type=float, default=0.60)
    parser.add_argument("--weak-ratio", type=float, default=0.20)
    parser.add_argument("--literal-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_ratio = args.strong_ratio + args.weak_ratio + args.literal_ratio
    if total_ratio <= 0:
        raise ValueError("Ratios must sum to a positive value.")
    strong_count = int(args.target_size * args.strong_ratio / total_ratio)
    weak_count = int(args.target_size * args.weak_ratio / total_ratio)
    literal_count = args.target_size - strong_count - weak_count

    rng = random.Random(args.seed)
    strong = read_jsonl(args.strong_jsonl)
    weak = read_jsonl(args.weak_jsonl)
    literal = read_jsonl(args.literal_jsonl)
    rows = sample(strong, strong_count, rng) + sample(weak, weak_count, rng) + sample(literal, literal_count, rng)
    rng.shuffle(rows)
    write_jsonl(args.output_jsonl, rows)

    summary = {
        "output_jsonl": str(args.output_jsonl),
        "target_size": args.target_size,
        "actual_size": len(rows),
        "available": {"strong": len(strong), "weak": len(weak), "literal": len(literal)},
        "requested": {"strong": strong_count, "weak": weak_count, "literal": literal_count},
        "ratios": {"strong": args.strong_ratio, "weak": args.weak_ratio, "literal": args.literal_ratio},
    }
    with args.output_jsonl.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
