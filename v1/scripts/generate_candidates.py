#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

from src.inference.generate_candidates import generate_candidates


def str2bool(v: str) -> bool:
    return v.lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--input-jsonl", type=Path, default=Path("data/processed/sft_test.jsonl"))
    p.add_argument("--output-jsonl", type=Path, default=Path("outputs/generations/candidates.jsonl"))
    p.add_argument("--num-candidates", type=int, default=10)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--dry-run", type=str2bool, default=True)
    args = p.parse_args()
    generate_candidates(args.input_jsonl, args.output_jsonl, args.num_candidates, args.model_name, args.dry_run)
