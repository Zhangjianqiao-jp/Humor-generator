#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

from src.evaluation.ranker import rank_candidates
from src.utils.io import read_jsonl, write_jsonl

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--input-jsonl", type=Path, default=Path("outputs/generations/candidates.jsonl"))
    p.add_argument("--output-jsonl", type=Path, default=Path("outputs/generations/ranked_top5.jsonl"))
    args = p.parse_args()
    rows = read_jsonl(args.input_jsonl)
    out = [rank_candidates(r["image"], r["image_id"], r["candidates"]) for r in rows]
    write_jsonl(args.output_jsonl, out)
