#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import read_jsonl
from humor_generator_v35.evaluation.diversity import summarize_diversity


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-candidates", type=int, default=10)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--skip-sbert", action="store_true")
    args = parser.parse_args()
    model = None
    if not args.skip_sbert:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.sbert_model)
    report = summarize_diversity(
        read_jsonl(args.generations), min_candidates=args.min_candidates, sbert_model=model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"systems": len(report["system_summary"]), "clusters": len(report["per_cluster"])}))


if __name__ == "__main__":
    main()
