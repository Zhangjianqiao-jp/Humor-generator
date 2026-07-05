#!/usr/bin/env python
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.review_html import write_guided_review_html
from src.utils.io import read_jsonl


def main() -> None:
    parser = ArgumentParser(description="Render VLM-guided candidate JSONL as an HTML review page.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    rows = read_jsonl(args.input_jsonl)
    write_guided_review_html(rows, args.output_html, max_rows=args.max_rows)
    print(f"[visualize] saved {args.output_html}")


if __name__ == "__main__":
    main()

