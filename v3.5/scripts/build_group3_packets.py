#!/usr/bin/env python3
"""Blind four-baseline generations into pairwise Group-of-3 packets."""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import read_jsonl
from humor_generator_v35.evaluation.formal import build_group3_packets


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--private-mapping", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--comparison", action="append", default=[], metavar="REFERENCE:CHALLENGER",
        help="repeatable planned comparison; defaults to text_homer versus every other condition",
    )
    parser.add_argument("--include-standard-description", action="store_true")
    args = parser.parse_args()
    comparisons = []
    for value in args.comparison:
        if value.count(":") != 1:
            parser.error(f"invalid --comparison {value!r}; expected REFERENCE:CHALLENGER")
        comparisons.append(tuple(value.split(":", 1)))
    packets, mapping = build_group3_packets(
        read_jsonl(args.generations),
        comparisons=comparisons or None,
        seed=args.seed,
        include_standard_description=args.include_standard_description,
    )
    args.packet.parent.mkdir(parents=True, exist_ok=True)
    args.private_mapping.parent.mkdir(parents=True, exist_ok=True)
    args.packet.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packets))
    args.private_mapping.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mapping)
    )
    print(json.dumps({"packets": len(packets), "mapping": len(mapping)}))


if __name__ == "__main__":
    main()
