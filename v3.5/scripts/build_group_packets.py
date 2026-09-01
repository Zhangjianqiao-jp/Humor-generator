#!/usr/bin/env python3
"""Build paper-aligned Group-of-10 or legacy Group-of-3 blind packets."""
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
    parser.add_argument("--generations", type=Path, nargs="+", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--private-mapping", type=Path, required=True)
    parser.add_argument("--group-size", type=int, choices=[3, 10], default=10)
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--calibration", type=Path, required=True,
        help="JSON list of five image-grounded A/B examples from non-test clusters",
    )
    parser.add_argument(
        "--mirror-sides", action="store_true",
        help="emit both A/B orientations, as in the Humor-in-AI evaluator",
    )
    parser.add_argument("--comparison", action="append", default=[], metavar="REFERENCE:CHALLENGER")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--include-standard-description", action="store_true")
    args = parser.parse_args()
    comparisons = []
    for value in args.comparison:
        if value.count(":") != 1:
            parser.error(f"invalid comparison {value!r}")
        comparisons.append(tuple(value.split(":", 1)))
    generations = [row for path in args.generations for row in read_jsonl(path)]
    packets, mapping = build_group3_packets(
        generations, comparisons=comparisons or None,
        seed=args.seed, include_standard_description=args.include_standard_description,
        group_size=args.group_size,
        comparison_family=args.family, mirror_sides=args.mirror_sides,
        calibration_examples=json.loads(args.calibration.read_text()),
    )
    args.packet.parent.mkdir(parents=True, exist_ok=True)
    args.private_mapping.parent.mkdir(parents=True, exist_ok=True)
    args.packet.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packets))
    args.private_mapping.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mapping))
    print(json.dumps({"group_size": args.group_size, "packets": len(packets)}))


if __name__ == "__main__":
    main()
