#!/usr/bin/env python3
"""Merge complete, disjoint response parts from one blinded LLM judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", action="append", type=Path, required=True)
    parser.add_argument("--expected-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_doc = json.load(args.expected_template.open(encoding="utf-8"))
    rater_id = expected_doc["rater_id"]
    expected = set(expected_doc["decisions"])
    merged = {}
    for path in args.part:
        doc = json.load(path.open(encoding="utf-8"))
        if doc.get("rater_id") != rater_id:
            raise ValueError(f"{path}: wrong rater_id")
        overlap = set(merged) & set(doc.get("decisions", {}))
        if overlap:
            raise ValueError(f"{path}: duplicate blind IDs: {sorted(overlap)[:3]}")
        merged.update(doc.get("decisions", {}))
    if set(merged) != expected:
        raise ValueError(f"response IDs differ: missing={len(expected-set(merged))}, extra={len(set(merged)-expected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rater_id": rater_id, "decisions": merged}, indent=2) + "\n")
    print(f"merged {len(args.part)} parts and {len(merged)} decisions into {args.output}")


if __name__ == "__main__":
    main()
