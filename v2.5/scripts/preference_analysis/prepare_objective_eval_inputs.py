#!/usr/bin/env python3
"""Materialize one fixed-hint held-out evaluation row per image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seen: set[str] = set()
    rows = []
    with args.pairs.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            image_id = str(row["image_id"])
            if image_id in seen:
                continue
            seen.add(image_id)
            rows.append(
                {
                    "image": row["image"],
                    "image_id": image_id,
                    "prompt": row["prompt"],
                    "gold_caption": row["chosen"],
                    "source_pair_id": row.get("pair_id"),
                    "evaluation_condition": "fixed_same_hint",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[objective-eval] wrote {len(rows)} image-distinct rows to {args.output}")


if __name__ == "__main__":
    main()
