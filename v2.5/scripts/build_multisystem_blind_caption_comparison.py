#!/usr/bin/env python3
"""Blind any number of caption systems per image and retain a private key."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_jsonl


def index_rows(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        image_id = str(row.get("image_id") or "").strip()
        candidates = row.get("candidates")
        if not image_id or image_id in indexed:
            raise ValueError(f"{path}: missing or duplicate image_id {image_id!r}")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{path}: {image_id} has no candidates")
        indexed[image_id] = row
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=JSONL",
        help="Repeat for every system.",
    )
    parser.add_argument("--blind-output", type=Path, required=True)
    parser.add_argument("--key-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    systems: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.system:
        if "=" not in spec:
            raise ValueError(f"Invalid --system value: {spec!r}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not name or name in systems:
            raise ValueError(f"Missing or duplicate system name: {name!r}")
        systems[name] = index_rows(Path(raw_path))
    if len(systems) < 2:
        raise ValueError("At least two systems are required")

    image_sets = [set(rows) for rows in systems.values()]
    if any(ids != image_sets[0] for ids in image_sets[1:]):
        raise ValueError("All systems must contain identical image IDs")

    rng = random.Random(args.seed)
    public: list[dict[str, Any]] = []
    key: list[dict[str, str]] = []
    labels = [chr(ord("A") + index) for index in range(len(systems))]
    for image_id in sorted(image_sets[0]):
        names = list(systems)
        rng.shuffle(names)
        for label, name in zip(labels, names, strict=True):
            row = systems[name][image_id]
            for candidate_index, caption in enumerate(row["candidates"], start=1):
                blind_id = hashlib.sha256(
                    f"{args.seed}:{image_id}:{label}:{candidate_index}".encode()
                ).hexdigest()[:12]
                public.append(
                    {
                        "blind_id": blind_id,
                        "image_id": image_id,
                        "image": row["image"],
                        "system_label": label,
                        "candidate_index": candidate_index,
                        "caption": str(caption).strip(),
                    }
                )
                key.append(
                    {"blind_id": blind_id, "image_id": image_id, "system": name}
                )

    write_jsonl(args.blind_output, public)
    args.key_output.parent.mkdir(parents=True, exist_ok=True)
    args.key_output.write_text(
        json.dumps(
            {"seed": args.seed, "systems": list(systems), "key": key},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[blind] saved {len(public)} candidates across {len(systems)} systems "
        f"and {len(image_sets[0])} images"
    )


if __name__ == "__main__":
    main()
