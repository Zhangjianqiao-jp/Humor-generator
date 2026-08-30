#!/usr/bin/env python
"""Blind two caption-generation systems per image and retain a private key."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_jsonl


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id = str(row.get("image_id") or "").strip()
        candidates = row.get("candidates")
        if not image_id or image_id in indexed:
            raise ValueError(f"missing or duplicate image_id: {image_id!r}")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{image_id}: candidates must be a non-empty list")
        indexed[image_id] = row
    return indexed


def blind_rows(
    joint_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    joint = index_rows(joint_rows)
    direct = index_rows(direct_rows)
    if set(joint) != set(direct):
        raise ValueError("joint and direct files do not contain identical image IDs")

    rng = random.Random(seed)
    public: list[dict[str, Any]] = []
    key: list[dict[str, str]] = []
    for image_id in sorted(joint):
        systems = ["joint", "direct"]
        rng.shuffle(systems)
        for blind_label, system in zip(("A", "B"), systems, strict=True):
            source = joint[image_id] if system == "joint" else direct[image_id]
            for candidate_index, caption in enumerate(source["candidates"], start=1):
                blind_id = hashlib.sha256(
                    f"{seed}:{image_id}:{blind_label}:{candidate_index}".encode()
                ).hexdigest()[:12]
                public.append(
                    {
                        "blind_id": blind_id,
                        "image_id": image_id,
                        "image": source["image"],
                        "system_label": blind_label,
                        "candidate_index": candidate_index,
                        "caption": str(caption).strip(),
                    }
                )
                key.append({"blind_id": blind_id, "image_id": image_id, "system": system})
    return public, key


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--blind-output", type=Path, required=True)
    parser.add_argument("--key-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    public, key = blind_rows(read_jsonl(args.joint), read_jsonl(args.direct), args.seed)
    write_jsonl(args.blind_output, public)
    args.key_output.parent.mkdir(parents=True, exist_ok=True)
    args.key_output.write_text(
        json.dumps({"seed": args.seed, "key": key}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[blind] saved {len(public)} public candidates and {len(key)} private key rows")


if __name__ == "__main__":
    main()
