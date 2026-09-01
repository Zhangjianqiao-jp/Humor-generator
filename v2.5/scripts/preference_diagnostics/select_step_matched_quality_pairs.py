#!/usr/bin/env python
"""Select an image-diverse, tier-stratified fixed-size Quality64 DPO pilot.

The selector never changes chosen/rejected labels. It allocates the requested
pair budget across quality tiers in proportion to the full pool, then samples
each tier round-robin across images. This avoids a file-prefix sample that
would cover only the first few contests.
"""
from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tier_name(row: dict[str, Any]) -> str:
    return str(row.get("quality_tier", "unknown")).split("_quota_fallback", 1)[0]


def proportional_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    if total < 1 or total > sum(counts.values()):
        raise ValueError(f"Requested total {total} is outside [1, {sum(counts.values())}].")
    exact = {key: total * value / sum(counts.values()) for key, value in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def round_robin_by_image(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_id"])].append(row)
    image_ids = sorted(grouped)
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < limit:
        added = 0
        for image_id in image_ids:
            candidates = grouped[image_id]
            if depth < len(candidates):
                selected.append(candidates[depth])
                added += 1
                if len(selected) == limit:
                    break
        if added == 0:
            raise ValueError(f"Only found {len(selected)} rows for requested tier quota {limit}.")
        depth += 1
    return selected


def select_rows(rows: list[dict[str, Any]], total: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tier[tier_name(row)].append(row)
    counts = {key: len(value) for key, value in sorted(by_tier.items())}
    quotas = proportional_quotas(counts, total)
    selected: list[dict[str, Any]] = []
    for key in sorted(by_tier):
        selected.extend(round_robin_by_image(by_tier[key], quotas[key]))
    selected.sort(key=lambda row: (str(row["image_id"]), str(row["pair_id"])))
    if len(selected) != total or len({str(row["pair_id"]) for row in selected}) != total:
        raise AssertionError("The selected pilot must contain exactly the requested number of unique pairs.")
    return selected, quotas


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = ArgumentParser(description="Build a step-matched Quality64 DPO pilot.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=1264)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    selected, quotas = select_rows(rows, args.pairs)
    if any("reference_logps" not in row for row in selected):
        raise ValueError("Every selected row must contain frozen reference_logps.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source": str(args.input),
        "output": str(args.output),
        "selection": "proportional-tier quotas plus deterministic image round-robin",
        "labels_changed": False,
        "pairs": len(selected),
        "images": len({str(row["image_id"]) for row in selected}),
        "tier_quotas": quotas,
        "source_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
