#!/usr/bin/env python3
"""Build deterministic swapped-plan and target-corrupted captioner inputs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_captioner_inputs_from_plans import build_compact_json_caption_prompt
from src.utils.io import read_jsonl, write_jsonl


REQUIRED_KEYS = {
    "scene",
    "type",
    "target",
    "primary_view",
    "views",
    "anchors",
    "external_knowledge",
}


def parse_payload(row: dict[str, Any], index: int) -> dict[str, Any]:
    try:
        payload = json.loads(str(row["compact_json"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"row {index} has invalid compact_json: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != REQUIRED_KEYS:
        raise ValueError(f"row {index} compact_json has unexpected keys")
    return payload


def donor_map(image_ids: list[str], seed: int) -> dict[str, str]:
    if len(image_ids) < 2:
        raise ValueError("at least two images are required for counterfactuals")
    order = list(image_ids)
    random.Random(seed).shuffle(order)
    mapping = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    if set(mapping) != set(mapping.values()) or any(k == v for k, v in mapping.items()):
        raise AssertionError("failed to construct a one-to-one derangement")
    return mapping


def render_row(
    base: dict[str, Any],
    payload: dict[str, Any],
    condition: str,
    donor_image_id: str,
) -> dict[str, Any]:
    prompt, compact_json = build_compact_json_caption_prompt(payload)
    return {
        "image": base["image"],
        "image_id": base["image_id"],
        "source_image_id": base.get("source_image_id") or base["image_id"],
        "prompt": prompt,
        "planner_candidate": compact_json,
        "compact_json": compact_json,
        "counterfactual_condition": condition,
        "donor_image_id": donor_image_id,
        "gold_caption": base.get("gold_caption", ""),
        "gold_captions": base.get("gold_captions"),
    }


def build(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids = [str(row.get("image_id") or "").strip() for row in rows]
    if any(not image_id for image_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("input rows contain missing or duplicate image_id")
    payloads = {image_id: parse_payload(row, i) for i, (image_id, row) in enumerate(zip(ids, rows))}
    rows_by_id = dict(zip(ids, rows))
    donors = donor_map(ids, seed)

    swapped: list[dict[str, Any]] = []
    corrupted: list[dict[str, Any]] = []
    for image_id in ids:
        donor_id = donors[image_id]
        base = rows_by_id[image_id]
        correct = payloads[image_id]
        donor = payloads[donor_id]
        swapped.append(render_row(base, dict(donor), "swapped_full_plan", donor_id))

        # Preserve literal grounding while corrupting only the proposed humor bridge.
        target_corrupted = dict(correct)
        target_corrupted["type"] = donor["type"]
        target_corrupted["target"] = donor["target"]
        target_corrupted["external_knowledge"] = donor["external_knowledge"]
        corrupted.append(
            render_row(base, target_corrupted, "target_corrupted", donor_id)
        )
    return swapped, corrupted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correct-inputs", type=Path, required=True)
    parser.add_argument("--swapped-output", type=Path, required=True)
    parser.add_argument("--corrupted-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    rows = read_jsonl(args.correct_inputs)
    swapped, corrupted = build(rows, args.seed)
    write_jsonl(args.swapped_output, swapped)
    write_jsonl(args.corrupted_output, corrupted)
    print(
        f"[counterfactual] saved {len(swapped)} swapped and {len(corrupted)} "
        f"target-corrupted rows; seed={args.seed}"
    )


if __name__ == "__main__":
    main()
