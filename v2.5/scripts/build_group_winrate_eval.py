#!/usr/bin/env python3
"""Build a blinded, position-balanced group-comparison evaluation packet."""

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


def index_system(path: Path, expected_candidates: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        image_id = str(row.get("image_id") or "").strip()
        candidates = row.get("candidates")
        if not image_id or image_id in result:
            raise ValueError(f"{path}: missing or duplicate image_id {image_id!r}")
        if not isinstance(candidates, list) or len(candidates) != expected_candidates:
            raise ValueError(
                f"{path}: {image_id} needs exactly {expected_candidates} candidates"
            )
        result[image_id] = row
    return result


def parse_named_paths(specs: list[str], expected_candidates: int) -> dict[str, dict[str, dict[str, Any]]]:
    systems: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid system specification: {spec!r}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not name or name in systems:
            raise ValueError(f"missing or duplicate system name: {name!r}")
        systems[name] = index_system(Path(raw_path), expected_candidates)
    image_sets = [set(rows) for rows in systems.values()]
    if len(systems) < 2 or any(ids != image_sets[0] for ids in image_sets[1:]):
        raise ValueError("all systems must contain identical image IDs")
    return systems


def build_packet(
    systems: dict[str, dict[str, dict[str, Any]]],
    comparisons: list[tuple[str, str]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    image_ids = sorted(next(iter(systems.values())))
    public: list[dict[str, Any]] = []
    key: list[dict[str, str]] = []
    rng = random.Random(seed)
    for comparison_index, (system_a, system_b) in enumerate(comparisons):
        if system_a not in systems or system_b not in systems or system_a == system_b:
            raise ValueError(f"invalid comparison: {system_a}:{system_b}")
        ordered_images = list(image_ids)
        rng.shuffle(ordered_images)
        # Exact balance when N is even; at most one-position imbalance otherwise.
        a_on_left = set(ordered_images[: (len(ordered_images) + 1) // 2])
        for image_id in image_ids:
            left, right = (system_a, system_b) if image_id in a_on_left else (system_b, system_a)
            source = systems[left][image_id]
            pair_id = hashlib.sha256(
                f"{seed}:{comparison_index}:{image_id}".encode()
            ).hexdigest()[:12]
            public.append(
                {
                    "pair_id": pair_id,
                    "image_id": image_id,
                    "image": source["image"],
                    "group_A": [str(x).strip() for x in systems[left][image_id]["candidates"]],
                    "group_B": [str(x).strip() for x in systems[right][image_id]["candidates"]],
                }
            )
            key.append(
                {
                    "pair_id": pair_id,
                    "image_id": image_id,
                    "comparison": f"{system_a}_vs_{system_b}",
                    "system_a": system_a,
                    "system_b": system_b,
                    "group_A_system": left,
                    "group_B_system": right,
                }
            )
    rng.shuffle(public)
    key_doc = {
        "seed": seed,
        "systems": list(systems),
        "comparisons": [f"{a}_vs_{b}" for a, b in comparisons],
        "key": key,
    }
    template = {
        "protocol": "NeurIPS-2024-style Group-of-3 blinded comparison",
        "instructions": {
            "overall": "Choose the group whose three captions are funnier overall: A, B, or Tie. Use Tie when the groups are identical or genuinely indistinguishable.",
            "best_pick": "Pick the funniest caption in each group, then choose A, B, or Tie when the two best captions are indistinguishable.",
            "allowed_labels": ["A", "B", "Tie"],
            "absolute_quality": {
                "good": "At least one caption is genuinely usable, image-grounded, and has a clear humorous turn.",
                "weak": "The best caption is relevant or mildly amusing, but generic, strained, or not clearly funny.",
                "bad": "All captions are off-image, incoherent, literal without humor, or unusable.",
            },
        },
        "decisions": {
            row["pair_id"]: {
                "overall": "",
                "best_pick": "",
                "best_A_index": None,
                "best_B_index": None,
                "absolute_A": "",
                "absolute_B": "",
            }
            for row in public
        },
    }
    return public, key_doc, template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        metavar="SYSTEM_A:SYSTEM_B",
        help="Win rate is reported from SYSTEM_A's perspective.",
    )
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--key-output", type=Path, required=True)
    parser.add_argument("--template-output", type=Path, required=True)
    parser.add_argument("--candidates-per-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    systems = parse_named_paths(args.system, args.candidates_per_group)
    comparisons = []
    for spec in args.comparison:
        if ":" not in spec:
            raise ValueError(f"invalid comparison specification: {spec!r}")
        comparisons.append(tuple(x.strip() for x in spec.split(":", 1)))
    public, key_doc, template = build_packet(systems, comparisons, args.seed)
    write_jsonl(args.public_output, public)
    for path, doc in ((args.key_output, key_doc), (args.template_output, template)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[group-eval] saved {len(public)} blinded trials across "
        f"{len(comparisons)} comparisons and {len(next(iter(systems.values())))} images"
    )


if __name__ == "__main__":
    main()
