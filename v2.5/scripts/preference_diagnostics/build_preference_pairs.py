#!/usr/bin/env python3
"""Construct conservative, same-image H1-H4 hard preference pairs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preference.diagnostics import read_jsonl, sha256, text_features, write_json, write_jsonl
from src.training.sft_dataset import extract_caption, extract_image_path, extract_original_prompt

DIMENSIONS = ("humor", "grounding", "originality", "specificity", "naturalness", "overall")


def number(value: Any) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def base_candidate(row: dict[str, Any], caption: str, extra: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    merged = {**meta, **extra}
    item: dict[str, Any] = {
        "image_id": str(row.get("image_id") or Path(str(extract_image_path(row) or row.get("image", "unknown"))).stem),
        "image_path": str(extract_image_path(row) or row.get("image") or row.get("image_path") or ""),
        "prompt": str(row.get("prompt") or extract_original_prompt(row) or ""),
        "caption": " ".join(str(caption).split()),
        "source": str(merged.get("source") or row.get("source") or "unknown"),
    }
    item["score"] = number(merged.get("score", merged.get("overall")))
    aliases = {"grounding": ("grounding", "image_specific"), "specificity": ("specificity", "image_specific")}
    for dimension in DIMENSIONS:
        keys = aliases.get(dimension, (dimension,))
        item[dimension] = next((number(merged.get(key)) for key in keys if number(merged.get(key)) is not None), None)
    if item["humor"] is None:
        item["humor"] = item["score"]
    if item["overall"] is None:
        item["overall"] = item["score"]
    item["features"] = text_features(item["caption"])
    return item


def flatten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        judged = row.get("judged_candidates")
        if isinstance(judged, list):
            for item in judged:
                caption = item.get("candidate") or item.get("caption")
                if caption:
                    candidates.append(base_candidate(row, str(caption), item))
            continue
        nested = row.get("candidates")
        if isinstance(nested, list) and nested and isinstance(nested[0], dict):
            for item in nested:
                caption = item.get("candidate") or item.get("caption")
                if caption:
                    candidates.append(base_candidate(row, str(caption), item))
            continue
        caption = row.get("caption") or extract_caption(row)
        if caption:
            candidates.append(base_candidate(row, str(caption), row))
    return [item for item in candidates if item["caption"] and item["image_path"]]


def close_length(a: dict[str, Any], b: dict[str, Any], maximum: float) -> bool:
    left, right = int(a["features"]["chars"]), int(b["features"]["chars"])
    return abs(left - right) / max(left, right, 1) <= maximum


def style_matched(a: dict[str, Any], b: dict[str, Any]) -> bool:
    features = ("emoji_count", "pov", "bro", "meanwhile", "internet_slang")
    return all(bool(a["features"][key]) == bool(b["features"][key]) for key in features)


def dim(item: dict[str, Any], name: str) -> float | None:
    return number(item.get(name))


def classify(chosen: dict[str, Any], rejected: dict[str, Any], args: Any) -> str | None:
    hc, hr = dim(chosen, "humor"), dim(rejected, "humor")
    gc, gr = dim(chosen, "grounding"), dim(rejected, "grounding")
    oc, or_ = dim(chosen, "originality"), dim(rejected, "originality")
    sc, sr = dim(chosen, "specificity"), dim(rejected, "specificity")
    nc, nr = dim(chosen, "naturalness"), dim(rejected, "naturalness")
    if nc is not None and nc < args.min_naturalness or nr is not None and nr < args.min_naturalness:
        return None
    if hc is not None and hr is not None and gc is not None and gr is not None:
        if hc >= args.high_humor and hr <= args.low_humor and min(gc, gr) >= args.min_grounding:
            return "H1"
    if hc is not None and hr is not None and gc is not None and gr is not None:
        if min(hc, hr) >= args.weak_humor and gc - gr >= args.grounding_margin:
            return "H3"
    rejected_generic = any(bool(rejected["features"][key]) for key in ("pov", "bro", "meanwhile", "internet_slang"))
    if hc is not None and hr is not None and sc is not None and sr is not None:
        if min(hc, hr) >= args.weak_humor and sc - sr >= args.specificity_margin and rejected_generic:
            return "H4"
    primary_chosen = chosen["score"] if chosen["score"] is not None else hc
    primary_rejected = rejected["score"] if rejected["score"] is not None else hr
    if primary_chosen is not None and primary_rejected is not None:
        originality_ok = oc is None or or_ is None or oc >= or_
        if primary_chosen - primary_rejected >= args.min_score_margin and originality_ok:
            return "H2"
    return None


def make_pair(chosen: dict[str, Any], rejected: dict[str, Any], pair_type: str) -> dict[str, Any]:
    chosen_score = chosen["score"] if chosen["score"] is not None else chosen["humor"]
    rejected_score = rejected["score"] if rejected["score"] is not None else rejected["humor"]
    return {
        "image_id": chosen["image_id"],
        "image_path": chosen["image_path"],
        "image": chosen["image_path"],
        "prompt": chosen["prompt"] or rejected["prompt"],
        "chosen": chosen["caption"],
        "rejected": rejected["caption"],
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "score_margin": None if chosen_score is None or rejected_score is None else chosen_score - rejected_score,
        "pair_type": pair_type,
        "judge_dimensions": {
            "chosen": {name: chosen[name] for name in DIMENSIONS},
            "rejected": {name: rejected[name] for name in DIMENSIONS},
        },
        "source": {"chosen": chosen["source"], "rejected": rejected["source"]},
        "matching": {
            "chosen_features": chosen["features"],
            "rejected_features": rejected["features"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-images-jsonl", type=Path, default=None)
    parser.add_argument("--max-pairs-per-image", type=int, default=8)
    parser.add_argument("--max-relative-length-difference", type=float, default=0.35)
    parser.add_argument("--min-score-margin", type=float, default=0.35)
    parser.add_argument("--high-humor", type=float, default=4.0)
    parser.add_argument("--weak-humor", type=float, default=3.0)
    parser.add_argument("--low-humor", type=float, default=2.0)
    parser.add_argument("--min-grounding", type=float, default=4.0)
    parser.add_argument("--grounding-margin", type=float, default=2.0)
    parser.add_argument("--specificity-margin", type=float, default=2.0)
    parser.add_argument("--min-naturalness", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    candidates = flatten(read_jsonl(args.input_jsonl))
    excluded: set[str] = set()
    if args.exclude_images_jsonl is not None:
        excluded = {
            str(row.get("image_id") or Path(str(extract_image_path(row) or row.get("image", ""))).stem)
            for row in read_jsonl(args.exclude_images_jsonl)
        }
        candidates = [item for item in candidates if item["image_id"] not in excluded]
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_image[item["image_id"]].append(item)

    rng = random.Random(args.seed)
    pairs: list[dict[str, Any]] = []
    available_types: set[str] = set()
    for image_id in sorted(by_image):
        items = by_image[image_id]
        proposed = []
        for chosen in items:
            for rejected in items:
                if chosen is rejected or chosen["caption"].lower() == rejected["caption"].lower():
                    continue
                if not close_length(chosen, rejected, args.max_relative_length_difference):
                    continue
                pair_type = classify(chosen, rejected, args)
                if pair_type is None:
                    continue
                if pair_type != "H4" and not style_matched(chosen, rejected):
                    continue
                pair = make_pair(chosen, rejected, pair_type)
                proposed.append(pair)
                available_types.add(pair_type)
        proposed.sort(key=lambda row: (-(row["score_margin"] or 0.0), row["chosen"], row["rejected"]))
        if len(proposed) > args.max_pairs_per_image:
            cutoff = proposed[: args.max_pairs_per_image * 3]
            rng.shuffle(cutoff)
            proposed = cutoff[: args.max_pairs_per_image]
        pairs.extend(proposed)

    counts = {pair_type: sum(row["pair_type"] == pair_type for row in pairs) for pair_type in ("H1", "H2", "H3", "H4")}
    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": sha256(args.input_jsonl),
        "excluded_image_file": None if args.exclude_images_jsonl is None else str(args.exclude_images_jsonl),
        "excluded_images": len(excluded),
        "candidate_count": len(candidates),
        "images_with_candidates": len(by_image),
        "pair_count": len(pairs),
        "pairs_by_type": counts,
        "available_pair_types": sorted(available_types),
        "seed": args.seed,
        "thresholds": {
            key: getattr(args, key)
            for key in (
                "max_pairs_per_image", "max_relative_length_difference", "min_score_margin",
                "high_humor", "weak_humor", "low_humor", "min_grounding",
                "grounding_margin", "specificity_margin", "min_naturalness",
            )
        },
        "scientific_guard": "A missing pair type means required annotations were unavailable; it is not treated as zero prevalence.",
    }
    write_jsonl(args.output_jsonl, pairs)
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
