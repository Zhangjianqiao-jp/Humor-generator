#!/usr/bin/env python
"""Audit compact-viewpoint pseudo-labels before they may enter 7B SFT."""

from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_sft_generations import parse_compact_viewpoint
from src.utils.io import read_jsonl


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold())


def ngrams(text: str, size: int = 6) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def label_text(label: dict[str, Any]) -> str:
    values = [label["scene"], label["target"]]
    for anchor in label["anchors"]:
        values.extend([anchor["label"], anchor["evidence"], anchor["role"]])
    return " ".join(values)


def audit(
    rows: list[dict[str, Any]],
    caption_evidence: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    types: Counter[str] = Counter()
    primary_views: Counter[str] = Counter()
    seen: set[str] = set()
    leaks: list[dict[str, Any]] = []
    lengths: list[int] = []
    external = 0
    for index, row in enumerate(rows):
        image_id = str(row.get("image_id") or "").strip()
        if not image_id or image_id in seen:
            raise ValueError(f"Missing or duplicate image_id at row {index}: {image_id!r}")
        seen.add(image_id)
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError(f"{image_id}: expected exactly one label candidate.")
        label = parse_compact_viewpoint(str(candidates[0]))
        types[label["type"]] += 1
        primary_views[label["primary_view"]] += 1
        external += int(label["external_knowledge"])
        rendered = label_text(label)
        lengths.append(len(words(rendered)))
        label_ngrams = ngrams(rendered)
        caption_list = (
            caption_evidence.get(image_id) if caption_evidence is not None else None
        )
        if caption_list is None:
            caption_list = row.get("gold_captions")
        if not isinstance(caption_list, list):
            caption_list = str(row.get("gold_caption") or "").splitlines()
        matches: set[str] = set()
        for caption in caption_list:
            caption_tokens = words(str(caption))
            for gram in ngrams(str(caption)) & label_ngrams:
                matches.add(" ".join(gram))
            normalized_caption = " ".join(caption_tokens)
            if len(caption_tokens) >= 4 and normalized_caption in " ".join(words(rendered)):
                matches.add(normalized_caption)
        if matches:
            leaks.append({"image_id": image_id, "matching_phrases": sorted(matches)[:10]})
    return {
        "rows": len(rows),
        "unique_images": len(seen),
        "schema_valid": True,
        "type_counts": dict(types.most_common()),
        "primary_view_counts": dict(primary_views.most_common()),
        "external_knowledge_count": external,
        "label_word_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "possible_caption_phrase_leaks": leaks,
        "possible_caption_phrase_leak_count": len(leaks),
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--caption-evidence-jsonl",
        type=Path,
        help="Optional JSONL keyed by image_id with a captions list.",
    )
    args = parser.parse_args()
    caption_evidence = None
    if args.caption_evidence_jsonl is not None:
        caption_evidence = {}
        for row in read_jsonl(args.caption_evidence_jsonl):
            image_id = str(row.get("image_id") or "").strip()
            captions = row.get("captions")
            if not image_id or not isinstance(captions, list):
                raise ValueError(f"Invalid caption evidence row: {image_id!r}")
            caption_evidence[image_id] = [
                str(item.get("caption") if isinstance(item, dict) else item)
                for item in captions
            ]
    report = audit(read_jsonl(args.input_jsonl), caption_evidence=caption_evidence)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
