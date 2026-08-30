#!/usr/bin/env python
"""Apply explicit, reviewable semantic overrides to raw compact-viewpoint labels."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_sft_generations import parse_compact_viewpoint
from src.utils.io import read_jsonl, write_jsonl


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Override file must be a JSON object keyed by image_id.")
    for image_id, override in value.items():
        if not isinstance(override, dict) or set(override) != {"reason", "label"}:
            raise ValueError(f"{image_id}: override must contain exactly reason and label.")
        if not str(override["reason"]).strip():
            raise ValueError(f"{image_id}: override reason is empty.")
        parse_compact_viewpoint(json.dumps(override["label"], ensure_ascii=False))
    return value


def finalize_rows(
    rows: list[dict[str, Any]], overrides: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    outputs: list[dict[str, Any]] = []
    applied: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        image_id = str(row.get("image_id") or "").strip()
        if not image_id or image_id in seen:
            raise ValueError(f"Missing or duplicate image_id at row {index}: {image_id!r}")
        seen.add(image_id)
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError(f"{image_id}: expected exactly one candidate.")
        copied = json.loads(json.dumps(row))
        if image_id in overrides:
            override = overrides[image_id]
            copied["raw_candidate_before_semantic_override"] = str(candidates[0])
            copied["semantic_override_reason"] = str(override["reason"])
            label = parse_compact_viewpoint(
                json.dumps(override["label"], ensure_ascii=False)
            )
            applied.append(image_id)
        else:
            label = parse_compact_viewpoint(str(candidates[0]))
        copied["candidates"] = [
            json.dumps(label, ensure_ascii=False, separators=(",", ":"))
        ]
        outputs.append(copied)
    unused = sorted(set(overrides) - seen)
    if unused:
        raise ValueError(f"Overrides do not occur in this input split: {unused}")
    return outputs, applied


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--overrides-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    overrides = load_overrides(args.overrides_json)
    outputs, applied = finalize_rows(read_jsonl(args.input_jsonl), overrides)
    write_jsonl(args.output_jsonl, outputs)
    report = {
        "input": str(args.input_jsonl),
        "rows": len(outputs),
        "semantic_override_count": len(applied),
        "semantic_override_image_ids": applied,
        "overrides_file": str(args.overrides_json),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
