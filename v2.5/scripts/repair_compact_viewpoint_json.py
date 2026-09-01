#!/usr/bin/env python
"""Repair narrowly defined JSON punctuation errors while preserving raw labels."""

from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_sft_generations import parse_compact_viewpoint
from src.utils.io import read_jsonl, write_jsonl


MISSING_ANCHOR_COMMA = re.compile(r'("evidence"\s*:\s*"(?:[^"\\]|\\.)*")\s*\n(\s*"role"\s*:)', re.DOTALL)


def repair_candidate(text: str) -> tuple[str, list[str]]:
    try:
        value = parse_compact_viewpoint(text)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")), []
    except ValueError:
        pass
    repaired, count = MISSING_ANCHOR_COMMA.subn(r"\1,\n\2", text)
    if count == 0:
        raise ValueError("Invalid compact viewpoint JSON does not match the allowed repair pattern.")
    value = parse_compact_viewpoint(repaired)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")), [f"inserted_missing_evidence_role_comma:{count}"]


def repair_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    repaired_ids: list[str] = []
    for index, row in enumerate(rows):
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError(f"Row {index} must contain exactly one candidate.")
        raw = str(candidates[0])
        candidate, repairs = repair_candidate(raw)
        copied = json.loads(json.dumps(row))
        copied["candidates"] = [candidate]
        if repairs:
            copied["raw_candidate_before_json_repair"] = raw
            copied["json_repairs"] = repairs
            repaired_ids.append(str(row.get("image_id") or index))
        outputs.append(copied)
    return outputs, {
        "rows": len(outputs),
        "repaired_rows": len(repaired_ids),
        "repaired_image_ids": repaired_ids,
        "allowed_repair": "missing comma between anchor evidence and role only",
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    outputs, report = repair_rows(read_jsonl(args.input_jsonl))
    write_jsonl(args.output_jsonl, outputs)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
