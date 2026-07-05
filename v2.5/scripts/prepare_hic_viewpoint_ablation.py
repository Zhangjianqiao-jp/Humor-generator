#!/usr/bin/env python
from __future__ import annotations

import json
import random
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_jsonl

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def confidence_ok(value: Any, minimum: str) -> bool:
    return CONFIDENCE_ORDER.get(str(value or "low"), 0) >= CONFIDENCE_ORDER[minimum]


def load_success_rows(
    input_jsonl: Path,
    min_confidence: str,
    drop_parse_errors: bool,
    exclude_external_knowledge: bool,
) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for row in read_jsonl(input_jsonl):
        if row.get("failed"):
            continue
        if drop_parse_errors and row.get("parse_error"):
            continue
        analysis = row.get("analysis")
        if not isinstance(analysis, dict):
            continue
        if not confidence_ok(analysis.get("confidence"), min_confidence):
            continue
        if exclude_external_knowledge and analysis.get("needs_external_knowledge"):
            continue
        row_key = str(row.get("row_key") or f"{row.get('image_id')}::{row.get('gold_caption')}")
        if row_key in seen:
            continue
        seen.add(row_key)
        rows.append(
            {
                "image": row.get("image"),
                "image_id": row.get("image_id"),
                "gold_caption": row.get("gold_caption"),
                "score": row.get("score"),
                "row_key": row_key,
                "source_index": row.get("source_index"),
                "analysis": analysis,
                "prompt_version": row.get("prompt_version"),
            }
        )
    return rows


def balanced_sample(rows: list[dict[str, Any]], limit: int | None, group_field: str, seed: int) -> list[dict[str, Any]]:
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        analysis = row.get("analysis") or {}
        key = str(analysis.get(group_field) or "unknown")
        groups[key].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    selected = []
    group_names = sorted(groups)
    while len(selected) < limit:
        added = False
        for name in group_names:
            if groups[name]:
                selected.append(groups[name].pop())
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    selected.sort(key=lambda row: int(row.get("source_index") or 0))
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = Counter()
    primary_counts = Counter()
    required_counts = Counter()
    confidence_counts = Counter()
    external_count = 0
    for row in rows:
        analysis = row.get("analysis") or {}
        type_counts[str(analysis.get("humor_type") or "unknown")] += 1
        primary_counts[str(analysis.get("primary_viewpoint") or "unknown")] += 1
        confidence_counts[str(analysis.get("confidence") or "unknown")] += 1
        if analysis.get("needs_external_knowledge"):
            external_count += 1
        for viewpoint in analysis.get("required_viewpoints") or []:
            required_counts[str(viewpoint)] += 1
    return {
        "rows": len(rows),
        "humor_type_counts": dict(type_counts.most_common()),
        "primary_viewpoint_counts": dict(primary_counts.most_common()),
        "required_viewpoint_counts": dict(required_counts.most_common()),
        "confidence_counts": dict(confidence_counts.most_common()),
        "external_knowledge_rows": external_count,
    }


def write_report(path: Path, summary: dict[str, Any], output_jsonl: Path) -> None:
    lines = [
        "# HIC Viewpoint Ablation Subset",
        "",
        f"Output JSONL: `{output_jsonl}`",
        f"Rows: {summary['rows']}",
        f"External knowledge rows: {summary['external_knowledge_rows']}",
        "",
        "## Humor Types",
        "",
        "| humor_type | count |",
        "|---|---:|",
    ]
    for key, count in summary["humor_type_counts"].items():
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Primary Viewpoints", "", "| viewpoint | count |", "|---|---:|"])
    for key, count in summary["primary_viewpoint_counts"].items():
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Required Viewpoints", "", "| viewpoint | count |", "|---|---:|"])
    for key, count in summary["required_viewpoint_counts"].items():
        lines.append(f"| {key} | {count} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(description="Prepare a clean balanced HIC viewpoint subset for prompt ablation.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("outputs/analysis/hic_humor_viewpoints_pairs_1000_minview.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/analysis/hic_viewpoint_ablation_120.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/hic_viewpoint_ablation_120_summary.json"))
    parser.add_argument("--report-md", type=Path, default=Path("outputs/analysis/hic_viewpoint_ablation_120.md"))
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=250704)
    parser.add_argument("--min-confidence", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--group-field", choices=("humor_type", "primary_viewpoint"), default="humor_type")
    parser.add_argument("--keep-parse-errors", action="store_true")
    parser.add_argument("--exclude-external-knowledge", action="store_true")
    args = parser.parse_args()

    rows = load_success_rows(
        input_jsonl=args.input_jsonl,
        min_confidence=args.min_confidence,
        drop_parse_errors=not args.keep_parse_errors,
        exclude_external_knowledge=args.exclude_external_knowledge,
    )
    selected = balanced_sample(rows, limit=args.limit, group_field=args.group_field, seed=args.seed)
    write_jsonl(args.output_jsonl, selected)
    summary = summarize(selected)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.report_md, summary, args.output_jsonl)
    print(f"[hic-ablation] loaded_success_rows={len(rows)} selected_rows={len(selected)}")
    print(f"[hic-ablation] wrote {args.output_jsonl}")
    print(f"[hic-ablation] wrote {args.summary_json}")
    print(f"[hic-ablation] wrote {args.report_md}")


if __name__ == "__main__":
    main()
