#!/usr/bin/env python
from __future__ import annotations

import json
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.hic_region_annotations import CORE_VIEWPOINTS, clean_region_text
from src.utils.io import read_jsonl, write_jsonl

DEFAULT_INPUT_JSONL = Path("outputs/analysis/hic_humor_viewpoints_pairs_10000_random_minview.jsonl")
DEFAULT_OUTPUT_JSONL = Path("outputs/annotations/hic_region_annotation_subset_800.jsonl")
DEFAULT_SUMMARY_JSON = Path("outputs/annotations/hic_region_annotation_subset_800_summary.json")
DEFAULT_SUMMARY_MD = Path("outputs/annotations/hic_region_annotation_subset_800_summary.md")
DEFAULT_PER_VIEWPOINT = 100
DEFAULT_SEED = 20260710

_VIEWPOINT_ALIASES = {"foreground_background_view": "relation_crop"}


def normalize_primary_viewpoint(value: Any) -> str | None:
    viewpoint = clean_region_text(value, max_chars=80)
    viewpoint = _VIEWPOINT_ALIASES.get(viewpoint, viewpoint)
    if viewpoint in CORE_VIEWPOINTS:
        return viewpoint
    return None


def eligible_rows(input_jsonl: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for row in read_jsonl(input_jsonl):
        if row.get("failed") or row.get("parse_error"):
            skipped += 1
            continue

        analysis = row.get("analysis")
        if not isinstance(analysis, dict):
            skipped += 1
            continue

        primary_viewpoint = normalize_primary_viewpoint(analysis.get("primary_viewpoint"))
        if primary_viewpoint is None:
            skipped += 1
            continue

        prepared = dict(row)
        prepared_analysis = dict(analysis)
        prepared_analysis["primary_viewpoint"] = primary_viewpoint
        prepared["analysis"] = prepared_analysis
        rows.append(prepared)
    return rows, skipped


def select_balanced_rows(rows: list[dict[str, Any]], *, per_viewpoint: int, seed: int) -> list[dict[str, Any]]:
    if per_viewpoint < 0:
        raise ValueError("per_viewpoint must be non-negative")

    grouped: dict[str, list[dict[str, Any]]] = {viewpoint: [] for viewpoint in CORE_VIEWPOINTS}
    for row in rows:
        analysis = row.get("analysis")
        if not isinstance(analysis, dict):
            continue
        primary_viewpoint = normalize_primary_viewpoint(analysis.get("primary_viewpoint"))
        if primary_viewpoint in grouped:
            grouped[primary_viewpoint].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for viewpoint in CORE_VIEWPOINTS:
        group_rows = list(grouped[viewpoint])
        rng.shuffle(group_rows)
        selected.extend(group_rows[:per_viewpoint])
    return selected


def summarize(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    rows_read: int,
    eligible_count: int,
    skipped_count: int,
    selected: list[dict[str, Any]],
    per_viewpoint: int,
    seed: int,
) -> dict[str, Any]:
    counts = dict.fromkeys(CORE_VIEWPOINTS, 0)
    for row in selected:
        analysis = row.get("analysis") or {}
        viewpoint = normalize_primary_viewpoint(analysis.get("primary_viewpoint"))
        if viewpoint is not None:
            counts[viewpoint] += 1

    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "per_viewpoint": per_viewpoint,
        "seed": seed,
        "viewpoint_set": list(CORE_VIEWPOINTS),
        "rows_read": rows_read,
        "eligible_rows": eligible_count,
        "skipped_rows": skipped_count,
        "selected_rows": len(selected),
        "selected_counts_by_viewpoint": counts,
    }


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# HIC Region Annotation Subset",
        "",
        f"Input JSONL: `{summary['input_jsonl']}`",
        f"Output JSONL: `{summary['output_jsonl']}`",
        f"Rows read: {summary['rows_read']}",
        f"Eligible rows: {summary['eligible_rows']}",
        f"Skipped rows: {summary['skipped_rows']}",
        f"Selected rows: {summary['selected_rows']}",
        f"Per viewpoint: {summary['per_viewpoint']}",
        f"Seed: {summary['seed']}",
        "",
        "## Selected Counts By Viewpoint",
        "",
        "| viewpoint | count |",
        "|---|---:|",
    ]
    for viewpoint, count in summary["selected_counts_by_viewpoint"].items():
        lines.append(f"| {viewpoint} | {count} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_annotation_subset(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    summary_json: Path,
    summary_md: Path,
    per_viewpoint: int = DEFAULT_PER_VIEWPOINT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    all_rows = read_jsonl(input_jsonl)
    source_rows, skipped = eligible_rows(input_jsonl)
    selected = select_balanced_rows(source_rows, per_viewpoint=per_viewpoint, seed=seed)
    summary = summarize(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        rows_read=len(all_rows),
        eligible_count=len(source_rows),
        skipped_count=skipped,
        selected=selected,
        per_viewpoint=per_viewpoint,
        seed=seed,
    )

    write_jsonl(output_jsonl, selected)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_summary(summary_md, summary)
    return summary


def main() -> None:
    parser = ArgumentParser(description="Prepare a balanced HIC region annotation subset by primary viewpoint.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--per-viewpoint", type=int, default=DEFAULT_PER_VIEWPOINT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    summary = prepare_annotation_subset(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        summary_md=args.summary_md,
        per_viewpoint=args.per_viewpoint,
        seed=args.seed,
    )
    print(
        "[hic-annotation-subset] "
        f"eligible_rows={summary['eligible_rows']} selected_rows={summary['selected_rows']}"
    )
    print(f"[hic-annotation-subset] wrote {args.output_jsonl}")
    print(f"[hic-annotation-subset] wrote {args.summary_json}")
    print(f"[hic-annotation-subset] wrote {args.summary_md}")


if __name__ == "__main__":
    main()
