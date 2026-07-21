#!/usr/bin/env python
from __future__ import annotations

import json
import math
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CORE_PRIMARY_VIEWPOINTS = (
    "face_expression_crop",
    "relation_crop",
    "context_scene_view",
    "text_region_crop",
    "pose_action_view",
    "object_crop",
    "full_image",
    "scale_reference_crop",
)

ALL_KNOWN_VIEWPOINTS = CORE_PRIMARY_VIEWPOINTS + ("foreground_background_view",)


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * math.log2(p)
    return value


def read_success_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("failed"):
                continue
            analysis = row.get("analysis")
            if isinstance(analysis, dict):
                rows.append(row)
    return rows


def clean_viewpoints(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def cumulative_coverage(counter: Counter[str], total: int) -> list[dict[str, Any]]:
    running = 0
    rows = []
    for index, (name, count) in enumerate(counter.most_common(), start=1):
        running += count
        rows.append(
            {
                "rank": index,
                "viewpoint": name,
                "count": count,
                "cumulative_share": running / total if total else 0.0,
            }
        )
    return rows


def stability_status(top1_share: float, top2_coverage: float, stable_top1: float, stable_top2: float) -> str:
    if top1_share >= stable_top1:
        return "stable_single_viewpoint"
    if top2_coverage >= stable_top2:
        return "stable_top2"
    return "mixed_or_unstable"


def summarize_taxonomy(
    rows: list[dict[str, Any]],
    coverage_target: float = 0.95,
    min_type_count: int = 30,
    stable_top1: float = 0.70,
    stable_top2: float = 0.85,
    foreground_min_required_share: float = 0.02,
    foreground_min_primary_count: int = 25,
    multiview_threshold: float = 0.25,
) -> dict[str, Any]:
    total = len(rows)
    primary_counts: Counter[str] = Counter()
    required_counts: Counter[str] = Counter()
    primary_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    required_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    combo_counts: Counter[tuple[str, ...]] = Counter()
    combo_by_type: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    multiview_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    foreground_types: Counter[str] = Counter()

    for row in rows:
        analysis = row.get("analysis") or {}
        humor_type = str(analysis.get("humor_type") or "unknown")
        primary = str(analysis.get("primary_viewpoint") or "unknown")
        required = clean_viewpoints(analysis.get("required_viewpoints")) or [primary]

        primary_counts[primary] += 1
        primary_by_type[humor_type][primary] += 1
        for viewpoint in required:
            required_counts[viewpoint] += 1
            required_by_type[humor_type][viewpoint] += 1

        combo = tuple(required)
        combo_counts[combo] += 1
        combo_by_type[humor_type][combo] += 1
        multiview = len(required) > 1
        multiview_by_type[humor_type]["multi" if multiview else "single"] += 1
        if primary == "foreground_background_view" or "foreground_background_view" in required:
            foreground_types[humor_type] += 1

    top_coverage = cumulative_coverage(primary_counts, total)
    viewpoints_needed_for_target = None
    for item in top_coverage:
        if item["cumulative_share"] >= coverage_target:
            viewpoints_needed_for_target = item["rank"]
            break

    core_primary_count = sum(primary_counts[v] for v in CORE_PRIMARY_VIEWPOINTS)
    known_primary_count = sum(primary_counts[v] for v in ALL_KNOWN_VIEWPOINTS)
    foreground_required = required_counts["foreground_background_view"]
    foreground_primary = primary_counts["foreground_background_view"]
    foreground_required_share = foreground_required / total if total else 0.0
    foreground_independent = (
        foreground_required_share >= foreground_min_required_share
        and foreground_primary >= foreground_min_primary_count
    )

    type_stability = {}
    stable_types = []
    top2_types = []
    unstable_types = []
    for humor_type, counter in sorted(primary_by_type.items()):
        count = sum(counter.values())
        if count < min_type_count:
            continue
        top = counter.most_common()
        top1_share = top[0][1] / count if top else 0.0
        top2_coverage = sum(value for _, value in top[:2]) / count if top else 0.0
        status = stability_status(top1_share, top2_coverage, stable_top1, stable_top2)
        if status == "stable_single_viewpoint":
            stable_types.append(humor_type)
        elif status == "stable_top2":
            top2_types.append(humor_type)
        else:
            unstable_types.append(humor_type)
        type_stability[humor_type] = {
            "count": count,
            "primary_viewpoints": dict(top),
            "required_viewpoints": dict(required_by_type[humor_type].most_common()),
            "top1_share": top1_share,
            "top2_coverage": top2_coverage,
            "entropy": entropy(counter),
            "status": status,
        }

    multiview_rows = sum(1 for row in rows if len(clean_viewpoints((row.get("analysis") or {}).get("required_viewpoints"))) > 1)
    multiview_by_humor_type = {}
    multiview_heavy_types = []
    for humor_type, counter in sorted(multiview_by_type.items()):
        count = counter["single"] + counter["multi"]
        share = counter["multi"] / count if count else 0.0
        if count >= min_type_count and share >= multiview_threshold:
            multiview_heavy_types.append(humor_type)
        multiview_by_humor_type[humor_type] = {
            "count": count,
            "multi_count": counter["multi"],
            "multi_share": share,
        }

    return {
        "rows": total,
        "coverage_target": coverage_target,
        "core_primary_viewpoints": list(CORE_PRIMARY_VIEWPOINTS),
        "all_known_viewpoints": list(ALL_KNOWN_VIEWPOINTS),
        "primary_viewpoint_counts": dict(primary_counts.most_common()),
        "required_viewpoint_counts": dict(required_counts.most_common()),
        "primary_distinct_count": len(primary_counts),
        "required_distinct_count": len(required_counts),
        "core_primary_coverage": core_primary_count / total if total else 0.0,
        "known_primary_coverage": known_primary_count / total if total else 0.0,
        "top_primary_cumulative_coverage": top_coverage,
        "viewpoints_needed_for_coverage_target": viewpoints_needed_for_target,
        "foreground_background_view": {
            "primary_count": foreground_primary,
            "required_count": foreground_required,
            "required_share": foreground_required_share,
            "humor_type_counts": dict(foreground_types.most_common()),
            "recommendation": "keep_as_independent_viewpoint"
            if foreground_independent
            else "merge_into_relation_or_context_until_more_evidence",
        },
        "type_stability": type_stability,
        "stable_single_viewpoint_types": stable_types,
        "stable_top2_types": top2_types,
        "mixed_or_unstable_types": unstable_types,
        "multi_view": {
            "rows": multiview_rows,
            "share": multiview_rows / total if total else 0.0,
            "top_required_viewpoint_combos": [
                {"combo": list(combo), "count": count} for combo, count in combo_counts.most_common(20)
            ],
            "by_humor_type": multiview_by_humor_type,
            "multi_view_heavy_types": multiview_heavy_types,
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# HIC Viewpoint Taxonomy Report",
        "",
        f"Rows analyzed: {summary['rows']}",
        f"Coverage target: {pct(summary['coverage_target'])}",
        "",
        "## Decision 1: How Many Primary Viewpoints?",
        "",
        f"- Distinct primary viewpoints observed: {summary['primary_distinct_count']}",
        f"- Core 8 primary coverage: {pct(summary['core_primary_coverage'])}",
        f"- Known 9 primary coverage: {pct(summary['known_primary_coverage'])}",
        f"- Viewpoints needed to reach target coverage: {summary['viewpoints_needed_for_coverage_target']}",
        "",
        "| viewpoint | count | cumulative coverage |",
        "|---|---:|---:|",
    ]
    for item in summary["top_primary_cumulative_coverage"]:
        lines.append(f"| {item['viewpoint']} | {item['count']} | {pct(item['cumulative_share'])} |")

    fg = summary["foreground_background_view"]
    lines.extend(
        [
            "",
            "## Decision 2: Is `foreground_background_view` Independent?",
            "",
            f"- Primary count: {fg['primary_count']}",
            f"- Required count: {fg['required_count']}",
            f"- Required share: {pct(fg['required_share'])}",
            f"- Recommendation: `{fg['recommendation']}`",
            "",
            "| humor_type | foreground count |",
            "|---|---:|",
        ]
    )
    for humor_type, count in fg["humor_type_counts"].items():
        lines.append(f"| {humor_type} | {count} |")

    lines.extend(
        [
            "",
            "## Decision 3: Is Viewpoint Stable Within Humor Type?",
            "",
            f"- Stable single-viewpoint types: {', '.join(summary['stable_single_viewpoint_types']) or 'none'}",
            f"- Stable top-2 types: {', '.join(summary['stable_top2_types']) or 'none'}",
            f"- Mixed/unstable types: {', '.join(summary['mixed_or_unstable_types']) or 'none'}",
            "",
            "| humor_type | count | top primary viewpoints | top1 | top2 | entropy | status |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for humor_type, info in summary["type_stability"].items():
        top = ", ".join(f"{name}:{count}" for name, count in list(info["primary_viewpoints"].items())[:3])
        lines.append(
            f"| {humor_type} | {info['count']} | {top} | "
            f"{pct(info['top1_share'])} | {pct(info['top2_coverage'])} | "
            f"{info['entropy']:.2f} | {info['status']} |"
        )

    mv = summary["multi_view"]
    lines.extend(
        [
            "",
            "## Decision 4: Which Types Need Multiple Viewpoints?",
            "",
            f"- Multi-view rows: {mv['rows']}",
            f"- Multi-view share: {pct(mv['share'])}",
            f"- Multi-view-heavy humor types: {', '.join(mv['multi_view_heavy_types']) or 'none'}",
            "",
            "| humor_type | rows | multi-view rows | multi-view share |",
            "|---|---:|---:|---:|",
        ]
    )
    for humor_type, info in mv["by_humor_type"].items():
        lines.append(f"| {humor_type} | {info['count']} | {info['multi_count']} | {pct(info['multi_share'])} |")

    lines.extend(["", "### Top Required Viewpoint Combos", "", "| combo | count |", "|---|---:|"])
    for item in mv["top_required_viewpoint_combos"]:
        lines.append(f"| {' + '.join(item['combo'])} | {item['count']} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(description="Report viewpoint taxonomy coverage and stability for HIC humor analysis.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--coverage-target", type=float, default=0.95)
    parser.add_argument("--min-type-count", type=int, default=30)
    parser.add_argument("--stable-top1", type=float, default=0.70)
    parser.add_argument("--stable-top2", type=float, default=0.85)
    parser.add_argument("--foreground-min-required-share", type=float, default=0.02)
    parser.add_argument("--foreground-min-primary-count", type=int, default=25)
    parser.add_argument("--multiview-threshold", type=float, default=0.25)
    args = parser.parse_args()

    rows = read_success_rows(args.input_jsonl)
    summary = summarize_taxonomy(
        rows,
        coverage_target=args.coverage_target,
        min_type_count=args.min_type_count,
        stable_top1=args.stable_top1,
        stable_top2=args.stable_top2,
        foreground_min_required_share=args.foreground_min_required_share,
        foreground_min_primary_count=args.foreground_min_primary_count,
        multiview_threshold=args.multiview_threshold,
    )
    write_json(args.summary_json, summary)
    write_report(args.report_md, summary)
    print(f"[taxonomy] rows={summary['rows']} report={args.report_md}")


if __name__ == "__main__":
    main()
