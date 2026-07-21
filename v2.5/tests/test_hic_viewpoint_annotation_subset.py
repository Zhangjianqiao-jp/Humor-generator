from __future__ import annotations

from pathlib import Path

from scripts.prepare_hic_viewpoint_annotation_subset import prepare_annotation_subset
from src.analysis.hic_region_annotations import CORE_VIEWPOINTS
from src.utils.io import read_jsonl, write_jsonl


def _row(row_key: str, primary_viewpoint: str, *, failed: bool = False, parse_error: bool = False) -> dict:
    row = {
        "row_key": row_key,
        "image_id": f"image-{row_key}",
        "gold_caption": f"caption {row_key}",
        "failed": failed,
        "analysis": {
            "primary_viewpoint": primary_viewpoint,
            "required_viewpoints": [primary_viewpoint],
        },
    }
    if parse_error:
        row["parse_error"] = "invalid json"
    return row


def _prepare(tmp_path: Path, rows: list[dict], *, per_viewpoint: int = 2, seed: int = 123) -> tuple[list[dict], dict]:
    input_jsonl = tmp_path / "input.jsonl"
    output_jsonl = tmp_path / "subset.jsonl"
    summary_json = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"
    write_jsonl(input_jsonl, rows)

    summary = prepare_annotation_subset(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        summary_md=summary_md,
        per_viewpoint=per_viewpoint,
        seed=seed,
    )

    return read_jsonl(output_jsonl), summary


def test_synthetic_rows_with_three_viewpoint_classes_select_at_most_per_class(tmp_path: Path) -> None:
    rows = [
        *[_row(f"face-{index}", "face_expression_crop") for index in range(4)],
        *[_row(f"text-{index}", "text_region_crop") for index in range(3)],
        _row("scale-0", "scale_reference_crop"),
    ]

    selected, summary = _prepare(tmp_path, rows, per_viewpoint=2)

    counts = summary["selected_counts_by_viewpoint"]
    assert counts["face_expression_crop"] == 2
    assert counts["text_region_crop"] == 2
    assert counts["scale_reference_crop"] == 1
    assert all(count <= 2 for count in counts.values())
    assert {row["row_key"] for row in selected if row["analysis"]["primary_viewpoint"] == "scale_reference_crop"} == {
        "scale-0"
    }


def test_failed_rows_parse_errors_and_invalid_analysis_are_skipped(tmp_path: Path) -> None:
    rows = [
        _row("keep", "object_crop"),
        _row("failed", "object_crop", failed=True),
        _row("parse-error", "object_crop", parse_error=True),
        {"row_key": "bad-analysis", "failed": False, "analysis": "not a dict"},
        _row("unknown-viewpoint", "not_a_core_viewpoint"),
    ]

    selected, summary = _prepare(tmp_path, rows, per_viewpoint=5)

    assert [row["row_key"] for row in selected] == ["keep"]
    assert summary["eligible_rows"] == 1
    assert summary["skipped_rows"] == 4
    assert summary["selected_counts_by_viewpoint"]["object_crop"] == 1


def test_foreground_background_view_maps_to_relation_crop(tmp_path: Path) -> None:
    selected, summary = _prepare(tmp_path, [_row("foreground", "foreground_background_view")], per_viewpoint=2)

    assert selected[0]["analysis"]["primary_viewpoint"] == "relation_crop"
    assert summary["selected_counts_by_viewpoint"]["relation_crop"] == 1


def test_repeated_runs_with_same_seed_produce_identical_row_key_order(tmp_path: Path) -> None:
    rows = []
    for viewpoint in CORE_VIEWPOINTS[:3]:
        rows.extend(_row(f"{viewpoint}-{index}", viewpoint) for index in range(6))

    first, _ = _prepare(tmp_path / "first", rows, per_viewpoint=3, seed=20260710)
    second, _ = _prepare(tmp_path / "second", rows, per_viewpoint=3, seed=20260710)

    assert [row["row_key"] for row in first] == [row["row_key"] for row in second]
