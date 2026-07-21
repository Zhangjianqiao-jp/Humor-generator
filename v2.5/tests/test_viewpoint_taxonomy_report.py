from scripts.report_hic_viewpoint_taxonomy import summarize_taxonomy
from scripts.analyze_hic_humor_viewpoints import select_input_rows


def _row(humor_type: str, primary: str, required: list[str] | None = None) -> dict:
    return {
        "analysis": {
            "humor_type": humor_type,
            "primary_viewpoint": primary,
            "required_viewpoints": required or [primary],
        }
    }


def test_taxonomy_counts_core_coverage_and_foreground_recommendation() -> None:
    rows = [
        _row("expression_context_contrast", "face_expression_crop"),
        _row("expression_context_contrast", "face_expression_crop"),
        _row("expression_context_contrast", "face_expression_crop", ["face_expression_crop", "context_scene_view"]),
        _row("text_image_contrast", "text_region_crop"),
        _row("text_image_contrast", "text_region_crop"),
        _row("foreground_background_misread", "relation_crop", ["relation_crop", "foreground_background_view"]),
    ]

    summary = summarize_taxonomy(rows, min_type_count=1, coverage_target=0.95)

    assert summary["rows"] == 6
    assert summary["primary_distinct_count"] == 3
    assert summary["core_primary_coverage"] == 1.0
    assert summary["required_distinct_count"] == 5
    assert summary["foreground_background_view"]["required_count"] == 1
    assert summary["foreground_background_view"]["recommendation"] == "merge_into_relation_or_context_until_more_evidence"


def test_taxonomy_detects_stability_and_multiview_types() -> None:
    rows = [
        _row("scale_contrast", "relation_crop", ["relation_crop", "scale_reference_crop"]),
        _row("scale_contrast", "relation_crop", ["relation_crop", "scale_reference_crop"]),
        _row("scale_contrast", "full_image"),
        _row("object_misuse", "object_crop"),
        _row("object_misuse", "object_crop"),
        _row("object_misuse", "object_crop"),
    ]

    summary = summarize_taxonomy(
        rows,
        min_type_count=1,
        stable_top1=0.70,
        stable_top2=0.85,
        multiview_threshold=0.50,
    )

    assert "object_misuse" in summary["stable_single_viewpoint_types"]
    assert "scale_contrast" in summary["stable_top2_types"]
    assert summary["multi_view"]["rows"] == 2
    assert "scale_contrast" in summary["multi_view"]["multi_view_heavy_types"]


def test_select_input_rows_can_use_reproducible_random_sample() -> None:
    rows = [{"id": index} for index in range(20)]

    sample_a = select_input_rows(rows, limit=5, sample_seed=7)
    sample_b = select_input_rows(rows, limit=5, sample_seed=7)
    first_five = select_input_rows(rows, limit=5, sample_seed=None)

    assert sample_a == sample_b
    assert len(sample_a) == 5
    assert sample_a != first_five
