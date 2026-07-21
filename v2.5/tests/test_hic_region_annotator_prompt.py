from __future__ import annotations

import pytest

from scripts.annotate_hic_humor_regions import (
    build_failure_output_row,
    build_region_annotation_meta,
    build_region_prompt,
    build_success_output_row,
    parse_region_annotation,
)
from src.analysis.hic_region_annotations import ANNOTATION_VERSION, CORE_VIEWPOINTS


def _row() -> dict:
    return {
        "row_key": "image-1::caption",
        "image": "/tmp/image.jpg",
        "image_id": "image-1",
        "gold_caption": "do not use as final caption",
        "analysis": {
            "humor_type": "role_mismatch",
            "humor_point": "the dog is treated like the driver",
            "visual_anchors": [
                {
                    "id": "a1",
                    "label": "dog at steering wheel",
                    "role": "unexpected driver",
                    "evidence": "dog is positioned behind the wheel",
                },
                {
                    "id": "a2",
                    "label": "steering wheel",
                    "role": "driver context",
                    "evidence": "wheel is in front of the dog",
                },
            ],
            "required_viewpoints": ["relation_crop", "context_scene_view"],
            "primary_viewpoint": "relation_crop",
            "confidence": "high",
            "uncertainty": "wheel edge is partly occluded",
        },
    }


def test_prompt_builder_includes_all_fixed_viewpoint_names() -> None:
    prompt = build_region_prompt(_row())

    for viewpoint in CORE_VIEWPOINTS:
        assert viewpoint in prompt


def test_prompt_builder_includes_humor_point_and_visual_anchor_labels() -> None:
    prompt = build_region_prompt(_row())

    assert "the dog is treated like the driver" in prompt
    assert "dog at steering wheel" in prompt
    assert "steering wheel" in prompt


def test_prompt_builder_does_not_ask_for_a_final_caption() -> None:
    prompt = build_region_prompt(_row()).lower()

    assert "no final humorous caption" in prompt
    assert "write a final caption" not in prompt
    assert "write a caption" not in prompt


def test_parser_accepts_valid_minimal_annotation_and_normalizes_it() -> None:
    annotation = parse_region_annotation(
        {
            "anchors": [
                {
                    "id": "driver",
                    "label": "dog at steering wheel",
                    "role": "humor target",
                    "source_anchor_id": "a1",
                    "viewpoint": "relation_crop",
                    "region": {
                        "kind": "bbox",
                        "bbox_xyxy_norm": ["0.10", 0.2, 0.8, "0.9"],
                        "point_xy_norm": [0.45, "0.55"],
                        "confidence": "high",
                        "evidence": "dog and wheel are visible",
                    },
                }
            ],
            "relations": [
                {
                    "subject": "driver",
                    "predicate": "positioned behind",
                    "object": "steering wheel",
                    "confidence": "medium",
                }
            ],
            "annotation_confidence": "medium",
        },
        primary_viewpoint="relation_crop",
        required_viewpoints=["relation_crop", "context_scene_view"],
    )

    assert annotation["annotation_version"] == ANNOTATION_VERSION
    assert annotation["viewpoint_set"] == list(CORE_VIEWPOINTS)
    assert annotation["primary_viewpoint"] == "relation_crop"
    assert annotation["required_viewpoints"] == ["relation_crop", "context_scene_view"]
    assert annotation["anchors"][0]["region"]["bbox_xyxy_norm"] == [0.1, 0.2, 0.8, 0.9]
    assert annotation["anchors"][0]["region"]["point_xy_norm"] == [0.45, 0.55]
    assert annotation["relations"][0]["confidence"] == "medium"


def test_parser_rejects_invalid_bbox_kind_coordinates() -> None:
    with pytest.raises(ValueError, match="invalid region coordinates"):
        parse_region_annotation(
            {
                "anchors": [
                    {
                        "id": "bad",
                        "label": "bad box",
                        "viewpoint": "relation_crop",
                        "region": {
                            "kind": "bbox",
                            "bbox_xyxy_norm": [0.9, 0.1, 0.2, 0.3],
                            "point_xy_norm": [0.5, 0.5],
                        },
                    }
                ]
            },
            primary_viewpoint="relation_crop",
            required_viewpoints=["relation_crop"],
        )


def test_parser_accepts_valid_bbox_with_missing_point_and_fills_center() -> None:
    annotation = parse_region_annotation(
        {
            "anchors": [
                {
                    "id": "box-only",
                    "label": "box only",
                    "viewpoint": "object_crop",
                    "region": {
                        "kind": "bbox",
                        "bbox_xyxy_norm": [0.2, 0.2, 0.6, 0.8],
                        "point_xy_norm": None,
                    },
                }
            ]
        },
        primary_viewpoint="object_crop",
        required_viewpoints=["object_crop"],
    )

    assert annotation["anchors"][0]["region"]["point_xy_norm"] == [0.4, 0.5]


def test_output_rows_include_failed_boolean_and_region_annotation_meta() -> None:
    meta = build_region_annotation_meta(
        model_name="/models/qwen",
        prompt="prompt text",
        input_jsonl="input.jsonl",
        min_pixels=1,
        max_pixels=2,
        timestamp="2026-07-10T00:00:00+00:00",
    )
    success = build_success_output_row(
        row=_row(),
        key="image-1::caption",
        index=7,
        annotation={"annotation_version": "hic-region-v1"},
        raw="{}",
        meta=meta,
    )
    failure = build_failure_output_row(
        row=_row(),
        key="image-1::caption",
        index=7,
        error="bad json",
        raw="{",
        meta=meta,
    )

    assert success["failed"] is False
    assert success["region_parse_error"] == ""
    assert success["region_annotation_meta"] == meta
    assert success["region_annotation_meta"]["model_name"] == "/models/qwen"
    assert success["region_annotation_meta"]["prompt_version"] == "hic-region-v1-qwen7b"
    assert success["region_annotation_meta"]["source_path"] == "input.jsonl"
    assert success["region_annotation_meta"]["min_pixels"] == 1
    assert success["region_annotation_meta"]["max_pixels"] == 2
    assert success["region_annotation_meta"]["created_at"] == "2026-07-10T00:00:00+00:00"
    assert failure["failed"] is True
    assert failure["region_annotation"] is None
    assert failure["region_parse_error"] == "bad json"
    assert failure["region_annotation_meta"] == meta
