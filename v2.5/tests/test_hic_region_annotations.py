from src.analysis.hic_region_annotations import (
    CORE_VIEWPOINTS,
    bbox_norm_to_pixels,
    compact_region_payload_for_prompt,
    normalize_bbox_xyxy_norm,
    normalize_point_xy_norm,
    normalize_region_annotation,
)


def test_normalize_bbox_xyxy_norm_accepts_numeric_strings() -> None:
    assert normalize_bbox_xyxy_norm([0, "0.2", 0.8, 1]) == [0.0, 0.2, 0.8, 1.0]


def test_normalize_bbox_xyxy_norm_rejects_invalid_boxes() -> None:
    assert normalize_bbox_xyxy_norm([-0.1, 0.2, 0.8, 1]) is None
    assert normalize_bbox_xyxy_norm([0, 0.2, 1.1, 1]) is None
    assert normalize_bbox_xyxy_norm([0.8, 0.2, 0.2, 1]) is None
    assert normalize_bbox_xyxy_norm([0, 0.8, 0.2, 0.2]) is None
    assert normalize_bbox_xyxy_norm([0, 0.2, 0.8]) is None
    assert normalize_bbox_xyxy_norm([0, "left", 0.8, 1]) is None


def test_normalize_point_xy_norm_accepts_valid_points_and_rejects_invalid_points() -> None:
    assert normalize_point_xy_norm(["0.25", 1]) == [0.25, 1.0]
    assert normalize_point_xy_norm([-0.1, 0.5]) is None
    assert normalize_point_xy_norm([0.1, 1.2]) is None
    assert normalize_point_xy_norm([0.1]) is None
    assert normalize_point_xy_norm(["x", 0.5]) is None


def test_bbox_norm_to_pixels_converts_normalized_coordinates() -> None:
    assert bbox_norm_to_pixels([0.1, 0.2, 0.5, 0.7], width=200, height=100) == (20, 20, 100, 70)


def test_core_viewpoints_order_is_fixed() -> None:
    assert CORE_VIEWPOINTS == (
        "face_expression_crop",
        "relation_crop",
        "context_scene_view",
        "text_region_crop",
        "object_crop",
        "full_image",
        "pose_action_view",
        "scale_reference_crop",
    )


def test_normalize_region_annotation_fills_defaults_caps_items_and_confidence() -> None:
    annotation = normalize_region_annotation(
        {
            "needs_full_image": True,
            "anchors": [
                {
                    "id": "a1",
                    "label": "  person  ",
                    "role": " setup ",
                    "source_anchor_id": "raw-1",
                    "viewpoint": "foreground_background_view",
                    "region": {
                        "kind": "bbox",
                        "bbox_xyxy_norm": [0, "0.1", 0.5, 0.6],
                        "point_xy_norm": ["0.25", "0.3"],
                        "confidence": "high",
                        "evidence": " visible face ",
                    },
                },
                {
                    "id": "a2",
                    "label": "sign",
                    "viewpoint": "text_region_crop",
                    "region": {
                        "bbox_xyxy_norm": [0.1, 0.1, 0.3, 0.2],
                        "point_xy_norm": [0.2, 0.15],
                        "confidence": "unexpected",
                    },
                },
                {
                    "id": "a3",
                    "label": "cup",
                    "region": {
                        "bbox_xyxy_norm": [0.2, 0.3, 0.4, 0.5],
                        "point_xy_norm": [0.3, 0.4],
                    },
                },
                {
                    "id": "a4",
                    "label": "table",
                    "region": {
                        "bbox_xyxy_norm": [0.0, 0.6, 0.9, 0.9],
                        "point_xy_norm": [0.4, 0.75],
                    },
                },
                {"id": "a5", "label": "ignored"},
            ],
            "relations": [
                {"subject": "a1", "predicate": "holding", "object": "a3", "confidence": "medium"},
                {"subject": "a1", "predicate": "near", "object": "a2"},
                {"subject": "a2", "predicate": "on", "object": "a4"},
                {"subject": "a3", "predicate": "on", "object": "a4"},
                {"subject": "a5", "predicate": "ignored", "object": "a1"},
            ],
            "annotation_confidence": "surprising",
        },
        primary_viewpoint="foreground_background_view",
        required_viewpoints=["text_region_crop", "foreground_background_view", "text_region_crop"],
    )

    assert annotation["annotation_version"] == "hic-region-v1"
    assert annotation["primary_viewpoint"] == "relation_crop"
    assert annotation["required_viewpoints"] == ["text_region_crop", "relation_crop"]
    assert annotation["viewpoint_set"] == list(CORE_VIEWPOINTS)
    assert annotation["needs_full_image"] is True
    assert annotation["annotation_confidence"] == "low"
    assert annotation["uncertainty"] == ""
    assert "visual_anchors" not in annotation
    assert "confidence" not in annotation
    assert len(annotation["anchors"]) == 4
    assert len(annotation["relations"]) == 4
    assert annotation["anchors"][0] == {
        "id": "a1",
        "label": "person",
        "role": "setup",
        "source_anchor_id": "raw-1",
        "viewpoint": "relation_crop",
        "region": {
            "kind": "bbox",
            "bbox_xyxy_norm": [0.0, 0.1, 0.5, 0.6],
            "point_xy_norm": [0.25, 0.3],
            "confidence": "high",
            "evidence": "visible face",
        },
    }
    assert annotation["anchors"][1]["region"]["confidence"] == "low"
    assert annotation["relations"][0]["confidence"] == "medium"
    assert annotation["relations"][1]["confidence"] == "low"


def test_normalize_region_annotation_preserves_coordinate_fields_and_flags_uncertainty() -> None:
    annotation = normalize_region_annotation(
        {
            "anchors": [
                {
                    "id": "bad",
                    "label": "bad box",
                    "viewpoint": "full_image",
                    "region": {
                        "kind": "bbox",
                        "bbox_xyxy_norm": [0.8, 0.2, 0.2, 0.6],
                        "point_xy_norm": [1.2, 0.5],
                    },
                }
            ],
        },
        primary_viewpoint="full_image",
        required_viewpoints=["full_image"],
    )

    region = annotation["anchors"][0]["region"]
    assert region["bbox_xyxy_norm"] is None
    assert region["point_xy_norm"] is None
    assert isinstance(annotation["uncertainty"], str)
    assert annotation["uncertainty"]


def test_normalize_region_annotation_fills_missing_point_from_valid_bbox_center() -> None:
    annotation = normalize_region_annotation(
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
            ],
        },
        primary_viewpoint="object_crop",
        required_viewpoints=["object_crop"],
    )

    region = annotation["anchors"][0]["region"]
    assert region["bbox_xyxy_norm"] == [0.2, 0.2, 0.6, 0.8]
    assert region["point_xy_norm"] == [0.4, 0.5]
    assert annotation["uncertainty"] == ""


def test_compact_region_payload_for_prompt_excludes_verbose_or_leaky_fields() -> None:
    annotation = {
        "annotation_version": "hic-region-v1",
        "primary_viewpoint": "relation_crop",
        "required_viewpoints": ["relation_crop"],
        "needs_full_image": False,
        "annotation_confidence": "medium",
        "uncertainty": "coordinate validation failed for bad",
        "gold_caption": "do not leak",
        "raw_region_response": {"large": "blob"},
        "literal_image_description": "long literal description",
        "anchors": [
            {
                "id": "a1",
                "label": "person",
                "role": "setup",
                "source_anchor_id": "raw-1",
                "evidence": "This evidence prose is intentionally long and should not be included.",
                "viewpoint": "relation_crop",
                "region": {
                    "kind": "bbox",
                    "bbox_xyxy_norm": [0.0, 0.1, 0.5, 0.6],
                    "point_xy_norm": [0.25, 0.3],
                    "confidence": "high",
                    "evidence": "This evidence prose is intentionally long and should not be included.",
                },
            }
        ],
        "relations": [
            {
                "subject": "person",
                "predicate": "holding",
                "object": "cup",
                "evidence": "long relation evidence",
                "confidence": "medium",
            }
        ],
    }

    compact = compact_region_payload_for_prompt(annotation)

    assert compact == {
        "annotation_version": "hic-region-v1",
        "primary_viewpoint": "relation_crop",
        "required_viewpoints": ["relation_crop"],
        "needs_full_image": False,
        "annotation_confidence": "medium",
        "uncertainty": "coordinate validation failed for bad",
        "anchors": [
            {
                "id": "a1",
                "label": "person",
                "role": "setup",
                "source_anchor_id": "raw-1",
                "viewpoint": "relation_crop",
                "region": {
                    "kind": "bbox",
                    "bbox_xyxy_norm": [0.0, 0.1, 0.5, 0.6],
                    "point_xy_norm": [0.25, 0.3],
                    "confidence": "high",
                },
            }
        ],
        "relations": [
            {
                "subject": "person",
                "predicate": "holding",
                "object": "cup",
                "confidence": "medium",
            }
        ],
    }
    assert "gold_caption" not in str(compact)
    assert "raw_region_response" not in str(compact)
    assert "literal_image_description" not in str(compact)
    assert "evidence prose" not in str(compact)


def test_compact_region_payload_for_prompt_enforces_version_and_coordinate_uncertainty() -> None:
    compact = compact_region_payload_for_prompt(
        {
            "annotation_version": "wrong-version",
            "primary_viewpoint": "relation_crop",
            "required_viewpoints": ["relation_crop"],
            "anchors": [
                {
                    "id": "a1",
                    "label": "bad region",
                    "viewpoint": "relation_crop",
                    "region": {
                        "bbox_xyxy_norm": [0.9, 0.1, 0.2, 0.3],
                        "point_xy_norm": [0.5, 1.2],
                    },
                }
            ],
        }
    )

    assert compact["annotation_version"] == "hic-region-v1"
    assert compact["anchors"][0]["region"]["bbox_xyxy_norm"] is None
    assert compact["anchors"][0]["region"]["point_xy_norm"] is None
    assert isinstance(compact["uncertainty"], str)
    assert compact["uncertainty"]
