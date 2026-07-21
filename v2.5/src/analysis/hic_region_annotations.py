from __future__ import annotations

import math
from typing import Any


CORE_VIEWPOINTS: tuple[str, ...] = (
    "face_expression_crop",
    "relation_crop",
    "context_scene_view",
    "text_region_crop",
    "object_crop",
    "full_image",
    "pose_action_view",
    "scale_reference_crop",
)
ANNOTATION_VERSION = "hic-region-v1"

_CONFIDENCE_VALUES = {"low", "medium", "high"}
_VIEWPOINT_ALIASES = {"foreground_background_view": "relation_crop"}


def clean_region_text(value: Any, max_chars: int = 240) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _normalize_viewpoint(value: Any) -> str:
    text = clean_region_text(value, max_chars=80)
    return _VIEWPOINT_ALIASES.get(text, text)


def clean_viewpoints(values: Any, fallback: str = "full_image") -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        values = [fallback]

    result: list[str] = []
    allowed = set(CORE_VIEWPOINTS)
    for value in values:
        viewpoint = _normalize_viewpoint(value)
        if viewpoint in allowed and viewpoint not in result:
            result.append(viewpoint)
        if len(result) >= 4:
            break

    fallback_viewpoint = _normalize_viewpoint(fallback)
    if not result and fallback_viewpoint in allowed:
        result.append(fallback_viewpoint)
    return result or ["full_image"]


def _coerce_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _coerce_normalized_sequence(value: Any, length: int) -> list[float] | None:
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or len(value) != length:
        return None

    numbers: list[float] = []
    for item in value:
        number = _coerce_float(item)
        if number is None or number < 0.0 or number > 1.0:
            return None
        numbers.append(number)
    return numbers


def normalize_bbox_xyxy_norm(value: Any) -> list[float] | None:
    numbers = _coerce_normalized_sequence(value, 4)
    if numbers is None:
        return None
    x1, y1, x2, y2 = numbers
    if x2 <= x1 or y2 <= y1:
        return None
    return numbers


def normalize_point_xy_norm(value: Any) -> list[float] | None:
    return _coerce_normalized_sequence(value, 2)


def _clean_confidence(value: Any) -> str:
    text = clean_region_text(value, max_chars=20).lower()
    if text in _CONFIDENCE_VALUES:
        return text
    return "low"


def _clean_viewpoint(value: Any, fallback: str = "full_image") -> str:
    return clean_viewpoints([value], fallback=fallback)[0]


def _normalize_region_payload(region: Any) -> tuple[dict[str, Any], str | None]:
    raw_region = region if isinstance(region, dict) else {}
    bbox = normalize_bbox_xyxy_norm(raw_region.get("bbox_xyxy_norm"))
    point = normalize_point_xy_norm(raw_region.get("point_xy_norm"))
    uncertainty = None
    if bbox is not None and point is None:
        x1, y1, x2, y2 = bbox
        point = [round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6)]
    elif bbox is None or point is None:
        bbox = None
        point = None
        uncertainty = "invalid or missing region coordinates"

    normalized = {
        "kind": clean_region_text(raw_region.get("kind"), max_chars=40) or "region",
        "bbox_xyxy_norm": bbox,
        "point_xy_norm": point,
        "confidence": _clean_confidence(raw_region.get("confidence")),
        "evidence": clean_region_text(raw_region.get("evidence")),
    }
    return normalized, uncertainty


def _join_uncertainty(items: list[str]) -> str:
    cleaned: list[str] = []
    for item in items:
        text = clean_region_text(item, max_chars=240)
        if text and text not in cleaned:
            cleaned.append(text)
    return "; ".join(cleaned)


def _clean_uncertainty(value: Any) -> str:
    if isinstance(value, list):
        return _join_uncertainty([str(item) for item in value])
    return clean_region_text(value, max_chars=600)


def validate_region(region: dict[str, Any]) -> dict[str, Any]:
    raw_region = region.get("region")
    if not isinstance(raw_region, dict):
        raw_region = region
    normalized_region, _ = _normalize_region_payload(raw_region)
    return normalized_region


def _normalize_anchor(anchor: dict[str, Any], index: int) -> tuple[dict[str, Any] | None, str | None]:
    label = clean_region_text(anchor.get("label"), max_chars=120)
    if not label:
        return None, None

    raw_region = anchor.get("region")
    if not isinstance(raw_region, dict):
        raw_region = anchor
    normalized_region, uncertainty = _normalize_region_payload(raw_region)
    anchor_id = clean_region_text(anchor.get("id"), max_chars=40) or f"a{index}"
    normalized = {
        "id": anchor_id,
        "label": label,
        "role": clean_region_text(anchor.get("role"), max_chars=160),
        "source_anchor_id": clean_region_text(anchor.get("source_anchor_id"), max_chars=80),
        "viewpoint": _clean_viewpoint(anchor.get("viewpoint"), fallback="full_image"),
        "region": normalized_region,
    }
    if uncertainty is not None:
        uncertainty = f"{uncertainty} for anchor {anchor_id}"
    return normalized, uncertainty


def _clean_relation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    subject = clean_region_text(value.get("subject"), max_chars=80)
    predicate = clean_region_text(value.get("predicate"), max_chars=80)
    obj = clean_region_text(value.get("object"), max_chars=80)
    if not (subject or predicate or obj):
        return None
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "confidence": _clean_confidence(value.get("confidence")),
    }


def normalize_region_annotation(
    value: dict[str, Any],
    *,
    primary_viewpoint: str,
    required_viewpoints: list[str],
) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    required = clean_viewpoints(required_viewpoints, fallback=primary_viewpoint)
    primary = clean_viewpoints([primary_viewpoint], fallback=required[0])[0]
    if primary not in required:
        required.insert(0, primary)
    required = required[:4]

    anchors: list[dict[str, Any]] = []
    uncertainty: list[str] = []
    raw_anchors = raw.get("anchors")
    if isinstance(raw_anchors, list):
        for index, item in enumerate(raw_anchors, start=1):
            if not isinstance(item, dict):
                continue
            anchor, anchor_uncertainty = _normalize_anchor(item, index)
            if anchor is None:
                continue
            anchors.append(anchor)
            if anchor_uncertainty is not None:
                uncertainty.append(anchor_uncertainty)
            if len(anchors) >= 4:
                break

    relations: list[dict[str, Any]] = []
    raw_relations = raw.get("relations")
    if isinstance(raw_relations, list):
        for item in raw_relations:
            relation = _clean_relation(item)
            if relation is not None:
                relations.append(relation)
            if len(relations) >= 4:
                break

    return {
        "annotation_version": ANNOTATION_VERSION,
        "viewpoint_set": list(CORE_VIEWPOINTS),
        "primary_viewpoint": primary,
        "required_viewpoints": required,
        "needs_full_image": bool(raw.get("needs_full_image")),
        "anchors": anchors,
        "relations": relations,
        "annotation_confidence": _clean_confidence(raw.get("annotation_confidence", raw.get("confidence"))),
        "uncertainty": _join_uncertainty(uncertainty) or _clean_uncertainty(raw.get("uncertainty")),
    }


def bbox_norm_to_pixels(
    bbox: list[float],
    *,
    width: int,
    height: int,
    min_size: int = 2,
) -> tuple[int, int, int, int]:
    normalized = normalize_bbox_xyxy_norm(bbox)
    if normalized is None:
        raise ValueError("bbox must be normalized xyxy coordinates with x2 > x1 and y2 > y1")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    x1, y1, x2, y2 = normalized
    px1 = round(x1 * width)
    py1 = round(y1 * height)
    px2 = round(x2 * width)
    py2 = round(y2 * height)

    if min_size > 0:
        px2 = max(px2, px1 + min_size)
        py2 = max(py2, py1 + min_size)
        if px2 > width:
            px1 = max(0, width - min_size)
            px2 = width
        if py2 > height:
            py1 = max(0, height - min_size)
            py2 = height

    return (
        max(0, min(width, px1)),
        max(0, min(height, py1)),
        max(0, min(width, px2)),
        max(0, min(height, py2)),
    )


def _compact_anchor(anchor: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(anchor, dict):
        return None, None
    compact: dict[str, Any] = {
        "id": clean_region_text(anchor.get("id"), max_chars=40),
        "label": clean_region_text(anchor.get("label"), max_chars=120),
        "role": clean_region_text(anchor.get("role"), max_chars=160),
        "source_anchor_id": clean_region_text(anchor.get("source_anchor_id"), max_chars=80),
        "viewpoint": _clean_viewpoint(anchor.get("viewpoint"), fallback="full_image"),
    }

    region = anchor.get("region")
    if not isinstance(region, dict):
        region = {}
    bbox = normalize_bbox_xyxy_norm(region.get("bbox_xyxy_norm"))
    point = normalize_point_xy_norm(region.get("point_xy_norm"))
    uncertainty = None
    if bbox is None or point is None:
        bbox = None
        point = None
        anchor_id = compact.get("id") or compact.get("label") or "anchor"
        uncertainty = f"invalid or missing region coordinates for anchor {anchor_id}"
    compact["region"] = {
        "kind": clean_region_text(region.get("kind"), max_chars=40) or "region",
        "bbox_xyxy_norm": bbox,
        "point_xy_norm": point,
        "confidence": _clean_confidence(region.get("confidence")),
    }
    return (compact, uncertainty) if compact.get("label") or compact.get("role") else (None, None)


def compact_region_payload_for_prompt(
    annotation: dict[str, Any] | None,
    *,
    max_anchors: int = 4,
    max_relations: int = 4,
) -> dict[str, Any]:
    annotation = annotation or {}
    required = clean_viewpoints(annotation.get("required_viewpoints"), fallback="full_image")
    primary = clean_viewpoints([annotation.get("primary_viewpoint")], fallback=required[0])[0]

    anchors: list[dict[str, Any]] = []
    raw_anchors = annotation.get("anchors")
    compact_uncertainty: list[str] = []
    if isinstance(raw_anchors, list):
        for item in raw_anchors:
            compact, anchor_uncertainty = _compact_anchor(item)
            if compact is not None:
                anchors.append(compact)
            if anchor_uncertainty is not None:
                compact_uncertainty.append(anchor_uncertainty)
            if len(anchors) >= max_anchors:
                break

    relations: list[dict[str, Any]] = []
    raw_relations = annotation.get("relations")
    if isinstance(raw_relations, list):
        for item in raw_relations:
            relation = _clean_relation(item)
            if relation is not None:
                relations.append(relation)
            if len(relations) >= max_relations:
                break

    uncertainty = _join_uncertainty([_clean_uncertainty(annotation.get("uncertainty")), *compact_uncertainty])

    return {
        "annotation_version": ANNOTATION_VERSION,
        "primary_viewpoint": primary,
        "required_viewpoints": required,
        "needs_full_image": bool(annotation.get("needs_full_image")),
        "annotation_confidence": _clean_confidence(annotation.get("annotation_confidence", annotation.get("confidence"))),
        "uncertainty": uncertainty,
        "anchors": anchors,
        "relations": relations,
    }
