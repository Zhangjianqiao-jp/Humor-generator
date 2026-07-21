from __future__ import annotations

from pathlib import Path

from scripts.generate_guided_lora_candidates import (
    auxiliary_images_for_method,
    build_generation_messages,
    context_has_required_fields,
)


def test_region_methods_require_region_annotation() -> None:
    context = {"analysis": {"primary_viewpoint": "relation_crop"}}

    assert context_has_required_fields(context, "hic-compact-json") is True
    assert context_has_required_fields(context, "hic-compact-json-region") is False
    assert context_has_required_fields(context, "hic-compact-json-overlay") is False
    assert context_has_required_fields(context, "hic-compact-json-crop") is False
    context["region_annotation"] = {"annotation_version": "hic-region-v1"}
    assert context_has_required_fields(context, "hic-compact-json-region") is True
    assert context_has_required_fields(context, "hic-compact-json-overlay") is True
    assert context_has_required_fields(context, "hic-compact-json-crop") is True


def test_auxiliary_images_for_region_json_method_returns_none() -> None:
    context = {"region_annotation": {"annotation_version": "hic-region-v1"}}

    assert auxiliary_images_for_method(context, "hic-compact-json-region") == []


def test_auxiliary_images_for_overlay_and_crop_methods_return_existing_paths(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.jpg"
    crop = tmp_path / "crop.jpg"
    overlay.write_bytes(b"overlay")
    crop.write_bytes(b"crop")
    context = {
        "overlay_image": str(overlay),
        "crop_sheet": str(crop),
        "region_annotation": {"annotation_version": "hic-region-v1"},
    }

    assert auxiliary_images_for_method(context, "hic-compact-json-overlay") == [overlay]
    assert auxiliary_images_for_method(context, "hic-compact-json-crop") == [crop]


def test_auxiliary_images_for_overlay_and_crop_methods_return_empty_for_missing_paths(tmp_path: Path) -> None:
    context = {
        "overlay_image": str(tmp_path / "missing-overlay.jpg"),
        "crop_sheet": str(tmp_path / "missing-crop.jpg"),
        "region_annotation": {"annotation_version": "hic-region-v1"},
    }

    assert auxiliary_images_for_method(context, "hic-compact-json-overlay") == []
    assert auxiliary_images_for_method(context, "hic-compact-json-crop") == []


def test_build_generation_messages_puts_original_then_auxiliary_images_then_text(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    aux1 = tmp_path / "overlay.jpg"
    aux2 = tmp_path / "crop.jpg"

    messages = build_generation_messages(image, "caption prompt", auxiliary_image_paths=[aux1, aux2])

    content = messages[0]["content"]
    assert content == [
        {"type": "image", "image": str(image)},
        {"type": "image", "image": str(aux1)},
        {"type": "image", "image": str(aux2)},
        {"type": "text", "text": "caption prompt"},
    ]
