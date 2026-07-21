from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageChops

from scripts.render_hic_region_overlays import (
    crop_anchor_regions,
    draw_overlay,
    make_crop_sheet,
    render_rows,
)


LONG_EVIDENCE = "This long evidence text should never be drawn on the rendered image or stored as metadata."


def _annotation(*, bbox: list[float] | None = None) -> dict:
    return {
        "annotation_version": "hic-region-v1",
        "primary_viewpoint": "face_expression_crop",
        "required_viewpoints": ["face_expression_crop"],
        "needs_full_image": False,
        "annotation_confidence": "high",
        "uncertainty": "",
        "anchors": [
            {
                "id": "a1",
                "label": "face",
                "role": "setup",
                "source_anchor_id": "raw-1",
                "viewpoint": "face_expression_crop",
                "region": {
                    "kind": "bbox",
                    "bbox_xyxy_norm": bbox if bbox is not None else [0.1, 0.1, 0.5, 0.5],
                    "point_xy_norm": [0.3, 0.3] if bbox is not None else [0.3, 0.3],
                    "confidence": "high",
                    "evidence": LONG_EVIDENCE,
                },
            }
        ],
        "relations": [],
    }


def _row(image_path: Path) -> dict:
    return {
        "image": str(image_path),
        "image_id": "tiny",
        "gold_caption": "a small joke",
        "analysis": {
            "humor_type": "contrast",
            "humor_point": "tiny contrast",
            "visual_anchors": [
                {
                    "id": "raw-1",
                    "label": "face",
                    "role": "setup",
                    "evidence": LONG_EVIDENCE,
                }
            ],
        },
        "region_annotation": _annotation(),
    }


def test_draw_overlay_changes_synthetic_image() -> None:
    image = Image.new("RGB", (100, 100), "white")

    rendered = draw_overlay(image, _annotation())

    assert rendered.size == image.size
    assert ImageChops.difference(image, rendered).getbbox() is not None


def test_make_crop_sheet_with_two_crops_writes_nonzero_image(tmp_path: Path) -> None:
    crops = [
        ("A1", Image.new("RGB", (20, 10), "red")),
        ("A2", Image.new("RGB", (12, 24), "blue")),
    ]

    sheet = make_crop_sheet(crops, cell_size=64)
    output = tmp_path / "sheet.jpg"
    sheet.save(output)

    assert output.stat().st_size > 0
    assert sheet.width > 0
    assert sheet.height > 0


def test_render_rows_writes_overlay_crop_sheet_and_enriched_jsonl(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)

    output_jsonl = tmp_path / "rendered.jsonl"
    html_path = tmp_path / "review.html"
    rendered = render_rows(
        [_row(image_path)],
        output_jsonl=output_jsonl,
        overlay_dir=tmp_path / "overlays",
        crop_dir=tmp_path / "crops",
        review_html=html_path,
        image_root=None,
        overwrite=True,
    )

    assert len(rendered) == 1
    output_row = json.loads(output_jsonl.read_text(encoding="utf-8").strip())
    assert Path(output_row["overlay_image"]).exists()
    assert Path(output_row["crop_sheet"]).exists()
    assert rendered[0]["image"] == str(image_path)
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Original" in html
    assert "Overlay" in html
    assert "Crop sheet" in html
    assert "a small joke" in html
    assert "region_annotation" in html
    assert 'src="overlays/' in html
    assert 'src="crops/' in html


def test_render_rows_handles_failed_rows_without_region_annotation(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)
    row = _row(image_path)
    row["failed"] = True
    row["region_annotation"] = None

    output_jsonl = tmp_path / "rendered.jsonl"
    rendered = render_rows(
        [row],
        output_jsonl=output_jsonl,
        overlay_dir=tmp_path / "overlays",
        crop_dir=tmp_path / "crops",
        review_html=tmp_path / "review.html",
        image_root=None,
        overwrite=True,
    )

    assert len(rendered) == 1
    output_row = json.loads(output_jsonl.read_text(encoding="utf-8").strip())
    assert output_row["failed"] is True
    assert Path(output_row["overlay_image"]).exists()
    assert Path(output_row["crop_sheet"]).exists()


def test_long_evidence_text_is_not_used_as_render_label_or_image_metadata() -> None:
    image = Image.new("RGB", (100, 100), "white")

    overlay = draw_overlay(image, _annotation())
    crops = crop_anchor_regions(image, _annotation())

    assert "hic_render_labels" in overlay.info
    assert overlay.info["hic_render_labels"] == ["A1"]
    assert LONG_EVIDENCE not in str(overlay.info)
    assert [label for label, _ in crops] == ["A1"]


def test_review_html_uses_resolved_image_root_path_and_relative_assets(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "image.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)
    row = _row(Path("image.jpg"))
    row["image"] = "image.jpg"

    html_path = tmp_path / "reviews" / "review.html"
    render_rows(
        [row],
        output_jsonl=tmp_path / "rendered.jsonl",
        overlay_dir=tmp_path / "annotations" / "overlays",
        crop_dir=tmp_path / "annotations" / "crops",
        review_html=html_path,
        image_root=image_root,
        overwrite=True,
    )

    html = html_path.read_text(encoding="utf-8")
    assert 'src="../images/image.jpg"' in html
    assert 'src="../annotations/overlays/' in html
    assert 'src="../annotations/crops/' in html
