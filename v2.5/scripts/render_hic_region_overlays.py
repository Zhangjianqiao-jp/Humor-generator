#!/usr/bin/env python
from __future__ import annotations

import html
import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.hic_region_annotations import bbox_norm_to_pixels, compact_region_payload_for_prompt
from src.utils.io import read_jsonl, write_jsonl


DEFAULT_INPUT_JSONL = Path("outputs/annotations/hic_region_annotations_800.jsonl")
DEFAULT_OUTPUT_JSONL = Path("outputs/annotations/hic_region_annotations_800_rendered.jsonl")
DEFAULT_OVERLAY_DIR = Path("outputs/annotations/hic_region_overlays")
DEFAULT_CROP_DIR = Path("outputs/annotations/hic_region_crops")
DEFAULT_REVIEW_HTML = Path("outputs/reviews/hic_region_annotations_800.html")


def anchor_color(index: int) -> tuple[int, int, int]:
    colors = (
        (230, 57, 70),
        (29, 113, 184),
        (42, 157, 143),
        (245, 166, 35),
        (128, 74, 166),
        (80, 170, 80),
    )
    return colors[index % len(colors)]


def _annotation_anchors(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = annotation.get("anchors") if isinstance(annotation, dict) else []
    if not isinstance(anchors, list):
        return []
    return [anchor for anchor in anchors if isinstance(anchor, dict)]


def _anchor_label(index: int) -> str:
    return f"A{index + 1}"


def _anchor_bbox(anchor: dict[str, Any], *, width: int, height: int) -> tuple[int, int, int, int] | None:
    region = anchor.get("region") if isinstance(anchor.get("region"), dict) else {}
    bbox = region.get("bbox_xyxy_norm")
    if bbox is None:
        return None
    try:
        return bbox_norm_to_pixels(bbox, width=width, height=height)
    except ValueError:
        return None


def _text_bbox(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> tuple[int, int, int, int]:
    try:
        return draw.textbbox(xy, text, font=ImageFont.load_default())
    except AttributeError:
        width = len(text) * 6
        height = 11
        x, y = xy
        return (x, y, x + width, y + height)


def draw_overlay(image: Image.Image, annotation: dict[str, Any]) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    labels: list[str] = []
    width, height = rendered.size

    for index, anchor in enumerate(_annotation_anchors(annotation)):
        box = _anchor_bbox(anchor, width=width, height=height)
        if box is None:
            continue
        label = _anchor_label(index)
        color = anchor_color(index)
        x1, y1, x2, y2 = box
        for inset in range(3):
            draw.rectangle((x1 - inset, y1 - inset, x2 + inset, y2 + inset), outline=color)

        text_x = max(0, min(x1, width - 1))
        text_y = max(0, y1 - 14)
        text_box = _text_bbox(draw, (text_x, text_y), label)
        if text_box[3] > y1 and y1 + 14 < height:
            text_y = y1 + 2
            text_box = _text_bbox(draw, (text_x, text_y), label)
        padded = (
            max(0, text_box[0] - 3),
            max(0, text_box[1] - 2),
            min(width, text_box[2] + 3),
            min(height, text_box[3] + 2),
        )
        draw.rectangle(padded, fill=color)
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=ImageFont.load_default())
        labels.append(label)

    rendered.info["hic_render_labels"] = labels
    return rendered


def crop_anchor_regions(image: Image.Image, annotation: dict[str, Any]) -> list[tuple[str, Image.Image]]:
    source = image.convert("RGB")
    width, height = source.size
    crops: list[tuple[str, Image.Image]] = []
    for index, anchor in enumerate(_annotation_anchors(annotation)):
        box = _anchor_bbox(anchor, width=width, height=height)
        if box is None:
            continue
        label = _anchor_label(index)
        crops.append((label, source.crop(box)))
    return crops


def make_crop_sheet(crops: list[tuple[str, Image.Image]], *, cell_size: int = 224) -> Image.Image:
    margin = 10
    label_height = 18
    if not crops:
        sheet = Image.new("RGB", (cell_size, cell_size), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        draw.text((margin, margin), "No crops", fill=(80, 80, 80), font=ImageFont.load_default())
        return sheet

    columns = min(4, max(1, len(crops)))
    rows = math.ceil(len(crops) / columns)
    sheet = Image.new(
        "RGB",
        (columns * cell_size, rows * (cell_size + label_height)),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, crop) in enumerate(crops):
        col = index % columns
        row = index // columns
        x = col * cell_size
        y = row * (cell_size + label_height)
        draw.rectangle((x, y, x + cell_size - 1, y + label_height - 1), fill=(30, 30, 30))
        draw.text((x + 6, y + 3), label, fill=(255, 255, 255), font=ImageFont.load_default())

        crop = crop.convert("RGB")
        crop.thumbnail((cell_size - 2 * margin, cell_size - 2 * margin), Image.Resampling.LANCZOS)
        paste_x = x + (cell_size - crop.width) // 2
        paste_y = y + label_height + (cell_size - crop.height) // 2
        sheet.paste(crop, (paste_x, paste_y))
    return sheet


def _row_key(row: dict[str, Any], index: int) -> str:
    key = str(row.get("row_key") or row.get("image_id") or Path(str(row.get("image") or f"row-{index}")).stem)
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in key)[:160] or f"row-{index}"


def _resolve_image_path(row: dict[str, Any], image_root: Path | None) -> Path:
    image_path = Path(str(row.get("image") or "")).expanduser()
    if image_path.is_absolute() or image_root is None:
        return image_path
    return image_root / image_path


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _json_html(value: Any) -> str:
    return _escape(json.dumps(value, ensure_ascii=False, indent=2))


def _compact_joke_annotation(row: dict[str, Any]) -> dict[str, Any]:
    analysis = row.get("analysis") if isinstance(row.get("analysis"), dict) else {}
    return {
        "humor_type": analysis.get("humor_type"),
        "humor_point": analysis.get("humor_point"),
        "visual_anchors": analysis.get("visual_anchors") or [],
        "required_viewpoints": analysis.get("required_viewpoints") or [],
        "primary_viewpoint": analysis.get("primary_viewpoint"),
        "uncertainty": analysis.get("uncertainty"),
    }


def _missing_anchor_labels(annotation: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for index, anchor in enumerate(_annotation_anchors(annotation)):
        region = anchor.get("region") if isinstance(anchor.get("region"), dict) else {}
        if region.get("bbox_xyxy_norm") is None:
            missing.append(_anchor_label(index))
    return missing


def _review_src(path: Any, review_html: Path) -> str:
    if path in (None, ""):
        return ""
    source = Path(str(path)).expanduser()
    try:
        rel = source.relative_to(review_html.parent)
    except ValueError:
        try:
            rel = Path(
                __import__("os").path.relpath(
                    source,
                    start=review_html.parent,
                )
            )
        except ValueError:
            rel = source
    return _escape(str(rel))


def _write_review_html(rows: list[dict[str, Any]], review_html: Path) -> None:
    review_html.parent.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for index, row in enumerate(rows, start=1):
        annotation = row.get("region_annotation") if isinstance(row.get("region_annotation"), dict) else {}
        missing = _missing_anchor_labels(annotation)
        missing_html = "<em>None</em>" if not missing else _escape(", ".join(missing))
        cards.append(
            f"""
            <section class="card">
              <h2>{index}. {_escape(row.get("image_id") or Path(str(row.get("image") or "")).stem)}</h2>
              <div class="media-grid">
                <figure><img src="{_review_src(row.get("review_original_image") or row.get("image"), review_html)}" alt="original"><figcaption>Original</figcaption></figure>
                <figure><img src="{_review_src(row.get("overlay_image"), review_html)}" alt="overlay"><figcaption>Overlay</figcaption></figure>
                <figure><img src="{_review_src(row.get("crop_sheet"), review_html)}" alt="crop sheet"><figcaption>Crop sheet</figcaption></figure>
              </div>
              <div class="details-grid">
                <section><h3>Gold caption</h3><p>{_escape(row.get("gold_caption"))}</p></section>
                <section><h3>Compact joke annotations</h3><pre>{_json_html(_compact_joke_annotation(row))}</pre></section>
                <section data-section="region_annotation"><h3>Compact region annotation</h3><pre>{_json_html(compact_region_payload_for_prompt(annotation))}</pre></section>
                <section><h3>Skipped crop anchors</h3><p>{missing_html}</p></section>
              </div>
            </section>
            """
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HIC Region Annotation Review</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f6f6f3; color: #222; }}
    header {{ padding: 16px 24px; background: #20252b; color: white; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 20px; }}
    .card {{ margin: 0 0 20px; padding: 16px; background: white; border: 1px solid #d8d8d8; border-radius: 8px; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; }}
    figcaption {{ margin-top: 6px; font-size: 13px; color: #555; }}
    img {{ width: 100%; max-height: 420px; object-fit: contain; background: #eee; border: 1px solid #ddd; }}
    .details-grid {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 12px; margin-top: 14px; }}
    pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; background: #f3f4f6; padding: 10px; border-radius: 6px; }}
    p {{ margin: 0; }}
    @media (max-width: 820px) {{ .media-grid, .details-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header><h1>HIC Region Annotation Review</h1></header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
    review_html.write_text(page, encoding="utf-8")


def render_rows(
    rows: list[dict[str, Any]],
    *,
    output_jsonl: Path,
    overlay_dir: Path,
    crop_dir: Path,
    review_html: Path,
    image_root: Path | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    if output_jsonl.exists() and not overwrite:
        raise FileExistsError(f"output JSONL already exists: {output_jsonl}")
    selected = rows[:limit] if limit is not None else rows
    overlay_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    rendered_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        image_path = _resolve_image_path(row, image_root)
        annotation = row.get("region_annotation") if isinstance(row.get("region_annotation"), dict) else {}
        key = _row_key(row, index)
        overlay_path = overlay_dir / f"{index:04d}_{key}_overlay.jpg"
        crop_path = crop_dir / f"{index:04d}_{key}_crops.jpg"
        if (overlay_path.exists() or crop_path.exists()) and not overwrite:
            raise FileExistsError(f"rendered image already exists for row {key}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            overlay = draw_overlay(image, annotation)
            crops = crop_anchor_regions(image, annotation)
            crop_sheet = make_crop_sheet(crops)
            overlay.save(overlay_path, quality=95)
            crop_sheet.save(crop_path, quality=95)

        rendered_rows.append(
            {
                **row,
                "review_original_image": str(image_path),
                "overlay_image": str(overlay_path),
                "crop_sheet": str(crop_path),
            }
        )

    write_jsonl(output_jsonl, rendered_rows)
    _write_review_html(rendered_rows, review_html)
    return rendered_rows


def main() -> None:
    parser = ArgumentParser(description="Render HIC region annotations as overlays and crop sheets.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--crop-dir", type=Path, default=DEFAULT_CROP_DIR)
    parser.add_argument("--review-html", type=Path, default=DEFAULT_REVIEW_HTML)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    rendered = render_rows(
        rows,
        output_jsonl=args.output_jsonl,
        overlay_dir=args.overlay_dir,
        crop_dir=args.crop_dir,
        review_html=args.review_html,
        image_root=args.image_root,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"[render-hic-regions] rendered rows={len(rendered)}")
    print(f"[render-hic-regions] output_jsonl={args.output_jsonl}")
    print(f"[render-hic-regions] review_html={args.review_html}")


if __name__ == "__main__":
    main()
