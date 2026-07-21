#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.hic_region_annotations import (
    ANNOTATION_VERSION,
    CORE_VIEWPOINTS,
    normalize_bbox_xyxy_norm,
    normalize_point_xy_norm,
    normalize_region_annotation,
)
from src.analysis.humor_context import extract_json, get_model_device, load_qwen_vl
from src.utils.io import read_jsonl


DEFAULT_INPUT_JSONL = Path("outputs/annotations/hic_region_annotation_subset_800.jsonl")
DEFAULT_OUTPUT_JSONL = Path("outputs/annotations/hic_region_annotations_800.jsonl")
DEFAULT_MODEL = "/home/zhang.jianqiao/models/Qwen2.5-VL-7B-Instruct"
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1024 * 28 * 28
PROMPT_VERSION = "hic-region-v1-qwen7b"
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 1024


def _clean_text(value: Any, max_chars: int = 600) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _compact_anchors(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    anchors: list[dict[str, str]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        label = _clean_text(item.get("label"), max_chars=120)
        if not label:
            continue
        anchors.append(
            {
                "id": _clean_text(item.get("id"), max_chars=40) or f"a{index}",
                "label": label,
                "role": _clean_text(item.get("role"), max_chars=160),
                "evidence": _clean_text(item.get("evidence"), max_chars=180),
            }
        )
        if len(anchors) >= 4:
            break
    return anchors


def _analysis_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    analysis = row.get("analysis") if isinstance(row.get("analysis"), dict) else {}
    return {
        "humor_type": _clean_text(analysis.get("humor_type"), max_chars=80),
        "humor_point": _clean_text(analysis.get("humor_point"), max_chars=360),
        "visual_anchors": _compact_anchors(analysis.get("visual_anchors")),
        "required_viewpoints": [str(item) for item in (analysis.get("required_viewpoints") or [])][:4],
        "primary_viewpoint": _clean_text(analysis.get("primary_viewpoint"), max_chars=80) or "full_image",
        "uncertainty": _clean_text(analysis.get("uncertainty"), max_chars=300),
    }


def build_region_prompt(row: dict[str, Any]) -> str:
    compact_analysis = _analysis_for_prompt(row)
    schema = {
        "annotation_version": ANNOTATION_VERSION,
        "viewpoint_set": list(CORE_VIEWPOINTS),
        "primary_viewpoint": "one fixed viewpoint name from the compact analysis",
        "required_viewpoints": ["one to four fixed viewpoint names from the compact analysis"],
        "needs_full_image": False,
        "anchors": [
            {
                "id": "a1",
                "label": "visible anchor label from the existing analysis",
                "role": "how this localized anchor supports the existing humor analysis",
                "source_anchor_id": "visual anchor id from the input analysis, or empty string",
                "viewpoint": "one fixed viewpoint name",
                "region": {
                    "kind": "bbox|point|full_image|nonlocalizable",
                    "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
                    "point_xy_norm": [0.5, 0.5],
                    "confidence": "low|medium|high",
                    "evidence": "short visible evidence for the localization",
                },
            }
        ],
        "relations": [
            {
                "subject": "anchor id or label",
                "predicate": "short visible relation",
                "object": "anchor id or label",
                "confidence": "low|medium|high",
            }
        ],
        "annotation_confidence": "low|medium|high",
        "uncertainty": "nonempty uncertainty string for any nonlocalizable or full-image clue",
    }
    return (
        "You are a conservative region annotator for a humorous image dataset.\n\n"
        "You will receive the original image plus compact analysis from a previous humor analyst. "
        "Your job is to localize the existing humor analysis in the image. Do not invent a new joke. "
        "No final humorous caption.\n\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Approved annotation schema version: {ANNOTATION_VERSION}\n\n"
        "Fixed 8 viewpoints:\n"
        + "\n".join(f"- {viewpoint}" for viewpoint in CORE_VIEWPOINTS)
        + "\n\n"
        "Compact existing analysis:\n"
        + json.dumps(compact_analysis, ensure_ascii=False, indent=2)
        + "\n\n"
        "Return only valid JSON with this exact approved schema. Use top-level anchors and nested region objects:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n\n"
        "Rules:\n"
        "- Include at most 4 anchors and at most 4 relations.\n"
        "- Use only the fixed viewpoint names listed above.\n"
        "- Localize the given humor_point and visual_anchors; do not create a different humor interpretation.\n"
        "- Use normalized xyxy boxes as [x1, y1, x2, y2] in [0, 1], with x2 > x1 and y2 > y1.\n"
        "- Use normalized points as [x, y] in [0, 1].\n"
        "- Use null for both bbox_xyxy_norm and point_xy_norm only for nonlocalizable or full-image clues.\n"
        "- If any anchor uses null coordinates, include a nonempty top-level uncertainty string explaining why.\n"
        "- Do not use markdown, comments, trailing commas, or extra top-level keys.\n"
        "- Return JSON only."
    )


def validate_raw_region_annotation(value: dict[str, Any]) -> None:
    anchors = value.get("anchors")
    if not isinstance(anchors, list):
        raise ValueError("region annotation must contain an anchors list")
    for index, anchor in enumerate(anchors[:4], start=1):
        if not isinstance(anchor, dict):
            raise ValueError(f"anchor {index} must be a JSON object")
        region = anchor.get("region")
        if not isinstance(region, dict):
            raise ValueError(f"anchor {anchor.get('id') or index} must contain a region object")
        kind = _clean_text(region.get("kind"), max_chars=40).lower()
        anchor_id = _clean_text(anchor.get("id"), max_chars=40) or str(index)
        bbox_raw = region.get("bbox_xyxy_norm")
        point_raw = region.get("point_xy_norm")
        bbox = normalize_bbox_xyxy_norm(bbox_raw)
        point = normalize_point_xy_norm(point_raw)
        if bbox is None:
            if kind not in {"full_image", "nonlocalizable"} or bbox_raw is not None or point_raw is not None:
                raise ValueError(f"invalid region coordinates for anchor {anchor_id}")
        elif point is None and point_raw not in (None, [], ""):
            raise ValueError(f"invalid region coordinates for anchor {anchor_id}")


def parse_region_annotation(
    raw: str | dict[str, Any],
    *,
    primary_viewpoint: str,
    required_viewpoints: list[str],
) -> dict[str, Any]:
    parsed = extract_json(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("region annotation must be a JSON object")
    validate_raw_region_annotation(parsed)
    return normalize_region_annotation(
        parsed,
        primary_viewpoint=primary_viewpoint,
        required_viewpoints=required_viewpoints,
    )


def build_region_annotation_meta(
    *,
    model_name: str,
    prompt: str,
    input_jsonl: str | Path,
    min_pixels: int | None,
    max_pixels: int | None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "temperature": TEMPERATURE,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "source_path": str(input_jsonl),
        "created_at": timestamp or datetime.now(timezone.utc).isoformat(),
    }


def build_success_output_row(
    *,
    row: dict[str, Any],
    key: str,
    index: int,
    annotation: dict[str, Any],
    raw: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        **row,
        "row_key": key,
        "source_index_in_region_run": index,
        "failed": False,
        "region_annotation": annotation,
        "raw_region_response": raw,
        "region_parse_error": "",
        "region_annotation_meta": meta,
    }


def build_failure_output_row(
    *,
    row: dict[str, Any],
    key: str,
    index: int,
    error: str,
    raw: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        **row,
        "row_key": key,
        "source_index_in_region_run": index,
        "failed": True,
        "region_annotation": None,
        "raw_region_response": raw,
        "region_parse_error": error or "region annotation failed",
        "region_annotation_meta": meta,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("row_key")
            if key:
                keys.add(str(key))
    return keys


def row_key(row: dict[str, Any], index: int) -> str:
    key = _clean_text(row.get("row_key"), max_chars=240)
    if key:
        return key
    image_id = _clean_text(row.get("image_id"), max_chars=120) or _clean_text(row.get("image"), max_chars=120)
    return f"{image_id}::{index}"


def query_gpu_free_mb(gpu_index: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def wait_for_gpu_memory(
    min_free_mb: int | None,
    gpu_index: int,
    stable_checks: int,
    check_seconds: int,
) -> None:
    if min_free_mb is None or min_free_mb <= 0:
        return
    stable = 0
    print(
        "[regions] "
        f"waiting for gpu={gpu_index} free memory >= {min_free_mb} MiB "
        f"for {stable_checks} consecutive checks"
    )
    while stable < stable_checks:
        free_mb = query_gpu_free_mb(gpu_index)
        stable = stable + 1 if free_mb >= min_free_mb else 0
        print(f"[regions] gpu={gpu_index} free_memory={free_mb} MiB stable={stable}/{stable_checks}")
        if stable < stable_checks:
            time.sleep(check_seconds)


def generate_text(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    image_path: Path,
    prompt: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(get_model_device(model))
    with __import__("torch").no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def annotate_rows(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    model_name: str,
    limit: int | None,
    resume: bool,
    overwrite: bool,
    skip_existing: bool,
    wait_gpu_free_mb: int | None,
    wait_gpu_index: int,
    wait_gpu_stable_checks: int,
    wait_gpu_check_seconds: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> None:
    if overwrite and output_jsonl.exists():
        output_jsonl.unlink()

    rows = read_jsonl(input_jsonl)
    selected = rows[:limit] if limit is not None else rows
    done = existing_keys(output_jsonl) if resume or skip_existing else set()

    wait_for_gpu_memory(
        min_free_mb=wait_gpu_free_mb,
        gpu_index=wait_gpu_index,
        stable_checks=wait_gpu_stable_checks,
        check_seconds=wait_gpu_check_seconds,
    )
    model, processor, process_vision_info = load_qwen_vl(
        model_name=model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    skipped_existing = 0
    failures = 0
    for index, row in enumerate(tqdm(selected, desc="annotating humor regions", dynamic_ncols=True)):
        key = row_key(row, index)
        if key in done:
            skipped_existing += 1
            continue

        prompt = build_region_prompt(row)
        meta = build_region_annotation_meta(
            model_name=model_name,
            prompt=prompt,
            input_jsonl=input_jsonl,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        analysis = _analysis_for_prompt(row)
        primary_viewpoint = analysis["primary_viewpoint"]
        required_viewpoints = analysis["required_viewpoints"] or [primary_viewpoint]
        image_path = Path(str(row.get("image") or "")).expanduser()

        raw = ""
        parse_error = None
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"missing image: {image_path}")
            raw = generate_text(
                model=model,
                processor=processor,
                process_vision_info=process_vision_info,
                image_path=image_path,
                prompt=prompt,
            )
            annotation = parse_region_annotation(
                raw,
                primary_viewpoint=primary_viewpoint,
                required_viewpoints=required_viewpoints,
            )
            output_row = build_success_output_row(
                row=row,
                key=key,
                index=index,
                annotation=annotation,
                raw=raw,
                meta=meta,
            )
            done.add(key)
        except Exception as exc:
            failures += 1
            parse_error = str(exc)
            output_row = build_failure_output_row(
                row=row,
                key=key,
                index=index,
                error=parse_error or "region annotation failed",
                raw=raw,
                meta=meta,
            )
            print(f"[regions] failed {key}: {exc}")
        append_jsonl(output_jsonl, output_row)

    print(f"[regions] saved to {output_jsonl}")
    print(f"[regions] rows={len(selected)} skipped_existing={skipped_existing} failures={failures}")


def main() -> None:
    parser = ArgumentParser(description="Annotate HIC humor-analysis rows with localized image regions using Qwen-VL.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--wait-gpu-free-mb", type=int, default=None)
    parser.add_argument("--wait-gpu-index", type=int, default=0)
    parser.add_argument("--wait-gpu-stable-checks", type=int, default=2)
    parser.add_argument("--wait-gpu-check-seconds", type=int, default=30)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    args = parser.parse_args()

    annotate_rows(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        model_name=args.model_name,
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        wait_gpu_free_mb=args.wait_gpu_free_mb,
        wait_gpu_index=args.wait_gpu_index,
        wait_gpu_stable_checks=args.wait_gpu_stable_checks,
        wait_gpu_check_seconds=args.wait_gpu_check_seconds,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )


if __name__ == "__main__":
    main()
