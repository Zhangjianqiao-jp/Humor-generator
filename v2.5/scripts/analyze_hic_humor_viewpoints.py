#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.humor_context import extract_json, get_model_device, load_qwen_vl
from src.training.sft_dataset import extract_caption, extract_image_path
from src.utils.io import read_jsonl


DEFAULT_MODEL = "/home/zhang.jianqiao/models/Qwen2.5-VL-7B-Instruct"
DEFAULT_HIC_ROOT = "/home/zhang.jianqiao/datasets/hic-data"
PROMPT_VERSION = "gold-caption-minimal-viewpoint-v2"
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1024 * 28 * 28

HUMOR_TYPES = (
    "scale_contrast",
    "role_mismatch",
    "object_misuse",
    "action_pose_incongruity",
    "text_image_contrast",
    "expression_context_contrast",
    "foreground_background_misread",
    "knowledge_reference",
    "dialogue_or_nonvisual",
    "unclear_or_weak",
)

VIEWPOINTS = (
    "full_image",
    "object_crop",
    "relation_crop",
    "scale_reference_crop",
    "face_expression_crop",
    "text_region_crop",
    "foreground_background_view",
    "context_scene_view",
    "pose_action_view",
)

CONFIDENCE_VALUES = {"low", "medium", "high"}

PROMPT = """You are analyzing a humorous image-caption dataset.

You will receive an image and its dataset gold caption. Your job is to explain the humor target of THAT caption, not to invent a better caption.

Return only valid JSON with this exact schema:
{
  "literal_image_description": "one short sentence describing visible content needed for the joke",
  "gold_joke_explanation": "why the gold caption can be funny for this image",
  "humor_type": "scale_contrast|role_mismatch|object_misuse|action_pose_incongruity|text_image_contrast|expression_context_contrast|foreground_background_misread|knowledge_reference|dialogue_or_nonvisual|unclear_or_weak",
  "humor_point": "one concise description of the gold-caption humor point",
  "visual_anchors": [
    {
      "id": "a1",
      "label": "visible anchor label",
      "role": "why this anchor matters",
      "evidence": "visible evidence in the image"
    }
  ],
  "required_viewpoints": ["relation_crop"],
  "primary_viewpoint": "relation_crop",
  "needs_external_knowledge": false,
  "confidence": "low|medium|high",
  "uncertainty": "what is unclear, or empty string"
}

Allowed humor_type definitions:
- scale_contrast: size, quantity, or proportion contrast is the joke target.
- role_mismatch: an entity is read as occupying an unexpected social/object role.
- object_misuse: an object is used or interpreted in the wrong function.
- action_pose_incongruity: a visible pose or action conflicts with the expected scene.
- text_image_contrast: visible text/sign/menu/UI contrasts with the image content.
- expression_context_contrast: facial expression or emotional look contrasts with context.
- foreground_background_misread: foreground/background composition creates a misread.
- knowledge_reference: the caption depends on culture, named entities, memes, idioms, or external knowledge.
- dialogue_or_nonvisual: the caption is mostly dialogue/story and the visual evidence is weak.
- unclear_or_weak: the gold caption is too unclear, too generic, or not recoverable from image + caption.

Allowed required_viewpoints:
- full_image: only when the joke needs the whole composition and no smaller view would preserve the joke.
- object_crop: a close view of one key object/entity is enough.
- relation_crop: a crop containing two or more anchors and their relation is enough.
- scale_reference_crop: a crop showing the size reference and the small/large object is enough.
- face_expression_crop: a face/expression crop is enough.
- text_region_crop: visible text, sign, menu, subtitle, UI, or label is enough.
- foreground_background_view: foreground/background alignment or depth relation is enough.
- context_scene_view: broader scene context is needed, but the entire image is not necessarily needed.
- pose_action_view: body pose, gesture, or action crop is enough.

Viewpoint decision policy:
- Choose the MINIMUM visual viewpoint(s) needed to understand why the gold caption is funny.
- Do not choose full_image as a safe default.
- Prefer a specific crop/viewpoint whenever it preserves the humor target.
- Use full_image only if cropping would remove essential context, global layout, or whole-scene composition.
- For visible text/sign/UI jokes, use text_region_crop, plus relation_crop or context_scene_view only if needed.
- For size/proportion jokes, use scale_reference_crop.
- For face/emotion jokes, use face_expression_crop.
- For pose/action jokes, use pose_action_view.
- For foreground/background illusion jokes, use foreground_background_view.
- For object function or misuse jokes, use object_crop if one object is enough; otherwise use relation_crop.
- For role mismatch jokes, use relation_crop when the role comes from an interaction; use context_scene_view when scene setting is essential.
- If the gold caption is mostly dialogue or not visually grounded, choose the closest minimal visual evidence and mark confidence low or medium.

Rules:
- Analyze the gold caption, not your own preferred joke.
- If the gold caption cannot be grounded in the image, use humor_type "dialogue_or_nonvisual" or "unclear_or_weak".
- Use required_viewpoints only from the allowed list.
- Include "full_image" only when the overall scene is truly necessary.
- If you choose full_image, explain in uncertainty why a smaller viewpoint is insufficient.
- Use at most 4 visual_anchors and at most 4 required_viewpoints.
- Do not write a new caption.
- Use English.

Gold caption:
{caption}

Return JSON only."""


def caption_hash(caption: str) -> str:
    return hashlib.sha1(caption.encode("utf-8")).hexdigest()[:12]


def clean_text(value: Any, max_chars: int = 600) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def row_key(row: dict[str, Any]) -> str:
    return f"{row['image_id']}::{caption_hash(row['gold_caption'])}"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)


def load_rows_from_jsonl(path: Path, dedupe_image: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(read_jsonl(path)):
        image = extract_image_path(raw)
        caption = extract_caption(raw)
        if not image or not caption:
            continue
        image_path = Path(str(image)).expanduser()
        score = safe_float((raw.get("meta") or {}).get("score"), default=0.0)
        rows.append(
            {
                "image": str(image_path),
                "image_id": str(raw.get("image_id") or image_path.stem),
                "gold_caption": clean_text(caption, max_chars=500),
                "score": score,
                "source_index": index,
                "source": str(path),
            }
        )
    return dedupe_rows(rows) if dedupe_image else rows


def load_rows_from_hic_root(root: Path, dedupe_image: bool) -> list[dict[str, Any]]:
    data_csv = root / "oxford_hic_data.csv"
    image_info_csv = root / "oxford_hic_image_info.csv"
    if not data_csv.exists():
        raise FileNotFoundError(f"Missing HIC caption CSV: {data_csv}")
    if not image_info_csv.exists():
        raise FileNotFoundError(f"Missing HIC image info CSV: {image_info_csv}")

    image_paths: dict[str, str] = {}
    with image_info_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = clean_text(row.get("image_id"), max_chars=120)
            image_path = clean_text(row.get("image_path"), max_chars=1000)
            if image_id and image_path:
                image_paths[image_id] = image_path

    rows: list[dict[str, Any]] = []
    with data_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            image_id = clean_text(row.get("image_id"), max_chars=120)
            caption = clean_text(row.get("caption"), max_chars=500)
            image = image_paths.get(image_id)
            if not image_id or not caption or not image:
                continue
            rows.append(
                {
                    "image": image,
                    "image_id": image_id,
                    "gold_caption": caption,
                    "score": safe_float(row.get("funny_score"), default=0.0),
                    "source_index": index,
                    "source": str(root),
                }
            )
    return dedupe_rows(rows) if dedupe_image else rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_image: dict[str, dict[str, Any]] = {}
    for row in rows:
        current = best_by_image.get(row["image_id"])
        if current is None or row.get("score", 0.0) > current.get("score", 0.0):
            best_by_image[row["image_id"]] = row
    return sorted(best_by_image.values(), key=lambda item: item["source_index"])


def load_input_rows(input_path: Path, dedupe_image: bool) -> list[dict[str, Any]]:
    if input_path.is_dir():
        return load_rows_from_hic_root(input_path, dedupe_image=dedupe_image)
    return load_rows_from_jsonl(input_path, dedupe_image=dedupe_image)


def select_input_rows(rows: list[dict[str, Any]], limit: int | None, sample_seed: int | None) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return list(rows)
    if sample_seed is None:
        return rows[:limit]
    return random.Random(sample_seed).sample(rows, limit)


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
            if not row.get("failed") and row.get("row_key"):
                keys.add(str(row["row_key"]))
    return keys


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
        "[viewpoints] "
        f"waiting for gpu={gpu_index} free memory >= {min_free_mb} MiB "
        f"for {stable_checks} consecutive checks"
    )
    while stable < stable_checks:
        free_mb = query_gpu_free_mb(gpu_index)
        if free_mb >= min_free_mb:
            stable += 1
        else:
            stable = 0
        print(
            "[viewpoints] "
            f"gpu={gpu_index} free_memory={free_mb} MiB "
            f"stable={stable}/{stable_checks}"
        )
        if stable < stable_checks:
            time.sleep(check_seconds)


def generate_text(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
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
    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generation_kwargs["do_sample"] = False
    with __import__("torch").no_grad():
        generated = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def is_cuda_oom(exc: Exception) -> bool:
    text = repr(exc).lower()
    return "outofmemoryerror" in text or "cuda out of memory" in text or "out of memory" in text


def clean_choice(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = clean_text(value, max_chars=120).lower().replace("-", "_").replace(" ", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_")
    return text if text in allowed else default


def clean_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def clean_anchors(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    anchors: list[dict[str, str]] = []
    for index, item in enumerate(value[:4], start=1):
        if not isinstance(item, dict):
            continue
        label = clean_text(item.get("label"), max_chars=100)
        if not label:
            continue
        anchor_id = clean_text(item.get("id"), max_chars=20) or f"a{index}"
        anchors.append(
            {
                "id": anchor_id,
                "label": label,
                "role": clean_text(item.get("role"), max_chars=160),
                "evidence": clean_text(item.get("evidence"), max_chars=180),
            }
        )
    return anchors


def clean_viewpoints(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ["full_image"]
    result: list[str] = []
    for item in value:
        viewpoint = clean_choice(item, VIEWPOINTS, default="")
        if viewpoint and viewpoint not in result:
            result.append(viewpoint)
        if len(result) >= 4:
            break
    return result or ["full_image"]


def normalize_analysis(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value or {}
    viewpoints = clean_viewpoints(value.get("required_viewpoints"))
    primary = clean_choice(value.get("primary_viewpoint"), VIEWPOINTS, default=viewpoints[0])
    if primary not in viewpoints:
        viewpoints.insert(0, primary)
    return {
        "literal_image_description": clean_text(value.get("literal_image_description"), max_chars=320),
        "gold_joke_explanation": clean_text(value.get("gold_joke_explanation"), max_chars=500),
        "humor_type": clean_choice(value.get("humor_type"), HUMOR_TYPES, default="unclear_or_weak"),
        "humor_point": clean_text(value.get("humor_point"), max_chars=320),
        "visual_anchors": clean_anchors(value.get("visual_anchors")),
        "required_viewpoints": viewpoints[:4],
        "primary_viewpoint": primary,
        "needs_external_knowledge": clean_bool(value.get("needs_external_knowledge")),
        "confidence": clean_choice(value.get("confidence"), ("low", "medium", "high"), default="low"),
        "uncertainty": clean_text(value.get("uncertainty"), max_chars=300),
    }


def analyze_rows(
    rows: list[dict[str, Any]],
    output_jsonl: Path,
    model_name: str,
    limit: int | None,
    overwrite: bool,
    skip_existing: bool,
    device_map: str,
    torch_dtype: str,
    max_new_tokens: int,
    temperature: float,
    min_pixels: int | None,
    max_pixels: int | None,
    continue_on_oom: bool,
    wait_gpu_free_mb: int | None,
    wait_gpu_index: int,
    wait_gpu_stable_checks: int,
    wait_gpu_check_seconds: int,
) -> None:
    selected = rows[:limit] if limit is not None else rows
    if overwrite and output_jsonl.exists():
        output_jsonl.unlink()
    done = existing_keys(output_jsonl) if skip_existing else set()

    wait_for_gpu_memory(
        min_free_mb=wait_gpu_free_mb,
        gpu_index=wait_gpu_index,
        stable_checks=wait_gpu_stable_checks,
        check_seconds=wait_gpu_check_seconds,
    )

    model, processor, process_vision_info = load_qwen_vl(
        model_name=model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    failures = 0
    skipped_existing = 0
    skipped_missing = 0
    parse_errors = 0
    stopped_on_oom = False
    for index, row in enumerate(tqdm(selected, desc="analyzing humor viewpoints", dynamic_ncols=True)):
        key = row_key(row)
        if key in done:
            skipped_existing += 1
            continue
        image_path = Path(row["image"]).expanduser()
        if not image_path.exists():
            skipped_missing += 1
            append_jsonl(
                output_jsonl,
                {
                    **row,
                    "row_key": key,
                    "source_index_in_run": index,
                    "failed": True,
                    "error": f"missing image: {image_path}",
                },
            )
            continue
        prompt = PROMPT.replace("{caption}", row["gold_caption"])
        oom_error = False
        try:
            raw = generate_text(
                model=model,
                processor=processor,
                process_vision_info=process_vision_info,
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            parse_error = None
            try:
                parsed = extract_json(raw)
            except Exception as exc:
                parsed = {}
                parse_error = str(exc)
                parse_errors += 1
            analysis = normalize_analysis(parsed)
            output_row = {
                **row,
                "row_key": key,
                "source_index_in_run": index,
                "analysis": analysis,
                "raw_response": raw,
                "parse_error": parse_error,
                "analyzer_model": model_name,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
                "temperature": temperature,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        except Exception as exc:
            failures += 1
            oom_error = is_cuda_oom(exc)
            output_row = {
                **row,
                "row_key": key,
                "source_index_in_run": index,
                "failed": True,
                "error": repr(exc),
                "analyzer_model": model_name,
                "prompt_version": PROMPT_VERSION,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
            print(f"[viewpoints] failed {row.get('image_id')}: {exc}")
        append_jsonl(output_jsonl, output_row)
        if not output_row.get("failed"):
            done.add(key)
        if oom_error and not continue_on_oom:
            stopped_on_oom = True
            print(
                "[viewpoints] stopping after CUDA OOM. Free GPU memory or lower --max-pixels, "
                "then resume without --overwrite."
            )
            break

    print(f"[viewpoints] saved to {output_jsonl}")
    print(
        "[viewpoints] "
        f"rows={len(selected)} skipped_existing={skipped_existing} "
        f"skipped_missing={skipped_missing} failures={failures} "
        f"parse_errors={parse_errors} stopped_on_oom={stopped_on_oom}"
    )


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * math.log2(p)
    return value


def summarize(output_jsonl: Path, summary_json: Path, report_md: Path, min_type_count: int = 1) -> None:
    rows: list[dict[str, Any]] = []
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("failed"):
                continue
            rows.append(row)

    type_counts: Counter[str] = Counter()
    primary_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    required_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    confidence_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    external_by_type: Counter[str] = Counter()
    examples_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_versions: Counter[str] = Counter()
    parse_errors = 0

    for row in rows:
        if row.get("parse_error"):
            parse_errors += 1
        prompt_versions[str(row.get("prompt_version") or "unknown")] += 1
        analysis = row.get("analysis") or {}
        humor_type = str(analysis.get("humor_type") or "unclear_or_weak")
        type_counts[humor_type] += 1
        primary = str(analysis.get("primary_viewpoint") or "full_image")
        primary_by_type[humor_type][primary] += 1
        for viewpoint in analysis.get("required_viewpoints") or []:
            required_by_type[humor_type][str(viewpoint)] += 1
        confidence_by_type[humor_type][str(analysis.get("confidence") or "low")] += 1
        if analysis.get("needs_external_knowledge"):
            external_by_type[humor_type] += 1
        if len(examples_by_type[humor_type]) < 5:
            examples_by_type[humor_type].append(
                {
                    "image_id": row.get("image_id"),
                    "gold_caption": row.get("gold_caption"),
                    "humor_point": analysis.get("humor_point"),
                    "primary_viewpoint": primary,
                    "required_viewpoints": analysis.get("required_viewpoints"),
                    "confidence": analysis.get("confidence"),
                }
            )

    by_type: dict[str, Any] = {}
    for humor_type, count in type_counts.most_common():
        if count < min_type_count:
            continue
        primary_counts = primary_by_type[humor_type]
        required_counts = required_by_type[humor_type]
        primary_total = sum(primary_counts.values())
        required_total = sum(required_counts.values())
        primary_sorted = primary_counts.most_common()
        required_sorted = required_counts.most_common()
        top1_share = primary_sorted[0][1] / primary_total if primary_sorted else 0.0
        top2_coverage = sum(count for _, count in primary_sorted[:2]) / primary_total if primary_sorted else 0.0
        required_top2 = sum(count for _, count in required_sorted[:2]) / required_total if required_sorted else 0.0
        by_type[humor_type] = {
            "count": count,
            "primary_viewpoints": dict(primary_sorted),
            "required_viewpoints": dict(required_sorted),
            "primary_top1_share": top1_share,
            "primary_top2_coverage": top2_coverage,
            "required_top2_coverage": required_top2,
            "primary_entropy": entropy(primary_counts),
            "confidence": dict(confidence_by_type[humor_type]),
            "external_knowledge_count": external_by_type[humor_type],
            "examples": examples_by_type[humor_type],
        }

    summary = {
        "source_jsonl": str(output_jsonl),
        "rows": len(rows),
        "parse_errors": parse_errors,
        "humor_type_counts": dict(type_counts.most_common()),
        "prompt_version_counts": dict(prompt_versions.most_common()),
        "viewpoint_set": list(VIEWPOINTS),
        "humor_type_set": list(HUMOR_TYPES),
        "by_type": by_type,
    }
    write_json(summary_json, summary)
    write_report(report_md, summary)
    print(f"[viewpoints] wrote {summary_json}")
    print(f"[viewpoints] wrote {report_md}")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# HIC Humor Viewpoint Analysis",
        "",
        f"Rows analyzed: {summary['rows']}",
        f"Parse errors: {summary['parse_errors']}",
        f"Prompt versions: {summary.get('prompt_version_counts', {})}",
        "",
        "## Humor Type Counts",
        "",
        "| humor_type | count |",
        "|---|---:|",
    ]
    for humor_type, count in summary["humor_type_counts"].items():
        lines.append(f"| {humor_type} | {count} |")
    lines.extend(["", "## Viewpoint Stability By Humor Type", ""])
    lines.append(
        "| humor_type | count | top primary viewpoint | top1 share | top2 coverage | entropy | top required viewpoints |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---|")
    for humor_type, info in summary["by_type"].items():
        primary = list(info["primary_viewpoints"].items())
        required = list(info["required_viewpoints"].items())
        top_primary = "none" if not primary else f"{primary[0][0]} ({primary[0][1]})"
        top_required = ", ".join(f"{name} ({count})" for name, count in required[:3]) or "none"
        lines.append(
            f"| {humor_type} | {info['count']} | {top_primary} | "
            f"{info['primary_top1_share']:.2f} | {info['primary_top2_coverage']:.2f} | "
            f"{info['primary_entropy']:.2f} | {top_required} |"
        )
    lines.extend(["", "## Reading The Stability Columns", ""])
    lines.extend(
        [
            "- `top1 share`: how often the most common primary viewpoint appears inside the humor type.",
            "- `top2 coverage`: how often the two most common primary viewpoints cover the type.",
            "- Low entropy and high top2 coverage mean viewpoints are stable for that humor type.",
            "- If a type has low coverage or high entropy, split the humor type more finely or avoid a fixed viewpoint rule.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(
        description="Analyze HIC gold-caption humor types and required visual viewpoints with Qwen2.5-VL-7B."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/sft_train.jsonl"),
        help="JSONL file or HIC root directory. HIC root should contain oxford_hic_data.csv and oxford_hic_image_info.csv.",
    )
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/analysis/hic_humor_viewpoints.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/hic_humor_viewpoint_summary.json"))
    parser.add_argument("--report-md", type=Path, default=Path("outputs/analysis/hic_humor_viewpoint_report.md"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--limit", type=int, default=200, help="Number of rows to analyze. Use 0 or a negative value for all rows.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="If set with --limit, analyze a reproducible random sample instead of the first rows.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--wait-gpu-free-mb", type=int, default=0)
    parser.add_argument("--wait-gpu-index", type=int, default=0)
    parser.add_argument("--wait-gpu-stable-checks", type=int, default=3)
    parser.add_argument("--wait-gpu-check-seconds", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--continue-on-oom", action="store_true")
    parser.add_argument("--no-dedupe-image", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Load and report input rows without loading the model.")
    parser.add_argument("--summarize-only", action="store_true", help="Only summarize an existing output JSONL.")
    parser.add_argument("--min-type-count", type=int, default=1)
    args = parser.parse_args()

    if args.summarize_only:
        summarize(
            output_jsonl=args.output_jsonl,
            summary_json=args.summary_json,
            report_md=args.report_md,
            min_type_count=args.min_type_count,
        )
        return

    limit = None if args.limit is not None and args.limit <= 0 else args.limit
    rows = load_input_rows(args.input, dedupe_image=not args.no_dedupe_image)
    loaded_count = len(rows)
    rows = select_input_rows(rows, limit=limit, sample_seed=args.sample_seed)
    preview_count = len(rows)
    print(
        "[viewpoints] "
        f"loaded_rows={loaded_count} selected_rows={preview_count} "
        f"dedupe_image={not args.no_dedupe_image} sample_seed={args.sample_seed} input={args.input}"
    )
    if rows:
        print(
            "[viewpoints] first_row="
            + json.dumps(
                {
                    "image_id": rows[0]["image_id"],
                    "score": rows[0].get("score"),
                    "caption": rows[0]["gold_caption"][:120],
                    "image": rows[0]["image"],
                },
                ensure_ascii=False,
            )
        )
    if args.prepare_only:
        return

    analyze_rows(
        rows=rows,
        output_jsonl=args.output_jsonl,
        model_name=args.model_name,
        limit=None,
        overwrite=args.overwrite,
        skip_existing=not args.no_skip_existing,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        continue_on_oom=args.continue_on_oom,
        wait_gpu_free_mb=args.wait_gpu_free_mb,
        wait_gpu_index=args.wait_gpu_index,
        wait_gpu_stable_checks=args.wait_gpu_stable_checks,
        wait_gpu_check_seconds=args.wait_gpu_check_seconds,
    )
    summarize(
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        report_md=args.report_md,
        min_type_count=args.min_type_count,
    )


if __name__ == "__main__":
    main()
