#!/usr/bin/env python3
"""Auxiliary 1--5 humor/grounding scorer for Best-of-N candidate pools.

This intentionally uses a smaller output contract than the general candidate
judge. Candidate chunks are validated strictly; omitted or out-of-range scores
abort the run instead of being converted into zeros.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.judge_sft_candidates_qwen import extract_json, get_model_device, load_config
from src.utils.io import read_jsonl, write_jsonl

PROMPT = """You are a strict evaluator of humorous captions for one image.

Score every candidate independently on these two 1--5 integer scales.

Humor:
1 = not a joke or unusable
2 = weak, generic, or forced
3 = mildly funny but ordinary
4 = clearly funny and usable
5 = exceptionally funny and original

Grounding:
1 = unrelated to or contradicted by the image
2 = mostly generic, with weak image connection
3 = plausibly connected to the image
4 = clearly uses image-specific details
5 = precise visual fit with no hallucinated detail

Do not rank candidates against each other. Do not use 0. Return exactly
{candidate_count} score objects, one for every index. Return JSON only:
{{"scores":[{{"index":<integer>,"humor":<integer 1-5>,"grounding":<integer 1-5>}}]}}

Gold caption is a weak reference, not the only correct answer:
{gold_caption}

Candidates:
{candidate_block}
"""


def candidate_block(candidates: list[str]) -> str:
    return "\n".join(f"{index}. {' '.join(text.split())}" for index, text in enumerate(candidates, 1))


def normalize_scores(payload: dict[str, Any], expected: int) -> list[dict[str, int]]:
    items = payload.get("scores")
    if not isinstance(items, list):
        raise ValueError("judge JSON has no scores list")
    by_index: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if index in by_index:
            raise ValueError(f"duplicate candidate index {index}")
        by_index[index] = item
    indices = set(by_index)
    one_based = set(range(1, expected + 1))
    zero_based = set(range(expected))
    if indices == zero_based:
        by_index = {index + 1: item for index, item in by_index.items()}
    elif indices != one_based and len(indices) == expected:
        ordered_indices = sorted(indices)
        if ordered_indices == list(range(ordered_indices[0], ordered_indices[0] + expected)):
            by_index = {rank: by_index[index] for rank, index in enumerate(ordered_indices, 1)}
        else:
            raise ValueError(
                f"judge candidate indices must form a complete consecutive set; received {ordered_indices}"
            )
    elif indices != one_based:
        raise ValueError(
            f"judge candidate indices must be exactly 1..{expected} or 0..{expected - 1}; "
            f"received {sorted(indices)}"
        )
    result = []
    for index in range(1, expected + 1):
        if index not in by_index:
            raise ValueError(f"judge omitted candidate index {index}/{expected}")
        item = by_index[index]
        try:
            humor = int(item.get("humor"))
            grounding = int(item.get("grounding"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate index {index} has non-integer scores") from exc
        if not (1 <= humor <= 5 and 1 <= grounding <= 5):
            raise ValueError(
                f"candidate index {index} has scores outside [1, 5]: "
                f"humor={humor}, grounding={grounding}"
            )
        result.append({"index": index, "humor": humor, "grounding": grounding})
    return result


def parse_score_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = extract_json(text)
    if isinstance(value, list):
        return {"scores": value}
    if not isinstance(value, dict):
        raise ValueError(f"judge JSON must be an object or list, received {type(value).__name__}")
    return value


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def judge_chunk(
    model: Any,
    processor: Any,
    image_path: str,
    gold_caption: str,
    candidates: list[str],
    max_new_tokens: int,
) -> tuple[list[dict[str, int]], str]:
    base_prompt = PROMPT.format(
        candidate_count=len(candidates),
        gold_caption=gold_caption,
        candidate_block=candidate_block(candidates),
    )
    decoded = ""
    last_error: ValueError | None = None
    for attempt in range(2):
        prompt = base_prompt
        if attempt:
            prompt += (
                "\nIMPORTANT FORMAT CORRECTION: Return one score object per candidate in input order. "
                f"Indices must be exactly 1 through {len(candidates)}, each used once."
            )
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(get_model_device(model))
        with torch.no_grad():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
        decoded = processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        try:
            return normalize_scores(parse_score_payload(decoded), len(candidates)), decoded
        except ValueError as exc:
            last_error = exc
            print(f"[humor-judge] invalid chunk response; retry={attempt + 1}/2: {exc}")
    assert last_error is not None
    raise ValueError(f"{last_error}; raw judge response={decoded[:1200]!r}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true", help="Reuse complete rows already stored in the output JSONL.")
    args = parser.parse_args()
    if args.candidate_batch_size < 1:
        raise ValueError("--candidate-batch-size must be positive")

    config = load_config(args.config)
    model_cfg = config["model"]
    model_name = model_cfg["model_name"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=model_cfg.get("device_map", "auto"),
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=model_cfg.get("trust_remote_code", True))
    model.eval()

    rows = read_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    existing = read_jsonl(args.output_jsonl) if args.resume and args.output_jsonl.exists() else []
    completed = {
        str(row.get("image_id")): row
        for row in existing
        if len(row.get("judged_candidates") or []) >= args.max_candidates
    }
    outputs = []
    for row_number, row in enumerate(rows, 1):
        image_id = str(row.get("image_id"))
        if image_id in completed:
            outputs.append(completed[image_id])
            print(f"[humor-judge] reused {row_number}/{len(rows)} rows")
            continue
        candidates = [str(value).strip() for value in row.get("candidates", []) if str(value).strip()]
        candidates = candidates[: args.max_candidates]
        if not candidates:
            raise ValueError(f"row {row_number} has no candidates")
        judged = []
        raw_responses = []
        for start in range(0, len(candidates), args.candidate_batch_size):
            chunk = candidates[start : start + args.candidate_batch_size]
            chunk_scores, raw = judge_chunk(
                model,
                processor,
                str(row["image"]),
                str(row.get("gold_caption") or ""),
                chunk,
                args.max_new_tokens,
            )
            for item, candidate in zip(chunk_scores, chunk, strict=True):
                item["index"] += start
                item["candidate"] = candidate
            judged.extend(chunk_scores)
            raw_responses.append(raw)
        outputs.append(
            {
                "image": row.get("image"),
                "image_id": row.get("image_id"),
                "gold_caption": row.get("gold_caption"),
                "judged_candidates": judged,
                "raw_judge_responses": raw_responses,
                "judge_scale": "strict independent 1--5 humor and grounding",
            }
        )
        atomic_write_jsonl(args.output_jsonl, outputs)
        print(f"[humor-judge] processed {row_number}/{len(rows)} rows")
    atomic_write_jsonl(args.output_jsonl, outputs)
    print(f"[humor-judge] saved {len(outputs)} complete rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
