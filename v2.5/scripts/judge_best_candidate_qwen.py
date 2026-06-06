#!/usr/bin/env python
from __future__ import annotations

import json
import math
import re
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl

JUDGE_PROMPT_TEMPLATE = """You are a strict evaluator for image-specific humorous caption candidates.

Pick the single best candidate as a humorous image caption. Be conservative. Most machine-generated captions are generic, weakly grounded, or only mildly funny. Do not reward a caption just because it is fluent.

Gold captions are weak references only; many different humorous captions can be correct. Prefer the visible image over the gold text.

Score only the best candidate. Use the full 1-5 scale:
- 1 = bad / unusable
- 2 = weak, generic, off-image, or barely humorous
- 3 = acceptable but not strong
- 4 = good, usable, image-specific, natural, and clearly somewhat funny
- 5 = excellent and rare; strongly image-specific and genuinely funny

Dimension rules:
- image_specific: penalize generic dialogue or captions that could fit many images.
- naturalness: penalize broken English, odd translation, repetition, or unnatural wording.
- humor: penalize plain descriptions, forced jokes, or captions with no surprise/contrast.
- format: penalize explanations, lists, prefixes, multiple captions, or long text.
- overall: should reflect real usefulness as the final selected caption.

Return only valid JSON with this exact schema:
{{
  "best_index": 1,
  "scores": {{
    "image_specific": 1,
    "naturalness": 1,
    "humor": 1,
    "format": 1,
    "overall": 1
  }},
  "reason": "short reason under 12 words"
}}

Gold captions:
{gold_block}

Candidates:
{candidate_block}
"""


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def read_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("image_id") or row.get("image") or "")
            if key:
                keys.add(key)
    return keys


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"judge response did not contain JSON: {text[:300]}")
        return json.loads(match.group(0))


def clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, score))


def get_gold_captions(row: dict[str, Any]) -> list[str]:
    golds: list[str] = []
    for key in ("gold_caption", "gold"):
        value = row.get(key)
        if value:
            text = str(value).strip()
            if text and text not in golds:
                golds.append(text)
    for value in row.get("gold_captions") or []:
        text = str(value).strip()
        if text and text not in golds:
            golds.append(text)
    return golds


def build_gold_block(golds: list[str]) -> str:
    if not golds:
        return "None provided."
    return "\n".join(f"- {' '.join(gold.split())}" for gold in golds[:5])


def build_candidate_block(candidates: list[str]) -> str:
    return "\n".join(f"{index}. {' '.join(str(candidate).split())}" for index, candidate in enumerate(candidates, start=1))


def qualified(scores: dict[str, int], thresholds: dict[str, int]) -> bool:
    return all(scores.get(name, 0) >= threshold for name, threshold in thresholds.items())


def judge_one(
    model: Any,
    processor: Any,
    image_path: str,
    golds: list[str],
    candidates: list[str],
    max_new_tokens: int,
) -> tuple[dict[str, Any], str]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        gold_block=build_gold_block(golds),
        candidate_block=build_candidate_block(candidates),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
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
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    new_tokens = generated_ids[:, inputs["input_ids"].shape[-1] :]
    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return extract_json(decoded), decoded


def normalize_judgment(judgment: dict[str, Any], candidates: list[str], thresholds: dict[str, int]) -> dict[str, Any]:
    try:
        best_index = int(judgment.get("best_index", 0) or 0)
    except (TypeError, ValueError):
        best_index = 0
    if best_index < 1 or best_index > len(candidates):
        best_index = 0
    raw_scores = judgment.get("scores") or {}
    scores = {
        "image_specific": clamp_score(raw_scores.get("image_specific")),
        "naturalness": clamp_score(raw_scores.get("naturalness")),
        "humor": clamp_score(raw_scores.get("humor")),
        "format": clamp_score(raw_scores.get("format")),
        "overall": clamp_score(raw_scores.get("overall")),
    }
    best_candidate = candidates[best_index - 1] if best_index else ""
    return {
        "best_index": best_index,
        "best_candidate": best_candidate,
        "scores": scores,
        "qualified": bool(best_index and qualified(scores, thresholds)),
        "reason": str(judgment.get("reason", ""))[:160],
    }


def summarize(output_jsonl: Path, summary_json: Path, thresholds: dict[str, int]) -> dict[str, Any]:
    rows = read_jsonl(output_jsonl) if output_jsonl.exists() else []
    scored = [row for row in rows if row.get("best_index")]
    failures = [row for row in rows if row.get("judge_failed")]
    pass_rows = [row for row in scored if row.get("qualified")]
    metrics = ["image_specific", "naturalness", "humor", "format", "overall"]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary: dict[str, Any] = {
        "output_jsonl": str(output_jsonl),
        "num_rows": len(rows),
        "num_scored_rows": len(scored),
        "num_failures": len(failures),
        "qualification_thresholds": thresholds,
        "qualified_count": len(pass_rows),
        "qualified_rate": len(pass_rows) / len(scored) if scored else 0.0,
        "avg_scores": {
            metric: mean([float(row.get("scores", {}).get(metric, 0)) for row in scored]) for metric in metrics
        },
        "score_distributions": {
            metric: {str(score): count for score, count in sorted(Counter(row.get("scores", {}).get(metric, 0) for row in scored).items())}
            for metric in metrics
        },
        "best_index_distribution": {
            str(index): count for index, count in sorted(Counter(row.get("best_index", 0) for row in scored).items())
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[judge-best] saved summary to {summary_json}")
    return summary


def run_judge(
    config_path: Path,
    input_jsonl: Path,
    output_jsonl: Path,
    summary_json: Path,
    judge_model: str | None,
    limit: int | None,
    offset: int,
    max_candidates: int,
    max_new_tokens: int,
    log_every: int,
    resume: bool,
    thresholds: dict[str, int],
) -> None:
    config = load_config(config_path)
    model_name = judge_model or config["model"]["model_name"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=config["model"].get("torch_dtype", "auto"),
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=config["model"].get("trust_remote_code", True))
    model.eval()

    rows = read_jsonl(input_jsonl)
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]

    if not resume and output_jsonl.exists():
        output_jsonl.unlink()
    processed_keys = read_existing_keys(output_jsonl) if resume else set()
    print(f"[judge-best] rows_to_consider={len(rows)} already_done={len(processed_keys)} model={model_name}")

    failures = 0
    written = 0
    for local_index, row in enumerate(rows, start=1):
        key = str(row.get("image_id") or row.get("image") or "")
        if resume and key in processed_keys:
            continue
        candidates = [str(candidate).strip() for candidate in (row.get("candidates") or []) if str(candidate).strip()]
        candidates = candidates[:max_candidates]
        if not candidates:
            output_row = {
                "image": row.get("image"),
                "image_id": row.get("image_id"),
                "gold_caption": row.get("gold_caption"),
                "gold_captions": get_gold_captions(row),
                "prompt": row.get("prompt"),
                "num_candidates": 0,
                "best_index": 0,
                "best_candidate": "",
                "scores": {"image_specific": 0, "naturalness": 0, "humor": 0, "format": 0, "overall": 0},
                "qualified": False,
                "judge_failed": True,
                "error": "no candidates",
            }
            failures += 1
            append_jsonl(output_jsonl, output_row)
            written += 1
            continue
        try:
            judgment, raw_response = judge_one(
                model=model,
                processor=processor,
                image_path=str(row["image"]),
                golds=get_gold_captions(row),
                candidates=candidates,
                max_new_tokens=max_new_tokens,
            )
            normalized = normalize_judgment(judgment, candidates, thresholds)
            judge_failed = False
            error = ""
        except Exception as exc:
            failures += 1
            raw_response = ""
            normalized = {
                "best_index": 0,
                "best_candidate": "",
                "scores": {"image_specific": 0, "naturalness": 0, "humor": 0, "format": 0, "overall": 0},
                "qualified": False,
                "reason": "",
            }
            judge_failed = True
            error = str(exc)
            print(f"[judge-best] failed row={offset + local_index - 1} image={row.get('image')}: {exc}")
        output_row = {
            "image": row.get("image"),
            "image_id": row.get("image_id"),
            "gold_caption": row.get("gold_caption"),
            "gold_captions": get_gold_captions(row),
            "prompt": row.get("prompt"),
            "num_candidates": len(candidates),
            **normalized,
            "judge_failed": judge_failed,
            "error": error,
            "raw_judge_response": raw_response,
        }
        append_jsonl(output_jsonl, output_row)
        written += 1
        if local_index % log_every == 0:
            print(f"[judge-best] processed {local_index}/{len(rows)} written={written} failures={failures}")

    summarize(output_jsonl, summary_json, thresholds)


def main() -> None:
    parser = ArgumentParser(description="Use Qwen2.5-VL to choose and score the best humorous caption candidate per image.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sft.yaml"))
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pass-image-specific", type=int, default=3)
    parser.add_argument("--pass-naturalness", type=int, default=3)
    parser.add_argument("--pass-humor", type=int, default=3)
    parser.add_argument("--pass-format", type=int, default=4)
    parser.add_argument("--pass-overall", type=int, default=4)
    args = parser.parse_args()
    thresholds = {
        "image_specific": args.pass_image_specific,
        "naturalness": args.pass_naturalness,
        "humor": args.pass_humor,
        "format": args.pass_format,
        "overall": args.pass_overall,
    }
    run_judge(
        config_path=args.config,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        judge_model=args.judge_model,
        limit=args.limit,
        offset=args.offset,
        max_candidates=args.max_candidates,
        max_new_tokens=args.max_new_tokens,
        log_every=args.log_every,
        resume=args.resume,
        thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
