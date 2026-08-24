#!/usr/bin/env python
from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import yaml
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_jsonl

JUDGE_PROMPT_TEMPLATE = """You are evaluating candidate humorous captions for one image.

Criteria:
- image_specific: the caption should use visible image details or plausible visual context.
- naturalness: the caption should sound like a short natural caption, not broken or translated badly.
- humor: the caption should have a mild joke, irony, contrast, surprise, or playful interpretation.
- format: the caption should be one short caption only, with no explanation, no list, no markdown, and no prefix like "Caption:".

Gold caption is only a weak reference, not the only correct answer.

Return only valid JSON with this schema:
{{
  "best_index": 1,
  "candidates": [
    {{"index": 1, "image_specific": 1, "naturalness": 1, "humor": 1, "format": 1, "overall": 1, "reason": "short reason"}}
  ]
}}

Use integer scores from 1 to 5. Keep each reason under 12 words.
You must score all {candidate_count} candidates and return exactly {candidate_count} candidate objects.

Gold caption: {gold_caption}

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


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Judge response did not contain JSON: {text[:300]}")
    return json.loads(match.group(0))


def build_candidate_block(candidates: list[str]) -> str:
    lines = []
    for index, candidate in enumerate(candidates, start=1):
        one_line = " ".join(str(candidate).strip().split())
        lines.append(f"{index}. {one_line}")
    return "\n".join(lines)


def judge_one(
    model: Any,
    processor: Any,
    image_path: str,
    gold_caption: str,
    candidates: list[str],
    max_new_tokens: int,
) -> tuple[dict[str, Any], str]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        gold_caption=gold_caption,
        candidate_block=build_candidate_block(candidates),
        candidate_count=len(candidates),
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


def normalize_judgment(judgment: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
    scored = judgment.get("candidates") or []
    by_index = {int(item.get("index", 0)): item for item in scored if str(item.get("index", "")).isdigit()}
    normalized = []
    for index, candidate in enumerate(candidates, start=1):
        if index not in by_index:
            raise ValueError(f"judge omitted candidate index {index}/{len(candidates)}")
        item = by_index[index]
        scores = {
            key: int(item.get(key, 0) or 0)
            for key in ("image_specific", "naturalness", "humor", "format", "overall")
        }
        if any(value < 1 or value > 5 for value in scores.values()):
            raise ValueError(f"candidate index {index} has a score outside [1, 5]: {scores}")
        normalized.append(
            {
                "index": index,
                "candidate": candidate,
                **scores,
                "reason": str(item.get("reason", "")),
            }
        )
    best_index = int(judgment.get("best_index", 0) or 0)
    if best_index < 1 or best_index > len(candidates):
        best_index = max(normalized, key=lambda item: item["overall"], default={"index": 0})["index"]
    return {"best_index": best_index, "candidates": normalized}


def run_judge(
    config_path: Path,
    input_jsonl: Path,
    output_jsonl: Path,
    limit: int | None,
    max_candidates: int,
    max_new_tokens: int,
    candidate_batch_size: int,
) -> None:
    config = load_config(config_path)
    model_name = config["model"]["model_name"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=config["model"].get("torch_dtype", "auto"),
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=config["model"].get("trust_remote_code", True))
    model.eval()

    rows = read_jsonl(input_jsonl)
    if limit is not None:
        rows = rows[:limit]

    outputs = []
    failures = 0
    for row_index, row in enumerate(rows):
        candidates = [str(candidate).strip() for candidate in (row.get("candidates") or []) if str(candidate).strip()]
        candidates = candidates[:max_candidates]
        if not candidates:
            continue
        try:
            judged_candidates = []
            raw_responses = []
            for start in range(0, len(candidates), candidate_batch_size):
                chunk = candidates[start : start + candidate_batch_size]
                judgment, raw_response = judge_one(
                    model=model,
                    processor=processor,
                    image_path=str(row["image"]),
                    gold_caption=str(row.get("gold_caption") or ""),
                    candidates=chunk,
                    max_new_tokens=max_new_tokens,
                )
                chunk_normalized = normalize_judgment(judgment, chunk)
                for item in chunk_normalized["candidates"]:
                    item["index"] += start
                judged_candidates.extend(chunk_normalized["candidates"])
                raw_responses.append(raw_response)
            best = max(judged_candidates, key=lambda item: (item["overall"], item["humor"], -item["index"]))
            normalized = {"best_index": best["index"], "candidates": judged_candidates}
            raw_response = "\n\n--- candidate chunk ---\n\n".join(raw_responses)
        except Exception as exc:
            failures += 1
            raw_response = ""
            normalized = {"best_index": 0, "candidates": []}
            print(f"[judge] failed row={row_index} image={row.get('image')}: {exc}")
        outputs.append(
            {
                "image": row.get("image"),
                "image_id": row.get("image_id"),
                "gold_caption": row.get("gold_caption"),
                "prompt": row.get("prompt"),
                "best_index": normalized["best_index"],
                "best_candidate": candidates[normalized["best_index"] - 1] if normalized["best_index"] else "",
                "judged_candidates": normalized["candidates"],
                "raw_judge_response": raw_response,
            }
        )
        if (row_index + 1) % 10 == 0:
            print(f"[judge] processed {row_index + 1}/{len(rows)} rows")

    write_jsonl(output_jsonl, outputs)
    print(f"[judge] saved {len(outputs)} rows to {output_jsonl}; failures={failures}")


def main() -> None:
    parser = ArgumentParser(description="Use base Qwen2.5-VL as an optional judge for generated humorous caption candidates.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sft.yaml"))
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=8,
        help="Score candidates in small chunks so the judge cannot silently omit a long tail.",
    )
    args = parser.parse_args()
    run_judge(
        config_path=args.config,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        limit=args.limit,
        max_candidates=args.max_candidates,
        max_new_tokens=args.max_new_tokens,
        candidate_batch_size=args.candidate_batch_size,
    )


if __name__ == "__main__":
    main()
