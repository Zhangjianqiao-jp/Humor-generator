#!/usr/bin/env python
from __future__ import annotations

import re
import sys
from argparse import ArgumentParser
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_inference import load_qwen_vl_lora_for_inference
from src.utils.io import read_jsonl, write_jsonl


def _extract_text(content: list[dict[str, Any]]) -> str:
    texts = [item.get("text", "") for item in content if item.get("type") == "text"]
    return "\n".join(text for text in texts if text).strip()


def _answer_caption(row: dict[str, Any]) -> str:
    return _extract_text(row["messages"][1]["content"])


def _prompt(row: dict[str, Any]) -> str:
    return _extract_text(row["messages"][0]["content"])


def _normalize(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _similarity(left: str, right: str) -> float:
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _messages(image: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _generate_caption(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    image: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = _messages(image, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated_ids[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def evaluate_lora_caption_match(
    input_jsonl: Path,
    model_name: str,
    adapter_dir: Path,
    fail_output_dir: Path,
    threshold: float,
    limit: int | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Path:
    if not 0 <= threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")

    rows = read_jsonl(input_jsonl)
    if limit is not None:
        rows = rows[:limit]

    model, processor, process_vision_info = load_qwen_vl_lora_for_inference(
        model_name=model_name,
        adapter_dir=adapter_dir,
    )

    total_score = 0
    failed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        answer = _answer_caption(row)
        prompt = _prompt(row)
        generated = _generate_caption(
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            image=row["image"],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        similarity = _similarity(generated, answer)
        score = int(similarity >= threshold)
        total_score += score

        if score == 0:
            failed_rows.append(
                {
                    "row_index": index,
                    "image": row["image"],
                    "image_id": row.get("image_id"),
                    "prompt": prompt,
                    "generated_caption": generated,
                    "answer_caption": answer,
                    "similarity": similarity,
                    "threshold": threshold,
                    "score": score,
                    "meta": row.get("meta", {}),
                }
            )

        done = index + 1
        print(
            f"[{done}/{len(rows)}] score={score} similarity={similarity:.3f} "
            f"total={total_score}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fail_path = fail_output_dir / f"{timestamp}_fail.jsonl"
    write_jsonl(fail_path, failed_rows)

    print("== Evaluation summary ==")
    print(f"input_jsonl: {input_jsonl}")
    print(f"adapter_dir: {adapter_dir}")
    print(f"threshold: {threshold:.2f}")
    print(f"total_examples: {len(rows)}")
    print(f"total_score: {total_score}")
    print(f"accuracy: {total_score / len(rows):.4f}" if rows else "accuracy: n/a")
    print(f"failed_examples: {len(failed_rows)}")
    print(f"fail_jsonl: {fail_path}")
    return fail_path


def main() -> None:
    parser = ArgumentParser(description="Evaluate LoRA captions by matching generated captions to test-set answers.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/processed/sft_test.jsonl"))
    parser.add_argument("--model-name", default="/home/zhang.jianqiao/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/lora_sft_v1_5/final_lora"))
    parser.add_argument("--fail-output-dir", type=Path, default=Path("outputs/evaluations/lora_caption_match"))
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    evaluate_lora_caption_match(
        input_jsonl=args.input_jsonl,
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
        fail_output_dir=args.fail_output_dir,
        threshold=args.threshold,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
