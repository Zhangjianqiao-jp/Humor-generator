#!/usr/bin/env python
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_inference import load_qwen_vl_lora_for_inference
from src.training.sft_dataset import clean_generated_caption, extract_caption
from src.utils.io import read_jsonl, write_jsonl

DEFAULT_PROMPT = (
    "Generate one short, natural, image-specific humorous caption for this image. "
    "Do not explain."
)


def _decode_new_tokens(processor, input_ids, generated_ids) -> str:
    prompt_len = input_ids.shape[-1]
    new_tokens = generated_ids[:, prompt_len:]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def generate_lora_candidates(
    input_jsonl: Path,
    output_jsonl: Path,
    model_name: str,
    adapter_dir: Path,
    prompt: str,
    num_candidates: int,
    limit: int | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    model, processor, process_vision_info = load_qwen_vl_lora_for_inference(
        model_name=model_name,
        adapter_dir=adapter_dir,
    )
    rows = read_jsonl(input_jsonl)
    if limit is not None:
        rows = rows[:limit]

    outputs = []
    for row in rows:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": row["image"]},
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
        if hasattr(model, "device"):
            inputs = inputs.to(model.device)

        candidates = []
        for _ in range(num_candidates):
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            candidates.append(clean_generated_caption(_decode_new_tokens(processor, inputs["input_ids"], generated_ids), prompt=prompt))

        outputs.append(
            {
                "image": row["image"],
                "image_id": row.get("image_id"),
                "gold_caption": "" if extract_caption(row) is None else str(extract_caption(row)).strip(),
                "prompt": prompt,
                "candidates": candidates,
                "meta": {
                    "generator": "v1.5_lora_sft",
                    "model_name": model_name,
                    "adapter_dir": str(adapter_dir),
                    "num_candidates": num_candidates,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
        )
        if len(outputs) % 10 == 0:
            print(f"Generated candidates for {len(outputs)} images")

    write_jsonl(output_jsonl, outputs)
    print(f"Saved {len(outputs)} generation rows to {output_jsonl}")


def main() -> None:
    parser = ArgumentParser(description="Generate captions with a V1.5 LoRA-SFT adapter.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/processed/sft_test.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/generations/lora_candidates.jsonl"))
    parser.add_argument("--model-name", default="/home/zhang.jianqiao/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/lora_sft_v1_5/final_lora"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    generate_lora_candidates(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
        prompt=args.prompt,
        num_candidates=args.num_candidates,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
