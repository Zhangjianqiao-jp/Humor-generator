#!/usr/bin/env python
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_inference import load_qwen_vl_lora_for_inference
from src.training.sft_dataset import (
    DEFAULT_SFT_PROMPT,
    clean_generated_caption,
    extract_caption,
    extract_image_path,
    resolve_image_path,
)
from src.utils.io import read_jsonl, write_jsonl


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("data", {})
    config["data"].setdefault("sft_prompt", DEFAULT_SFT_PROMPT)
    config["data"].setdefault("image_root", None)
    config.setdefault("generation", {})
    config["generation"].setdefault("max_new_tokens", 48)
    config["generation"].setdefault("temperature", 0.8)
    config["generation"].setdefault("top_p", 0.9)
    config["generation"].setdefault("do_sample", True)
    config["generation"].setdefault("num_candidates", 10)
    config["generation"].setdefault("repetition_penalty", 1.05)
    return config


def get_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def build_messages(image_path: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def decode_new_tokens(processor: Any, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> list[str]:
    prompt_len = input_ids.shape[-1]
    new_tokens = generated_ids[:, prompt_len:]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def generate_candidates(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    image_path: str,
    prompt: str,
    generation_config: dict[str, Any],
    num_candidates: int,
) -> list[str]:
    messages = build_messages(image_path, prompt)
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
            do_sample=bool(generation_config.get("do_sample", True)),
            temperature=float(generation_config.get("temperature", 0.8)),
            top_p=float(generation_config.get("top_p", 0.9)),
            max_new_tokens=int(generation_config.get("max_new_tokens", 48)),
            repetition_penalty=float(generation_config.get("repetition_penalty", 1.05)),
            num_return_sequences=num_candidates,
        )
    decoded = decode_new_tokens(processor, inputs["input_ids"], generated_ids)
    return [clean_generated_caption(text, prompt=prompt) for text in decoded]


def run_generation(
    config_path: Path,
    adapter_dir: Path,
    input_jsonl: Path,
    output_jsonl: Path,
    num_candidates: int | None,
    limit: int | None,
    prompt_override: str | None,
) -> None:
    config = load_config(config_path)
    model, processor, process_vision_info = load_qwen_vl_lora_for_inference(
        model_name=config["model"]["model_name"],
        adapter_dir=adapter_dir,
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=config["model"].get("torch_dtype", "auto"),
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    rows = read_jsonl(input_jsonl)
    if limit is not None:
        rows = rows[:limit]

    image_root_value = config["data"].get("image_root")
    image_root = None if image_root_value in (None, "", "null") else Path(image_root_value)
    prompt = prompt_override or str(config["data"].get("sft_prompt", DEFAULT_SFT_PROMPT))
    generation_config = dict(config.get("generation", {}))
    candidate_count = num_candidates or int(generation_config.get("num_candidates", 10))
    generation_config["num_candidates"] = candidate_count

    outputs = []
    for index, row in enumerate(rows):
        raw_image = extract_image_path(row)
        if raw_image is None:
            raise ValueError(f"Input row {index} has no image path.")
        image_path = str(resolve_image_path(str(raw_image), image_root))
        gold_caption = extract_caption(row)
        candidates = generate_candidates(
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            image_path=image_path,
            prompt=prompt,
            generation_config=generation_config,
            num_candidates=candidate_count,
        )
        outputs.append(
            {
                "image": image_path,
                "image_id": row.get("image_id") or Path(image_path).stem,
                "gold_caption": "" if gold_caption is None else str(gold_caption).strip(),
                "prompt": prompt,
                "candidates": candidates,
            }
        )
        if (index + 1) % 10 == 0:
            print(f"[generate] processed {index + 1}/{len(rows)} images")

    write_jsonl(output_jsonl, outputs)
    print(f"[generate] saved {len(outputs)} rows to {output_jsonl}")


def main() -> None:
    parser = ArgumentParser(description="Generate captions with a V1.5 clean-prompt LoRA-SFT adapter.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sft.yaml"))
    parser.add_argument("--adapter", "--adapter-dir", dest="adapter", type=Path, default=Path("outputs/lora_sft_v1_5/best_val_loss"))
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/processed/sft_val.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/generations/v1_5_candidates.jsonl"))
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Explicit prompt override. Defaults to data.sft_prompt.")
    args = parser.parse_args()
    run_generation(
        config_path=args.config,
        adapter_dir=args.adapter,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        num_candidates=args.num_candidates,
        limit=args.limit,
        prompt_override=args.prompt,
    )


if __name__ == "__main__":
    main()
