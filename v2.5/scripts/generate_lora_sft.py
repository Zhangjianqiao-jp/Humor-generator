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
    extract_original_prompt,
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


def build_messages(
    image_path: str,
    prompt: str,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
) -> list[dict[str, Any]]:
    image_content: dict[str, Any] = {"type": "image", "image": image_path}
    if image_min_pixels is not None:
        image_content["min_pixels"] = int(image_min_pixels)
    if image_max_pixels is not None:
        image_content["max_pixels"] = int(image_max_pixels)
    return [
        {
            "role": "user",
            "content": [
                image_content,
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


def select_unique_image_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first row for each image while preserving input order."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        raw_image = extract_image_path(row)
        image_key = row.get("image_id") or raw_image
        if image_key is None:
            # Let run_generation produce the more useful missing-image error.
            selected.append(row)
            continue
        key = str(image_key)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def load_prompt_override(prompt: str | None, prompt_file: Path | None) -> str | None:
    """Resolve an explicit prompt while preserving multiline prompt files verbatim."""
    if prompt is not None and prompt_file is not None:
        raise ValueError("Use only one of --prompt and --prompt-file.")
    if prompt_file is None:
        return prompt
    value = prompt_file.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return value


def load_prompt_template(prompt_template_file: Path | None) -> str | None:
    if prompt_template_file is None:
        return None
    value = prompt_template_file.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Prompt template file is empty: {prompt_template_file}")
    if "{caption}" not in value:
        raise ValueError(f"Prompt template has no {{caption}} placeholder: {prompt_template_file}")
    return value


def generate_candidates(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    image_path: str,
    prompt: str,
    generation_config: dict[str, Any],
    num_candidates: int,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
) -> list[str]:
    messages = build_messages(
        image_path,
        prompt,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )
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
    do_sample = bool(generation_config.get("do_sample", True))
    generation_kwargs: dict[str, Any] = {
        "do_sample": do_sample,
        "max_new_tokens": int(generation_config.get("max_new_tokens", 48)),
        "repetition_penalty": float(generation_config.get("repetition_penalty", 1.05)),
    }
    if do_sample:
        generation_kwargs["temperature"] = float(generation_config.get("temperature", 0.8))
        generation_kwargs["top_p"] = float(generation_config.get("top_p", 0.9))
        top_k = generation_config.get("top_k")
        if top_k is not None and int(top_k) > 0:
            generation_kwargs["top_k"] = int(top_k)
    candidate_batch_size = int(generation_config.get("candidate_batch_size", num_candidates))
    if candidate_batch_size < 1:
        raise ValueError("generation.candidate_batch_size must be positive")
    decoded: list[str] = []
    with torch.no_grad():
        while len(decoded) < num_candidates:
            current = min(candidate_batch_size, num_candidates - len(decoded))
            generated_ids = model.generate(**inputs, **generation_kwargs, num_return_sequences=current)
            decoded.extend(decode_new_tokens(processor, inputs["input_ids"], generated_ids))
    preserve_newlines = bool(generation_config.get("preserve_newlines", False))
    return [
        clean_generated_caption(text, prompt=prompt, preserve_newlines=preserve_newlines)
        for text in decoded
    ]


def run_generation(
    config_path: Path,
    adapter_dir: Path | None,
    input_jsonl: Path,
    output_jsonl: Path,
    num_candidates: int | None,
    limit: int | None,
    prompt_override: str | None,
    unique_images: bool = False,
    seed: int | None = None,
    max_new_tokens: int | None = None,
    prompt_template: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> None:
    if prompt_override is not None and prompt_template is not None:
        raise ValueError("A fixed prompt and a per-row prompt template cannot both be used.")
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[generate] random seed: {seed}")
    config = load_config(config_path)
    model, processor, process_vision_info = load_qwen_vl_lora_for_inference(
        model_name=config["model"]["model_name"],
        adapter_dir=adapter_dir,
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=config["model"].get("torch_dtype", "auto"),
        trust_remote_code=config["model"].get("trust_remote_code", True),
        load_in_4bit=bool(config["model"].get("quantization", {}).get("load_in_4bit", False)),
        bnb_4bit_quant_type=str(
            config["model"].get("quantization", {}).get("bnb_4bit_quant_type", "nf4")
        ),
        bnb_4bit_use_double_quant=bool(
            config["model"].get("quantization", {}).get("bnb_4bit_use_double_quant", True)
        ),
    )
    rows = read_jsonl(input_jsonl)
    if unique_images:
        original_count = len(rows)
        rows = select_unique_image_rows(rows)
        print(f"[generate] unique-image selection: {len(rows)}/{original_count} rows")
    if limit is not None:
        rows = rows[:limit]

    image_root_value = config["data"].get("image_root")
    image_root = None if image_root_value in (None, "", "null") else Path(image_root_value)
    fallback_prompt = str(config["data"].get("sft_prompt", DEFAULT_SFT_PROMPT))
    generation_config = dict(config.get("generation", {}))
    if max_new_tokens is not None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        generation_config["max_new_tokens"] = max_new_tokens
    if temperature is not None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        generation_config["temperature"] = temperature
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        generation_config["top_p"] = top_p
    if top_k is not None:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        generation_config["top_k"] = top_k
    candidate_count = num_candidates or int(generation_config.get("num_candidates", 10))
    generation_config["num_candidates"] = candidate_count

    outputs = []
    for index, row in enumerate(rows):
        raw_image = extract_image_path(row)
        if raw_image is None:
            raise ValueError(f"Input row {index} has no image path.")
        image_path = str(resolve_image_path(str(raw_image), image_root))
        gold_caption = extract_caption(row)
        if prompt_template is not None:
            if gold_caption is None or not str(gold_caption).strip():
                raise ValueError(f"Input row {index} has no caption for the prompt template.")
            prompt = prompt_template.replace("{caption}", str(gold_caption).strip())
        else:
            prompt = prompt_override or extract_original_prompt(row) or fallback_prompt
        candidates = generate_candidates(
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            image_path=image_path,
            prompt=prompt,
            generation_config=generation_config,
            num_candidates=candidate_count,
            image_min_pixels=config["data"].get("image_min_pixels"),
            image_max_pixels=config["data"].get("image_max_pixels"),
        )
        outputs.append(
            {
                "image": image_path,
                "image_id": row.get("pair_id") or row.get("image_id") or Path(image_path).stem,
                "source_image_id": row.get("image_id") or Path(image_path).stem,
                "gold_caption": "" if gold_caption is None else str(gold_caption).strip(),
                "gold_captions": row.get("gold_captions"),
                "caption_count": row.get("caption_count"),
                "caption_set_sha256": row.get("caption_set_sha256"),
                "prompt": prompt,
                "candidates": candidates,
                "generation_config": generation_config,
                "seed": seed,
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
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Load the configured base model without any LoRA adapter.",
    )
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/processed/sft_val.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/generations/v1_5_candidates.jsonl"))
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--unique-images",
        action="store_true",
        help="Keep the first row per image_id before applying --limit.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Explicit prompt override. Defaults to the row prompt or data.sft_prompt.",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="UTF-8 file containing the exact multiline prompt override.",
    )
    prompt_group.add_argument(
        "--prompt-template-file",
        type=Path,
        default=None,
        help="UTF-8 prompt containing {caption}, replaced from each row's target caption.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional generation RNG seed.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional generation-length override from the config.",
    )
    args = parser.parse_args()
    prompt_override = load_prompt_override(args.prompt, args.prompt_file)
    prompt_template = load_prompt_template(args.prompt_template_file)
    run_generation(
        config_path=args.config,
        adapter_dir=None if args.base_only else args.adapter,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        num_candidates=args.num_candidates,
        limit=args.limit,
        prompt_override=prompt_override,
        unique_images=args.unique_images,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        prompt_template=prompt_template,
        temperature=None,
        top_p=None,
        top_k=None,
    )


if __name__ == "__main__":
    main()
