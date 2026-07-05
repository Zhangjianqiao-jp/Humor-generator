#!/usr/bin/env python
from __future__ import annotations

import json
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.guided_prompting import DEFAULT_BASE_PROMPT, GUIDED_PROMPT_METHODS, build_guided_prompt
from src.analysis.review_html import write_guided_review_html
from src.models.qwen_vl_lora_inference import load_qwen_vl_lora_for_inference
from src.training.sft_dataset import clean_generated_caption, extract_caption, extract_image_path, resolve_image_path
from src.utils.io import read_jsonl, write_jsonl


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def keys_for_row(row: dict[str, Any]) -> list[str]:
    keys = []
    row_key = row.get("row_key")
    if row_key:
        keys.append(f"row_key:{row_key}")
    image_id = row.get("image_id")
    if image_id:
        keys.append(f"id:{image_id}")
    image = row.get("image") or extract_image_path(row)
    if image:
        keys.append(f"image:{image}")
        keys.append(f"stem:{Path(str(image)).stem}")
    return keys


def load_context_map(context_jsonl: Path) -> dict[str, dict[str, Any]]:
    context_map: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(context_jsonl):
        if row.get("failed"):
            continue
        for key in keys_for_row(row):
            context_map[key] = row
    return context_map


def find_context(row: dict[str, Any], context_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for key in keys_for_row(row):
        if key in context_map:
            return context_map[key]
    return None


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


def wait_for_gpu_memory(min_free_mb: int, gpu_index: int, stable_checks: int, check_seconds: int) -> None:
    if min_free_mb <= 0:
        return
    stable = 0
    print(
        "[guided-generate] "
        f"waiting for gpu={gpu_index} free memory >= {min_free_mb} MiB "
        f"for {stable_checks} consecutive checks"
    )
    while stable < stable_checks:
        free_mb = query_gpu_free_mb(gpu_index)
        stable = stable + 1 if free_mb >= min_free_mb else 0
        print(
            "[guided-generate] "
            f"gpu={gpu_index} free_memory={free_mb} MiB "
            f"stable={stable}/{stable_checks}"
        )
        if stable < stable_checks:
            time.sleep(check_seconds)


def decode_new_tokens(processor: Any, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
    prompt_len = input_ids.shape[-1]
    new_tokens = generated_ids[:, prompt_len:]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def get_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_one_image(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    image_path: Path,
    prompt: str,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int | None = None,
) -> list[str]:
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

    set_seed(seed)
    candidates = []
    for _ in range(num_candidates):
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
        if temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            generation_kwargs["do_sample"] = False
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_kwargs)
        candidates.append(clean_generated_caption(decode_new_tokens(processor, inputs["input_ids"], generated_ids), prompt=prompt))
    return candidates


def method_key(method: str) -> str:
    return method.replace("-", "_")


def default_output_for_method(config: dict[str, Any], method: str) -> Path:
    output = config.get("output", {})
    if method == "prompt-method":
        return Path(output.get("prompt_method_jsonl", "outputs/generations/vlm_guided_prompt_method.jsonl"))
    if method == "feature-method":
        return Path(output.get("feature_method_jsonl", "outputs/generations/vlm_guided_feature_method.jsonl"))
    key = method_key(method)
    return Path(output.get(f"{key}_jsonl", f"outputs/generations/vlm_guided_{key}.jsonl"))


def default_html_for_method(config: dict[str, Any], method: str) -> Path:
    output = config.get("output", {})
    if method == "prompt-method":
        return Path(output.get("prompt_method_html", "outputs/reviews/vlm_guided_prompt_method.html"))
    if method == "feature-method":
        return Path(output.get("feature_method_html", "outputs/reviews/vlm_guided_feature_method.html"))
    key = method_key(method)
    return Path(output.get(f"{key}_html", f"outputs/reviews/vlm_guided_{key}.html"))


def optional_path(value: Any) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    return Path(str(value))


def context_has_required_fields(context: dict[str, Any], method: str) -> bool:
    if method.startswith("hic-"):
        return isinstance(context.get("analysis") or context.get("humor_viewpoint"), dict)
    if method in ("structured-nl", "structured-json"):
        return "structured_humor" in context
    if method in ("prompt-method", "feature-method"):
        return "visual_facts" in context or "humor_points" in context
    if method == "description-only":
        return bool(context.get("image_description")) or bool(context.get("visual_facts"))
    return True


def run_generate(
    config_path: Path,
    method: str,
    input_jsonl: Path | None,
    context_jsonl: Path | None,
    output_jsonl: Path | None,
    review_html: Path | None,
    limit: int | None,
    num_candidates: int | None,
    overwrite: bool,
    seed: int | None,
    wait_gpu_free_mb: int,
    wait_gpu_index: int,
    wait_gpu_stable_checks: int,
    wait_gpu_check_seconds: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> None:
    config = load_config(config_path)
    data_config = config.get("data", {})
    generator_config = config["generator"]
    output_config = config.get("output", {})

    input_path = input_jsonl or Path(data_config.get("input_jsonl", "data/processed/sft_test.jsonl"))
    context_path = context_jsonl or Path(output_config.get("context_jsonl", "outputs/analysis/vlm_humor_context.jsonl"))
    output_path = output_jsonl or default_output_for_method(config, method)
    review_path = review_html or default_html_for_method(config, method)
    if limit is None:
        limit = data_config.get("limit")
    image_root = data_config.get("image_root")
    image_root_path = None if image_root in (None, "", "null") else Path(image_root)
    candidate_count = int(num_candidates or generator_config.get("num_candidates", 8))
    base_prompt = str(generator_config.get("base_prompt", DEFAULT_BASE_PROMPT))
    base_seed = seed if seed is not None else generator_config.get("seed")
    base_seed = None if base_seed in (None, "", "null") else int(base_seed)

    rows = read_jsonl(input_path)
    if limit is not None:
        rows = rows[: int(limit)]
    context_map = {} if method == "plain" else load_context_map(context_path)

    if overwrite and output_path.exists():
        output_path.unlink()

    wait_for_gpu_memory(
        min_free_mb=wait_gpu_free_mb,
        gpu_index=wait_gpu_index,
        stable_checks=wait_gpu_stable_checks,
        check_seconds=wait_gpu_check_seconds,
    )

    adapter_dir = optional_path(generator_config.get("adapter_dir"))
    model, processor, process_vision_info = load_qwen_vl_lora_for_inference(
        model_name=str(generator_config["model_name"]),
        adapter_dir=adapter_dir,
        device_map=str(generator_config.get("device_map", "auto")),
        torch_dtype=str(generator_config.get("torch_dtype", "auto")),
        trust_remote_code=bool(generator_config.get("trust_remote_code", True)),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    outputs: list[dict[str, Any]] = []
    skipped_missing_context = 0
    skipped_incomplete_context = 0
    skipped_missing_image = 0
    for index, row in enumerate(tqdm(rows, desc=f"generating {method}", dynamic_ncols=True)):
        context = {} if method == "plain" else find_context(row, context_map)
        if context is None:
            skipped_missing_context += 1
            print(f"[guided-generate] missing context row={index} image_id={row.get('image_id')}")
            continue
        if not context_has_required_fields(context, method):
            skipped_incomplete_context += 1
            print(f"[guided-generate] incomplete context for {method} row={index} image_id={row.get('image_id')}")
            continue
        raw_image = extract_image_path(row)
        if raw_image is None:
            skipped_missing_image += 1
            continue
        image_path = resolve_image_path(str(raw_image), image_root_path)
        if not image_path.exists():
            skipped_missing_image += 1
            print(f"[guided-generate] missing image row={index}: {image_path}")
            continue

        visual_facts = context.get("visual_facts") or context.get("humor_points") or {}
        humor_viewpoint = context.get("analysis") or context.get("humor_viewpoint") or {}
        prompt = build_guided_prompt(
            method=method,
            image_description=str(context.get("image_description") or ""),
            visual_facts=visual_facts,
            structured_humor=context.get("structured_humor") or {},
            humor_viewpoint=humor_viewpoint,
            gold_caption=str(context.get("gold_caption") or extract_caption(row) or ""),
            base_prompt=base_prompt,
        )
        if not prompt.endswith(base_prompt) or prompt.count(base_prompt) != 1:
            raise RuntimeError("Guided prompt must preserve the base prompt exactly once at the end")
        row_seed = None if base_seed is None else base_seed + index
        candidates = generate_one_image(
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            image_path=image_path,
            prompt=prompt,
            num_candidates=candidate_count,
            max_new_tokens=int(generator_config.get("max_new_tokens", 48)),
            temperature=float(generator_config.get("temperature", 0.8)),
            top_p=float(generator_config.get("top_p", 0.9)),
            repetition_penalty=float(generator_config.get("repetition_penalty", 1.05)),
            seed=row_seed,
        )
        outputs.append(
            {
                "image": str(image_path),
                "image_id": row.get("image_id") or image_path.stem,
                "row_key": row.get("row_key") or context.get("row_key"),
                "gold_caption": "" if extract_caption(row) is None else str(extract_caption(row)).strip(),
                "method": method,
                "prompt": prompt,
                "image_description": context.get("image_description"),
                "visual_facts": visual_facts,
                "structured_humor": context.get("structured_humor"),
                "structured_humor_parse_error": context.get("structured_humor_parse_error"),
                "humor_viewpoint": humor_viewpoint,
                "humor_viewpoint_prompt_version": context.get("prompt_version"),
                "candidates": candidates,
                "meta": {
                    "generator_model_name": generator_config["model_name"],
                    "adapter_dir": None if adapter_dir is None else str(adapter_dir),
                    "generator_mode": "base" if adapter_dir is None else "lora_adapter",
                    "context_jsonl": str(context_path),
                    "num_candidates": candidate_count,
                    "temperature": float(generator_config.get("temperature", 0.8)),
                    "top_p": float(generator_config.get("top_p", 0.9)),
                    "repetition_penalty": float(generator_config.get("repetition_penalty", 1.05)),
                    "seed": row_seed,
                },
            }
        )

    write_jsonl(output_path, outputs)
    write_guided_review_html(outputs, review_path)
    print(f"[guided-generate] saved {len(outputs)} rows to {output_path}")
    print(f"[guided-generate] saved review HTML to {review_path}")
    print(
        "[guided-generate] "
        f"skipped_missing_context={skipped_missing_context} "
        f"skipped_incomplete_context={skipped_incomplete_context} "
        f"skipped_missing_image={skipped_missing_image}"
    )


def main() -> None:
    parser = ArgumentParser(description="Generate base or LoRA candidates with VLM visual facts and structured humor guidance.")
    parser.add_argument("--config", type=Path, default=Path("configs/vlm_guided_generation.yaml"))
    parser.add_argument("--method", choices=GUIDED_PROMPT_METHODS, default="structured-brief")
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--context-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--review-html", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wait-gpu-free-mb", type=int, default=0)
    parser.add_argument("--wait-gpu-index", type=int, default=0)
    parser.add_argument("--wait-gpu-stable-checks", type=int, default=3)
    parser.add_argument("--wait-gpu-check-seconds", type=int, default=60)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_generate(
        config_path=args.config,
        method=args.method,
        input_jsonl=args.input_jsonl,
        context_jsonl=args.context_jsonl,
        output_jsonl=args.output_jsonl,
        review_html=args.review_html,
        limit=args.limit,
        num_candidates=args.num_candidates,
        overwrite=args.overwrite,
        seed=args.seed,
        wait_gpu_free_mb=args.wait_gpu_free_mb,
        wait_gpu_index=args.wait_gpu_index,
        wait_gpu_stable_checks=args.wait_gpu_stable_checks,
        wait_gpu_check_seconds=args.wait_gpu_check_seconds,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )


if __name__ == "__main__":
    main()
