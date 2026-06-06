#!/usr/bin/env python
from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_LITERAL_PROMPT = (
    "Describe this image in one short, literal, objective English sentence. "
    "Do not make a joke. Do not be humorous. Do not explain. Output only the sentence."
)
BAD_PREFIX_RE = re.compile(r"^(?:caption|description|literal caption|answer)\s*:\s*", re.IGNORECASE)
LEADING_LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def clean_literal(text: str) -> str:
    text = str(text).strip()
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    text = LEADING_LIST_RE.sub("", text)
    text = BAD_PREFIX_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    text = " ".join(text.split())
    return text


def normalize_for_dedupe(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


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
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def generate_once(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    num_return_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> list[str]:
    messages = build_messages(image_path, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(get_model_device(model))
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=num_return_sequences > 1,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
        )
    return [clean_literal(item) for item in decode_new_tokens(processor, inputs["input_ids"], generated_ids)]


def collect_images(positive_jsonl: Path, max_positives_per_image: int) -> list[dict[str, Any]]:
    by_image: dict[str, dict[str, Any]] = {}
    positives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(positive_jsonl):
        image_id = str(row.get("image_id") or Path(str(row.get("image", ""))).stem)
        image = str(row.get("image") or "")
        if not image:
            continue
        by_image.setdefault(image_id, {"image": image, "image_id": image_id})
        if len(positives[image_id]) < max_positives_per_image:
            positives[image_id].append(
                {
                    "caption": str(row.get("caption") or "").strip(),
                    "score": row.get("score"),
                    "rank_pct": row.get("rank_pct"),
                }
            )
    items = []
    for image_id, item in sorted(by_image.items()):
        item["positive_captions"] = positives.get(image_id, [])
        items.append(item)
    return items


def read_done_image_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = row.get("image_id")
            if image_id:
                done.add(str(image_id))
    return done


def main() -> None:
    parser = ArgumentParser(description="Generate non-humorous literal caption negatives with base Qwen2.5-VL.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sft.yaml"))
    parser.add_argument("--positive-jsonl", type=Path, default=Path("data/processed/reranker_score_pools_strict/strong_positive.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/processed/reranker_hard_negatives/literal_captions_qwen.jsonl"))
    parser.add_argument("--prompt", type=str, default=DEFAULT_LITERAL_PROMPT)
    parser.add_argument("--num-captions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.03)
    parser.add_argument("--max-positive-captions-per-image", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action=BooleanOptionalAction, default=True)
    parser.add_argument("--skip-missing-images", action=BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model"]["model_name"]
    print(f"[literal] loading base model={model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=config["model"].get("torch_dtype", "auto"),
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=config["model"].get("trust_remote_code", True))
    model.eval()

    items = collect_images(args.positive_jsonl, max_positives_per_image=args.max_positive_captions_per_image)
    items = items[args.offset :]
    if args.limit is not None:
        items = items[: args.limit]
    done = read_done_image_ids(args.output_jsonl) if args.resume else set()
    print(f"[literal] images_to_consider={len(items)} already_done={len(done)} output={args.output_jsonl}")

    failures = 0
    written = 0
    for index, item in enumerate(tqdm(items, desc="literal captions", dynamic_ncols=True), start=1):
        image_id = str(item["image_id"])
        image_path = str(item["image"])
        if image_id in done:
            continue
        if not Path(image_path).exists():
            if args.skip_missing_images:
                continue
            raise FileNotFoundError(image_path)
        try:
            raw_literals = generate_once(
                model=model,
                processor=processor,
                image_path=image_path,
                prompt=args.prompt,
                num_return_sequences=max(args.num_captions * 2, args.num_captions),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            literals: list[str] = []
            seen: set[str] = set()
            for literal in raw_literals:
                norm = normalize_for_dedupe(literal)
                if not literal or len(literal) < 5 or norm in seen:
                    continue
                seen.add(norm)
                literals.append(literal)
                if len(literals) >= args.num_captions:
                    break
            if len(literals) < args.num_captions:
                extra = generate_once(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    prompt=args.prompt,
                    num_return_sequences=args.num_captions - len(literals),
                    max_new_tokens=args.max_new_tokens,
                    temperature=max(args.temperature, 0.8),
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                for literal in extra:
                    norm = normalize_for_dedupe(literal)
                    if literal and norm not in seen:
                        seen.add(norm)
                        literals.append(literal)
                    if len(literals) >= args.num_captions:
                        break
            append_jsonl(
                args.output_jsonl,
                {
                    "image": image_path,
                    "image_id": image_id,
                    "literal_captions": literals,
                    "prompt": args.prompt,
                    "model": model_name,
                    "positive_captions": item.get("positive_captions", []),
                    "negative_type": "literal_caption",
                },
            )
            written += 1
        except Exception as exc:
            failures += 1
            print(f"[literal] failed image_id={image_id} image={image_path}: {exc}")
        if args.log_every > 0 and index % args.log_every == 0:
            print(f"[literal] processed={index}/{len(items)} written={written} failures={failures}")
    print(f"[literal] done written={written} failures={failures} output={args.output_jsonl}")


if __name__ == "__main__":
    main()
