#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.sft_dataset import extract_caption


BASE_MODEL = Path("/home/zhang.jianqiao/models/Qwen2.5-VL-3B-Instruct")
GUIDANCE_MODEL = Path("/home/zhang.jianqiao/models/Qwen2.5-VL-7B-Instruct")
PYTHON_REQUIRED = Path("/home/zhang.jianqiao/miniconda3/envs/humor/bin/python")
BASE_PROMPT = (
    "Generate one short, natural, image-specific humorous caption for this image. "
    "Do not explain."
)
NUM_IMAGES = 1000
NUM_CANDIDATES = 8
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_NEW_TOKENS = 48
REPETITION_PENALTY = 1.05
DEFAULT_SEED = 250618
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


GUIDANCE_PROMPT = """You are a conservative visual analyst supporting a controlled image-caption experiment.

Return only valid JSON with exactly this schema:
{
  "description": "one conservative sentence describing only clearly visible content",
  "humor_cue": "one high-confidence visible incongruity, or an empty string"
}

Rules:
- The description must be exactly one sentence and must be literal, concise, and accurate.
- In the description, use generic labels such as person, animal, clothing, and object; never infer a profession or relationship from attire or context.
- The humor_cue may contain at most one observation.
- A humor cue is allowed only when it is directly visible as an unusual physical detail, clear size difference, action contrast, composition relationship, or visible role/object mismatch.
- Do not guess identity, age, profession, social role, relationship, emotion, intention, motivation, thought, or hidden story.
- Do not infer that anyone is stealing, guarding, escaping, pretending, waiting, planning, helping, competing, or trying to do something.
- Do not introduce any object or detail that is not clearly visible.
- Do not use metaphor, analogy, fictional characters, pop-culture references, or "looks like/as if" comparisons.
- When one allowed pattern is unmistakably visible, return that plain visual observation instead of leaving humor_cue empty.
- If the description already contains an unmistakable size, action, composition, or object-role contrast, state that single contrast in humor_cue; do not leave it empty merely because it also appears in the description.
- Valid cue example: "The miniature meal is much smaller than the hand holding it."
- Valid cue example: "One person is much smaller than the oversized chair beside them."
- Valid cue example: "A dog is positioned behind a car steering wheel."
- Invalid cue example: "The dog is driving to work." This invents an action and story.
- If there is no reliable visible incongruity, set humor_cue to an empty string.
- Do not write or suggest a final caption. The humor_cue must be a plain visual observation, not a punchline.
- Use English.

Return JSON only. No markdown and no explanation."""


JUDGE_PROMPT_TEMPLATE = """You are judging two anonymous sets of humorous captions for the attached image.

Choose captions using only what is visibly supported by the image. The sets come from two unknown systems.

Evaluation order:
1. Reject captions that hallucinate important visual facts or depend on an unsupported identity, profession, relationship, emotion, intention, or hidden story.
2. Prefer genuinely humorous captions over literal descriptions.
3. Prefer image-specific, natural, concise captions.
4. Minor wording issues matter less than visual grounding and humor.

For each set, count how many of its 8 captions are usable: visually grounded, understandable, and at least mildly humorous.
Select the best caption in each set, then decide which set contains the stronger best caption.
Use "tie" only when the best captions are genuinely comparable.
Do not rewrite or create captions.

Group A:
{group_a}

Group B:
{group_b}

Return only valid JSON:
{{
  "best_a_index": 1,
  "best_b_index": 1,
  "usable_a_count": 0,
  "usable_b_count": 0,
  "winner_group": "A",
  "confidence": 1,
  "reason": "brief comparison"
}}

Indices are 1-8. Counts are 0-8. winner_group is A, B, or tie. confidence is 1-5."""


FORBIDDEN_CUE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:maybe|perhaps|probably|likely|possibly|seems?|appears?)\b", "speculation"),
    (r"\b(?:looks? like|as if|as though|resembl(?:e|es|ing))\b", "analogy"),
    (
        r"\b(?:trying|wants?|intends?|plans?|thinks?|feels?|hopes?|decides?|attempts?)\b",
        "mental state or intention",
    ),
    (
        r"\b(?:steal(?:s|ing)?|guard(?:s|ing)?|escap(?:e|es|ing)|pretend(?:s|ing)?|"
        r"wait(?:s|ing)?|help(?:s|ing)?|compet(?:e|es|ing)|chase(?:s|d|ing)?|"
        r"protect(?:s|ing)?|threaten(?:s|ing)?)\b",
        "hidden story or inferred action",
    ),
    (
        r"\b(?:happy|sad|angry|afraid|scared|worried|excited|confused|surprised|"
        r"embarrassed|bored|relaxed|serious|proud|jealous)\b",
        "emotion",
    ),
    (
        r"\b(?:doctor|nurse|police|officer|soldier|teacher|student|boss|employee|"
        r"worker|chef|waiter|driver|guard|thief|criminal|husband|wife|parent|"
        r"mother|father|boyfriend|girlfriend|friend|owner)\b",
        "identity, profession, or relationship",
    ),
    (r"[!?]", "caption-like punctuation"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def image_id_for(row: dict[str, Any]) -> str:
    image = Path(str(row.get("image") or ""))
    return str(row.get("image_id") or image.stem)


def existing_successes(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if not row.get("failed"):
            result[image_id_for(row)] = row
    return result


def validate_runtime() -> None:
    actual = Path(sys.executable).resolve()
    required = PYTHON_REQUIRED.resolve()
    if actual != required:
        raise RuntimeError(f"Wrong Python: {actual}; required: {required}")
    for model_path in (BASE_MODEL, GUIDANCE_MODEL):
        if not model_path.is_dir():
            raise FileNotFoundError(f"Missing model directory: {model_path}")
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"Missing model config: {model_path / 'config.json'}")


def validate_subset(rows: list[dict[str, Any]], source_test: Path) -> dict[str, Any]:
    if len(rows) != NUM_IMAGES:
        raise ValueError(f"Expected exactly {NUM_IMAGES} rows, found {len(rows)}")
    ids = [image_id_for(row) for row in rows]
    images = [str(Path(str(row.get("image") or "")).resolve()) for row in rows]
    if len(set(ids)) != NUM_IMAGES:
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate image_id values: {duplicates[:10]}")
    if len(set(images)) != NUM_IMAGES:
        duplicates = [key for key, count in Counter(images).items() if count > 1]
        raise ValueError(f"Duplicate image paths: {duplicates[:10]}")
    missing = [image for image in images if not Path(image).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")

    source_rows = read_jsonl(source_test)
    source_pairs = {
        (image_id_for(row), str(Path(str(row.get("image") or "")).resolve()))
        for row in source_rows
    }
    absent = [
        (image_id_for(row), str(Path(str(row.get("image") or "")).resolve()))
        for row in rows
        if (image_id_for(row), str(Path(str(row.get("image") or "")).resolve())) not in source_pairs
    ]
    if absent:
        raise ValueError(f"{len(absent)} subset rows are absent from {source_test}; first: {absent[0]}")
    return {
        "rows": len(rows),
        "unique_image_ids": len(set(ids)),
        "unique_image_paths": len(set(images)),
        "all_images_exist": True,
        "all_rows_in_source_test": True,
    }


def load_qwen(model_path: Path) -> tuple[Any, Any, Any]:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model.eval()
    lora_names = [name for name, _ in model.named_parameters() if "lora" in name.lower()]
    if lora_names:
        raise RuntimeError(f"LoRA parameters unexpectedly present: {lora_names[:5]}")
    return model, processor, process_vision_info


def model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and torch.device(device).type != "meta":
        return torch.device(device)
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine model device")


def prepare_inputs(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    image: str,
    prompt: str,
) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(model_device(model))


def generate_text(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    image: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    inputs = prepare_inputs(model, processor, process_vision_info, image, prompt)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Model output is not a JSON object")
    return value


def one_sentence(text: str) -> str:
    text = normalize_space(text).strip(" \"'")
    if not text:
        return ""
    matches = list(re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", text))
    if matches:
        text = text[: matches[0].start()].strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text



def conservative_description(value: Any) -> tuple[str, bool]:
    original = one_sentence(value)
    text = original
    plural_roles = (
        r"\b(?:men|women|boys|girls|babies|children|soldiers|officers|police|"
        r"athletes|players|wrestlers|ballerinas|dancers|artists|models|workers|"
        r"doctors|nurses|teachers|students|chefs|waiters|drivers|guards|parents)\b"
    )
    singular_roles = (
        r"\b(?:man|woman|boy|girl|baby|child|soldier|officer|policeman|policewoman|"
        r"athlete|tennis player|player|sumo wrestler|wrestler|ballerina|dancer|"
        r"makeup artist|artist|model|worker|doctor|nurse|teacher|student|chef|"
        r"waiter|driver|guard|mother|father|husband|wife|owner)\b"
    )
    text = re.sub(plural_roles, "people", text, flags=re.IGNORECASE)
    text = re.sub(singular_roles, "person", text, flags=re.IGNORECASE)
    text = re.sub(r"\ban person\b", "a person", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:he|she)\b", "they", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:his|her)\b", "their", text, flags=re.IGNORECASE)
    text = normalize_space(text)
    return text, text != original

def validate_humor_cue(value: Any) -> tuple[str, str | None]:
    if value is None:
        return "", None
    if isinstance(value, list):
        if not value:
            return "", None
        if len(value) > 1:
            return "", "more than one cue"
        value = value[0]
    cue = normalize_space(value).strip(" \"'")
    if not cue:
        return "", None
    if len(cue) > 240:
        return "", "cue longer than 240 characters"
    if cue.count(";") > 0 or cue.count("\n") > 0:
        return "", "possibly multiple cues"
    for pattern, reason in FORBIDDEN_CUE_PATTERNS:
        if re.search(pattern, cue, flags=re.IGNORECASE):
            return "", reason
    return cue.rstrip("."), None


def guided_prompt(description: str, humor_cue: str) -> str:
    return (
        f"Image description: {description}\n"
        f"Humor cue: {humor_cue}\n\n"
        f"{BASE_PROMPT}"
    )


def extract_guidance(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    existing = existing_successes(output_path)
    if len(existing) == len(rows):
        print(f"[extract] already complete: {output_path}")
        return
    model, processor, process_vision_info = load_qwen(GUIDANCE_MODEL)
    try:
        for index, row in enumerate(tqdm(rows, desc="7B guidance", dynamic_ncols=True)):
            image_id = image_id_for(row)
            if image_id in existing:
                continue
            image = str(Path(str(row["image"])).resolve())
            try:
                raw = generate_text(
                    model,
                    processor,
                    process_vision_info,
                    image,
                    GUIDANCE_PROMPT,
                    max_new_tokens=192,
                )
                parsed = extract_json_object(raw)
                description, description_genericized = conservative_description(parsed.get("description"))
                if not description:
                    raise ValueError("Empty description")
                cue, rejected_reason = validate_humor_cue(parsed.get("humor_cue"))
                output_row = {
                    "image": image,
                    "image_id": image_id,
                    "source_index": index,
                    "description": description,
                    "description_genericized": description_genericized,
                    "humor_cue": cue,
                    "cue_rejected_reason": rejected_reason,
                    "raw_response": raw,
                    "extractor_model": str(GUIDANCE_MODEL),
                    "prompt_sha256": hashlib.sha256(GUIDANCE_PROMPT.encode()).hexdigest(),
                }
            except Exception as exc:
                output_row = {
                    "image": image,
                    "image_id": image_id,
                    "source_index": index,
                    "failed": True,
                    "error": repr(exc),
                }
                print(f"[extract] failed {image_id}: {exc}")
            append_jsonl(output_path, output_row)
            if not output_row.get("failed"):
                existing[image_id] = output_row
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def seed_for_image(index: int, base_seed: int) -> int:
    return base_seed + index


def sample_candidates(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    image: str,
    prompt: str,
    seed: int,
) -> list[str]:
    inputs = prepare_inputs(model, processor, process_vision_info, image, prompt)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=NUM_CANDIDATES,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            use_cache=True,
        )
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if len(decoded) != NUM_CANDIDATES:
        raise RuntimeError(f"Expected {NUM_CANDIDATES} candidates, got {len(decoded)}")
    return [candidate.strip() for candidate in decoded]


def generate_pairs(
    rows: list[dict[str, Any]],
    guidance_path: Path,
    output_path: Path,
    base_seed: int,
) -> None:
    contexts = existing_successes(guidance_path)
    if len(contexts) != len(rows):
        raise RuntimeError(
            f"Guidance incomplete: expected {len(rows)} successful rows, got {len(contexts)}"
        )
    existing = existing_successes(output_path)
    if len(existing) == len(rows):
        print(f"[generate] already complete: {output_path}")
        return
    model, processor, process_vision_info = load_qwen(BASE_MODEL)
    try:
        for index, row in enumerate(tqdm(rows, desc="3B paired generation", dynamic_ncols=True)):
            image_id = image_id_for(row)
            if image_id in existing:
                continue
            image = str(Path(str(row["image"])).resolve())
            context = contexts[image_id]
            seed = seed_for_image(index, base_seed)
            prompt_b = guided_prompt(context["description"], context["humor_cue"])
            if not prompt_b.endswith(BASE_PROMPT) or prompt_b.count(BASE_PROMPT) != 1:
                raise RuntimeError("Guided prompt does not preserve the base prompt exactly once")
            try:
                plain = sample_candidates(
                    model,
                    processor,
                    process_vision_info,
                    image,
                    BASE_PROMPT,
                    seed,
                )
                guided = sample_candidates(
                    model,
                    processor,
                    process_vision_info,
                    image,
                    prompt_b,
                    seed,
                )
                output_row = {
                    "image": image,
                    "image_id": image_id,
                    "source_index": index,
                    "gold_caption": normalize_space(extract_caption(row)),
                    "generator_model": str(BASE_MODEL),
                    "adapter": None,
                    "plain_prompt": BASE_PROMPT,
                    "guided_prompt": prompt_b,
                    "description": context["description"],
                    "humor_cue": context["humor_cue"],
                    "seed": seed,
                    "sampling": {
                        "num_candidates": NUM_CANDIDATES,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "repetition_penalty": REPETITION_PENALTY,
                    },
                    "plain_candidates": plain,
                    "guided_candidates": guided,
                }
            except Exception as exc:
                output_row = {
                    "image": image,
                    "image_id": image_id,
                    "source_index": index,
                    "failed": True,
                    "error": repr(exc),
                }
                print(f"[generate] failed {image_id}: {exc}")
            append_jsonl(output_path, output_row)
            if not output_row.get("failed"):
                existing[image_id] = output_row
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def balanced_plain_first_ids(rows: list[dict[str, Any]], base_seed: int) -> set[str]:
    if len(rows) % 2:
        raise ValueError("Balanced group assignment requires an even number of rows")
    ids = [image_id_for(row) for row in rows]
    rng = random.Random(base_seed + 700_001)
    rng.shuffle(ids)
    return set(ids[: len(ids) // 2])


def shuffled_candidates(
    candidates: list[str],
    seed: int,
) -> tuple[list[str], list[int]]:
    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)
    return [candidates[index] for index in order], order


def numbered(candidates: Iterable[str]) -> str:
    return "\n".join(f"{index}. {caption}" for index, caption in enumerate(candidates, 1))


def parse_judgment(raw: str) -> dict[str, Any]:
    parsed = extract_json_object(raw)
    winner = normalize_space(parsed.get("winner_group")).upper()
    if winner not in {"A", "B", "TIE"}:
        raise ValueError(f"Invalid winner_group: {winner}")
    result = {
        "best_a_index": int(parsed["best_a_index"]),
        "best_b_index": int(parsed["best_b_index"]),
        "usable_a_count": int(parsed["usable_a_count"]),
        "usable_b_count": int(parsed["usable_b_count"]),
        "winner_group": "tie" if winner == "TIE" else winner,
        "confidence": int(parsed.get("confidence", 0)),
        "reason": normalize_space(parsed.get("reason")),
    }
    if not 1 <= result["best_a_index"] <= NUM_CANDIDATES:
        raise ValueError("best_a_index outside 1-8")
    if not 1 <= result["best_b_index"] <= NUM_CANDIDATES:
        raise ValueError("best_b_index outside 1-8")
    if not 0 <= result["usable_a_count"] <= NUM_CANDIDATES:
        raise ValueError("usable_a_count outside 0-8")
    if not 0 <= result["usable_b_count"] <= NUM_CANDIDATES:
        raise ValueError("usable_b_count outside 0-8")
    if not 0 <= result["confidence"] <= 5:
        raise ValueError("confidence outside 0-5")
    return result


def judge_pairs(
    rows: list[dict[str, Any]],
    generations_path: Path,
    output_path: Path,
    base_seed: int,
) -> None:
    generations = existing_successes(generations_path)
    if len(generations) != len(rows):
        raise RuntimeError(
            f"Generation incomplete: expected {len(rows)} successful rows, got {len(generations)}"
        )
    existing = existing_successes(output_path)
    if len(existing) == len(rows):
        print(f"[judge] already complete: {output_path}")
        return
    plain_first_ids = balanced_plain_first_ids(rows, base_seed)
    model, processor, process_vision_info = load_qwen(GUIDANCE_MODEL)
    try:
        for index, row in enumerate(tqdm(rows, desc="7B blind judging", dynamic_ncols=True)):
            image_id = image_id_for(row)
            if image_id in existing:
                continue
            generation = generations[image_id]
            plain, plain_order = shuffled_candidates(
                generation["plain_candidates"],
                base_seed + 1_000_003 + index * 2,
            )
            guided, guided_order = shuffled_candidates(
                generation["guided_candidates"],
                base_seed + 1_000_004 + index * 2,
            )
            plain_first = image_id in plain_first_ids
            if plain_first:
                group_a, group_b = plain, guided
                mapping = {"A": "plain", "B": "guided"}
                orders = {"A": plain_order, "B": guided_order}
            else:
                group_a, group_b = guided, plain
                mapping = {"A": "guided", "B": "plain"}
                orders = {"A": guided_order, "B": plain_order}
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                group_a=numbered(group_a),
                group_b=numbered(group_b),
            )
            try:
                raw = generate_text(
                    model,
                    processor,
                    process_vision_info,
                    generation["image"],
                    prompt,
                    max_new_tokens=256,
                )
                judgment = parse_judgment(raw)
                winner_group = judgment["winner_group"]
                winner_method = "tie" if winner_group == "tie" else mapping[winner_group]
                best_a_caption = group_a[judgment["best_a_index"] - 1]
                best_b_caption = group_b[judgment["best_b_index"] - 1]
                usable_by_method = {
                    mapping["A"]: judgment["usable_a_count"],
                    mapping["B"]: judgment["usable_b_count"],
                }
                output_row = {
                    "image": generation["image"],
                    "image_id": image_id,
                    "source_index": index,
                    "judge_model": str(GUIDANCE_MODEL),
                    "mapping": mapping,
                    "candidate_orders": orders,
                    "winner_group": winner_group,
                    "winner_method": winner_method,
                    "best_a_index": judgment["best_a_index"],
                    "best_b_index": judgment["best_b_index"],
                    "best_a_caption": best_a_caption,
                    "best_b_caption": best_b_caption,
                    "plain_usable_count": usable_by_method["plain"],
                    "guided_usable_count": usable_by_method["guided"],
                    "confidence": judgment["confidence"],
                    "reason": judgment["reason"],
                    "raw_response": raw,
                }
            except Exception as exc:
                output_row = {
                    "image": generation["image"],
                    "image_id": image_id,
                    "source_index": index,
                    "mapping": mapping,
                    "failed": True,
                    "error": repr(exc),
                }
                print(f"[judge] failed {image_id}: {exc}")
            append_jsonl(output_path, output_row)
            if not output_row.get("failed"):
                existing[image_id] = output_row
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [center - margin, center + margin]


def bootstrap_mean_ci(values: list[float], seed: int, draws: int = 20_000) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def candidate_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    all_candidates = [candidate for row in rows for candidate in row[key]]
    per_image_unique = [
        len({normalize_space(candidate).lower() for candidate in row[key]})
        for row in rows
    ]
    blank = sum(not normalize_space(candidate) for candidate in all_candidates)
    meta_pattern = re.compile(
        r"\b(?:sorry|cannot|can't|unable|caption:|here(?:'s| is)|the image (?:shows|depicts))\b",
        flags=re.IGNORECASE,
    )
    return {
        "candidate_count": len(all_candidates),
        "blank_count": blank,
        "meta_or_literal_lead_count": sum(bool(meta_pattern.search(x)) for x in all_candidates),
        "mean_words": statistics.fmean(word_count(x) for x in all_candidates),
        "median_words": statistics.median(word_count(x) for x in all_candidates),
        "mean_unique_candidates_per_image": statistics.fmean(per_image_unique),
        "images_with_8_unique_candidates": sum(value == NUM_CANDIDATES for value in per_image_unique),
    }


def subgroup_stats(
    image_ids: set[str],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [row for row in judgments if image_id_for(row) in image_ids]
    wins = Counter(row["winner_method"] for row in selected)
    decisive = wins["plain"] + wins["guided"]
    return {
        "images": len(selected),
        "plain_wins": wins["plain"],
        "guided_wins": wins["guided"],
        "ties": wins["tie"],
        "guided_decisive_rate": wins["guided"] / decisive if decisive else math.nan,
        "mean_plain_usable": statistics.fmean(row["plain_usable_count"] for row in selected)
        if selected
        else math.nan,
        "mean_guided_usable": statistics.fmean(row["guided_usable_count"] for row in selected)
        if selected
        else math.nan,
    }


def render_review_html(
    generations: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
) -> str:
    cards: list[str] = []
    for row in generations:
        judgment = judgments[image_id_for(row)]
        plain_items = "".join(f"<li>{html.escape(x)}</li>" for x in row["plain_candidates"])
        guided_items = "".join(f"<li>{html.escape(x)}</li>" for x in row["guided_candidates"])
        cards.append(
            f"""
<article>
  <h2>{html.escape(image_id_for(row))} — winner: {html.escape(judgment["winner_method"])}</h2>
  <img src="file://{html.escape(row["image"])}" loading="lazy">
  <p><b>Description:</b> {html.escape(row["description"])}</p>
  <p><b>Humor cue:</b> {html.escape(row["humor_cue"] or "(empty)")}</p>
  <p><b>Judge:</b> {html.escape(judgment["reason"])}</p>
  <div class="columns">
    <section><h3>Plain ({judgment["plain_usable_count"]}/8 usable)</h3><ol>{plain_items}</ol></section>
    <section><h3>Guided ({judgment["guided_usable_count"]}/8 usable)</h3><ol>{guided_items}</ol></section>
  </div>
</article>"""
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Base 3B guidance A/B review</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1400px;margin:auto;padding:20px;background:#f5f5f5}}
article{{background:white;padding:18px;margin:18px 0;border-radius:10px}}
img{{max-width:520px;max-height:420px;object-fit:contain}}
.columns{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
li{{margin:.45em 0}} @media(max-width:800px){{.columns{{grid-template-columns:1fr}}}}
</style></head><body><h1>Base 3B Plain vs Guided — 1000 images</h1>{''.join(cards)}</body></html>"""


def report(
    rows: list[dict[str, Any]],
    guidance_path: Path,
    generations_path: Path,
    judgments_path: Path,
    output_dir: Path,
    base_seed: int,
) -> dict[str, Any]:
    guidance = existing_successes(guidance_path)
    generations_by_id = existing_successes(generations_path)
    judgments_by_id = existing_successes(judgments_path)
    for label, values in (
        ("guidance", guidance),
        ("generations", generations_by_id),
        ("judgments", judgments_by_id),
    ):
        if len(values) != len(rows):
            raise RuntimeError(f"{label} incomplete: expected {len(rows)}, got {len(values)}")
    ordered_generations = [generations_by_id[image_id_for(row)] for row in rows]
    ordered_judgments = [judgments_by_id[image_id_for(row)] for row in rows]

    settings_errors: list[str] = []
    expected_sampling = {
        "num_candidates": NUM_CANDIDATES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
    }
    for index, generation in enumerate(ordered_generations):
        if generation.get("generator_model") != str(BASE_MODEL):
            settings_errors.append(f"{image_id_for(generation)} wrong generator")
        if generation.get("adapter") is not None:
            settings_errors.append(f"{image_id_for(generation)} adapter is not null")
        if generation.get("plain_prompt") != BASE_PROMPT:
            settings_errors.append(f"{image_id_for(generation)} altered plain prompt")
        if not str(generation.get("guided_prompt") or "").endswith(BASE_PROMPT):
            settings_errors.append(f"{image_id_for(generation)} altered guided base prompt")
        if generation.get("sampling") != expected_sampling:
            settings_errors.append(f"{image_id_for(generation)} wrong sampling settings")
        if generation.get("seed") != seed_for_image(index, base_seed):
            settings_errors.append(f"{image_id_for(generation)} wrong seed")
        if len(generation.get("plain_candidates") or []) != NUM_CANDIDATES:
            settings_errors.append(f"{image_id_for(generation)} wrong plain candidate count")
        if len(generation.get("guided_candidates") or []) != NUM_CANDIDATES:
            settings_errors.append(f"{image_id_for(generation)} wrong guided candidate count")
    if settings_errors:
        raise RuntimeError("Experiment integrity failure: " + "; ".join(settings_errors[:10]))

    wins = Counter(row["winner_method"] for row in ordered_judgments)
    decisive = wins["plain"] + wins["guided"]
    guided_rate = wins["guided"] / decisive if decisive else math.nan
    binomial_p = (
        float(binomtest(wins["guided"], decisive, p=0.5, alternative="two-sided").pvalue)
        if decisive
        else math.nan
    )
    usable_differences = [
        row["guided_usable_count"] - row["plain_usable_count"]
        for row in ordered_judgments
    ]
    try:
        usable_wilcoxon_p = float(
            wilcoxon(usable_differences, zero_method="pratt", alternative="two-sided").pvalue
        )
    except ValueError:
        usable_wilcoxon_p = 1.0
    cue_ids = {key for key, value in guidance.items() if value.get("humor_cue")}
    no_cue_ids = set(guidance) - cue_ids

    plain_first = [
        row
        for row in ordered_judgments
        if row["mapping"].get("A") == "plain"
    ]
    guided_first = [
        row
        for row in ordered_judgments
        if row["mapping"].get("A") == "guided"
    ]
    a_wins = sum(row["winner_group"] == "A" for row in ordered_judgments)
    b_wins = sum(row["winner_group"] == "B" for row in ordered_judgments)

    ci = wilson_interval(wins["guided"], decisive)
    mean_usable_diff = statistics.fmean(usable_differences)
    if decisive and ci[0] > 0.5 and binomial_p < 0.05:
        conclusion = "Guided improves best-of-8 caption quality under the preregistered blind-judge endpoint."
    elif decisive and ci[1] < 0.5 and binomial_p < 0.05:
        conclusion = "Guided reduces best-of-8 caption quality under the preregistered blind-judge endpoint."
    else:
        conclusion = "The experiment does not establish a statistically reliable best-of-8 quality difference."

    summary = {
        "completed_at_utc": utc_now(),
        "integrity": {
            "num_images": len(rows),
            "unique_images": len({row["image"] for row in rows}),
            "generator_model": str(BASE_MODEL),
            "guidance_and_judge_model": str(GUIDANCE_MODEL),
            "adapter_used": False,
            "clip_reranker_used": False,
            "training_or_finetuning_used": False,
            "python": str(Path(sys.executable).resolve()),
            "base_prompt": BASE_PROMPT,
            "sampling": expected_sampling,
            "same_seed_per_image_across_methods": True,
            "base_seed": base_seed,
            "position_balance": {
                "plain_as_group_a": len(plain_first),
                "guided_as_group_a": len(guided_first),
                "group_a_wins": a_wins,
                "group_b_wins": b_wins,
            },
            "settings_errors": settings_errors,
        },
        "guidance": {
            "rows": len(guidance),
            "nonempty_humor_cues": len(cue_ids),
            "empty_humor_cues": len(no_cue_ids),
            "rejected_cues": sum(
                bool(value.get("cue_rejected_reason")) for value in guidance.values()
            ),
            "rejection_reasons": Counter(
                value.get("cue_rejected_reason")
                for value in guidance.values()
                if value.get("cue_rejected_reason")
            ),
        },
        "automatic_output_stats": {
            "plain": candidate_stats(ordered_generations, "plain_candidates"),
            "guided": candidate_stats(ordered_generations, "guided_candidates"),
        },
        "primary_blind_judge_endpoint": {
            "plain_wins": wins["plain"],
            "guided_wins": wins["guided"],
            "ties": wins["tie"],
            "decisive_images": decisive,
            "guided_decisive_rate": guided_rate,
            "guided_rate_wilson_95pct": ci,
            "two_sided_exact_binomial_p": binomial_p,
        },
        "usable_candidate_endpoint": {
            "mean_plain_usable_of_8": statistics.fmean(
                row["plain_usable_count"] for row in ordered_judgments
            ),
            "mean_guided_usable_of_8": statistics.fmean(
                row["guided_usable_count"] for row in ordered_judgments
            ),
            "mean_guided_minus_plain": mean_usable_diff,
            "bootstrap_95pct_ci": bootstrap_mean_ci(
                usable_differences,
                seed=base_seed + 2_000_001,
            ),
            "two_sided_paired_wilcoxon_p": usable_wilcoxon_p,
        },
        "subgroups": {
            "nonempty_humor_cue": subgroup_stats(cue_ids, ordered_judgments),
            "empty_humor_cue_description_only": subgroup_stats(no_cue_ids, ordered_judgments),
        },
        "conclusion": conclusion,
        "limitation": (
            "Qwen2.5-VL-7B-Instruct produced the guidance and also served as the blind evaluator. "
            "Although the evaluator never saw the guidance or method labels and A/B position was balanced, "
            "shared-model preference remains a possible source of bias; human blind evaluation would be the "
            "strongest independent confirmation."
        ),
    }
    # Convert Counter to ordinary JSON object.
    summary["guidance"]["rejection_reasons"] = dict(summary["guidance"]["rejection_reasons"])
    write_json(output_dir / "summary.json", summary)

    markdown = f"""# Base Qwen2.5-VL-3B Plain vs Guided — 1000-image A/B experiment

## Conclusion

{conclusion}

## Integrity

- Images: {len(rows)} unique images from `data/processed/sft_test.jsonl`
- Generator: `{BASE_MODEL}` (base model only; no adapter)
- Guidance extractor: `{GUIDANCE_MODEL}`
- Candidates: 8 per method per image
- Prompt: `{BASE_PROMPT}`
- Sampling: temperature={TEMPERATURE}, top_p={TOP_P}, max_new_tokens={MAX_NEW_TOKENS}, repetition_penalty={REPETITION_PENALTY}
- Randomness: same per-image seed for Plain and Guided; base seed={base_seed}
- CLIP reranker: not used
- Training/fine-tuning: not performed

## Primary endpoint: blind best-of-8 set comparison

The 7B judge saw only the image and two anonymous candidate sets. Plain was Group A on
{len(plain_first)} images and Guided was Group A on {len(guided_first)} images.

| Outcome | Count |
|---|---:|
| Plain wins | {wins["plain"]} |
| Guided wins | {wins["guided"]} |
| Ties | {wins["tie"]} |

- Guided decisive win rate: {guided_rate:.4f}
- 95% Wilson CI: [{ci[0]:.4f}, {ci[1]:.4f}]
- Two-sided exact binomial p-value: {binomial_p:.6g}
- Group-A wins / Group-B wins: {a_wins} / {b_wins}

## Usable candidates

- Plain mean usable candidates: {summary["usable_candidate_endpoint"]["mean_plain_usable_of_8"]:.3f} / 8
- Guided mean usable candidates: {summary["usable_candidate_endpoint"]["mean_guided_usable_of_8"]:.3f} / 8
- Guided − Plain: {mean_usable_diff:.3f}
- Bootstrap 95% CI: [{summary["usable_candidate_endpoint"]["bootstrap_95pct_ci"][0]:.3f}, {summary["usable_candidate_endpoint"]["bootstrap_95pct_ci"][1]:.3f}]
- Paired Wilcoxon p-value: {usable_wilcoxon_p:.6g}

## Guidance coverage

- Non-empty accepted humor cue: {len(cue_ids)}
- Empty humor cue: {len(no_cue_ids)}
- Rejected by deterministic safety filter: {summary["guidance"]["rejected_cues"]}

## Important limitation

{summary["limitation"]}
"""
    write_text(output_dir / "REPORT.md", markdown)
    write_text(
        output_dir / "comparison_review.html",
        render_review_html(ordered_generations, judgments_by_id),
    )
    return summary


def manifest(
    input_path: Path,
    source_test: Path,
    output_dir: Path,
    subset_validation: dict[str, Any],
    base_seed: int,
) -> dict[str, Any]:
    value = {
        "created_at_utc": utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": str(Path(sys.executable).resolve()),
        "input_jsonl": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "source_test_jsonl": str(source_test.resolve()),
        "source_test_sha256": sha256_file(source_test),
        "subset_validation": subset_validation,
        "generator_model": str(BASE_MODEL),
        "guidance_model": str(GUIDANCE_MODEL),
        "adapter": None,
        "clip_reranker": None,
        "training": False,
        "base_prompt": BASE_PROMPT,
        "guided_template": "Image description: {description}\\nHumor cue: {humor_cue}\\n\\n" + BASE_PROMPT,
        "sampling": {
            "num_candidates": NUM_CANDIDATES,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "repetition_penalty": REPETITION_PENALTY,
        },
        "base_seed": base_seed,
        "image_pixels": {
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
        "outputs": {
            "guidance": str((output_dir / "guidance_7b.jsonl").resolve()),
            "generations": str((output_dir / "paired_generations.jsonl").resolve()),
            "judgments": str((output_dir / "blind_judgments_7b.jsonl").resolve()),
            "summary": str((output_dir / "summary.json").resolve()),
        },
    }
    write_json(output_dir / "manifest.json", value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the strict 1000-image Base 3B Plain-vs-Guided A/B experiment."
    )
    parser.add_argument(
        "--stage",
        choices=("validate", "extract", "generate", "judge", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("outputs/evaluations/base3b_guidance_comparison_1000/eval_1000_unique.jsonl"),
    )
    parser.add_argument(
        "--source-test-jsonl",
        type=Path,
        default=Path("data/processed/sft_test.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluations/base3b_guidance_comparison_1000"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    rows = read_jsonl(args.input_jsonl)
    subset_validation = validate_subset(rows, args.source_test_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest(args.input_jsonl, args.source_test_jsonl, args.output_dir, subset_validation, args.seed)
    print(json.dumps(subset_validation, ensure_ascii=False))
    if args.stage == "validate":
        return

    guidance_path = args.output_dir / "guidance_7b.jsonl"
    generations_path = args.output_dir / "paired_generations.jsonl"
    judgments_path = args.output_dir / "blind_judgments_7b.jsonl"

    if args.stage in ("extract", "all"):
        extract_guidance(rows, guidance_path)
    if args.stage in ("generate", "all"):
        generate_pairs(rows, guidance_path, generations_path, args.seed)
    if args.stage in ("judge", "all"):
        judge_pairs(rows, generations_path, judgments_path, args.seed)
    if args.stage in ("report", "all"):
        summary = report(
            rows,
            guidance_path,
            generations_path,
            judgments_path,
            args.output_dir,
            args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
