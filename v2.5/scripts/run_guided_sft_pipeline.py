#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from scipy.stats import binomtest
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.guided_humor_sft_dataset import build_training_prompt
from src.training.sft_dataset import DEFAULT_SFT_PROMPT, extract_caption


PYTHON = Path("/home/zhang.jianqiao/miniconda3/envs/humor/bin/python")
BASE_MODEL = Path("/home/zhang.jianqiao/models/Qwen2.5-VL-3B-Instruct")
ANALYST_MODEL = Path("/home/zhang.jianqiao/models/Qwen2.5-VL-7B-Instruct")
OUTPUT = Path("outputs/guided_sft_pipeline")
DATA_DIR = OUTPUT / "data"
SEED = 260618

PILOT_POOL_TRAIN = 1200
PILOT_POOL_VAL = 400
FULL_POOL_TRAIN = 6000
FULL_POOL_VAL = 800
PILOT_TRAIN_TARGET = 512
PILOT_VAL_TARGET = 64
PILOT_GATE_TARGET = 128
FULL_TRAIN_TARGET = 3000
FULL_VAL_TARGET = 256
MIN_PILOT_TRAIN = 384
MIN_PILOT_VAL = 48
MIN_PILOT_GATE = 96
MIN_FULL_TRAIN = 1500
MIN_FULL_VAL = 128

NUM_EVAL_CANDIDATES = 4
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_NEW_TOKENS = 48
REPETITION_PENALTY = 1.05
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28


SCREEN_PROMPT = """You are a strict quality auditor for a supervised humorous image-caption dataset.

Evaluate the supplied target caption against the attached image. Do not rewrite it.
The caption must be a short, natural, image-specific humorous caption suitable as a
high-quality training target.

Penalize:
- unsupported visual facts, identity, profession, relationship, emotion, intention, or hidden story;
- generic dialogue that could fit many images;
- literal descriptions with no humor;
- broken translation, repetition, platform/meta language, meme-template instructions;
- multiple captions, explanations, labels, or long multi-part text.

Return only valid JSON:
{
  "visual_grounding": 1,
  "naturalness": 1,
  "humor": 1,
  "format": 1,
  "overall": 1,
  "reason": "brief reason"
}

Use integer scores 1-5. A score of 4 means clearly good; 5 should be rare.

Target caption:
{caption}"""


GUIDANCE_PROMPT = """You are a conservative visual analyst supporting image-caption training.

Return only valid JSON with exactly this schema:
{
  "description": "one conservative sentence describing only clearly visible content",
  "humor_cue": "one high-confidence visible incongruity, or an empty string"
}

Rules:
- The description must be exactly one sentence, literal, concise, and accurate.
- Use generic labels such as person, animal, clothing, and object; never infer a profession or relationship.
- The humor_cue may contain at most one plain visual observation.
- A cue is allowed only for a directly visible unusual physical detail, clear size difference,
  action contrast, composition relationship, or visible role/object mismatch.
- Do not guess identity, age, profession, social role, relationship, emotion, intention,
  motivation, thought, or hidden story.
- Do not infer stealing, guarding, escaping, pretending, waiting, planning, helping,
  competing, or trying to do something.
- Do not introduce anything not clearly visible.
- Do not use metaphor, fictional characters, pop culture, or "looks like/as if" comparisons.
- If an allowed contrast is unmistakable, state it even if it also appears in the description.
- If no reliable visible incongruity exists, set humor_cue to an empty string.
- Do not write or suggest a final caption.
- Use English.

Valid cue: "The miniature meal is much smaller than the hand holding it."
Valid cue: "A dog is positioned behind a car steering wheel."
Invalid cue: "The dog is driving to work."

Return JSON only. No markdown or explanation."""


BLIND_JUDGE_TEMPLATE = """You are comparing two anonymous sets of humorous captions for the attached image.

Reject captions that hallucinate important visual facts or depend on unsupported identity,
profession, relationship, emotion, intention, or hidden story. Prefer genuine humor,
image specificity, natural wording, and concision.

For each set:
- count captions that are visually grounded, understandable, and at least mildly humorous;
- select its best caption;
- choose the set with the stronger best caption.

Use tie only when genuinely comparable. Do not create a new caption.

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

Indices are 1-{n}. Counts are 0-{n}. winner_group is A, B, or tie. confidence is 1-5."""


BAD_CAPTION_PATTERNS = (
    r"\b(?:upvote|downvote|karma|imgflip|meme generator|image tagged|subscribe|repost)\b",
    r"\b(?:caption|candidate|answer)\s*:",
    r"\b(?:i(?:'m| am) sorry|i (?:do not|don't) know|as an ai)\b",
    r"\b(?:music ends?|applause|inaudible)\b",
)

FORBIDDEN_GUIDANCE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:maybe|perhaps|probably|likely|possibly|seems?|appears?)\b", "speculation"),
    (r"\b(?:looks? like|as if|as though|resembl(?:e|es|ing))\b", "analogy"),
    (r"\b(?:trying|wants?|intends?|plans?|thinks?|feels?|hopes?|attempts?)\b", "intention"),
    (
        r"\b(?:steal(?:s|ing)?|guard(?:s|ing)?|escap(?:e|es|ing)|pretend(?:s|ing)?|"
        r"help(?:s|ing)?|compet(?:e|es|ing)|protect(?:s|ing)?)\b",
        "hidden story",
    ),
    (
        r"\b(?:happy|sad|angry|afraid|scared|worried|excited|confused|surprised|"
        r"embarrassed|bored|relaxed|serious|proud)\b",
        "emotion",
    ),
    (
        r"\b(?:doctor|nurse|police|officer|soldier|teacher|student|boss|employee|"
        r"chef|waiter|driver|guard|thief|husband|wife|mother|father|owner)\b",
        "identity or relationship",
    ),
    (r"[!?]", "caption-like punctuation"),
)


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def image_id(row: dict[str, Any]) -> str:
    return str(row.get("image_id") or Path(str(row.get("image") or "")).stem)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{number}")
            rows.append(value)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_json(text: str) -> dict[str, Any]:
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
        raise ValueError("Expected a JSON object")
    return value


def caption_precheck(caption: str, score: float) -> tuple[bool, str]:
    text = normalize_space(caption)
    words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)
    if score < 30:
        return False, "score_below_30"
    if not 4 <= len(words) <= 20:
        return False, "word_count"
    if not 12 <= len(text) <= 140:
        return False, "char_count"
    if "\n" in caption or ";" in caption:
        return False, "multi_part"
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in BAD_CAPTION_PATTERNS):
        return False, "meta_or_platform"
    letters = [character for character in text if character.isalpha()]
    if letters and sum(ord(character) < 128 for character in letters) / len(letters) < 0.95:
        return False, "non_latin"
    alphabetic_words = [word for word in words if any(c.isalpha() for c in word)]
    if alphabetic_words:
        uppercase = sum(word.isupper() and len(word) > 1 for word in alphabetic_words)
        if uppercase / len(alphabetic_words) > 0.6:
            return False, "mostly_uppercase"
    return True, ""


def choose_pool(source_path: Path, limit: int, seed: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for row in read_jsonl(source_path):
        key = image_id(row)
        score = float(row.get("meta", {}).get("score") or 0)
        if key not in best or score > best[key][0]:
            best[key] = (score, row)

    rejected: Counter[str] = Counter()
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    seen_captions: set[str] = set()
    for score, row in best.values():
        caption = normalize_space(extract_caption(row))
        ok, reason = caption_precheck(caption, score)
        if not ok:
            rejected[reason] += 1
            continue
        normalized = re.sub(r"\W+", " ", caption.lower()).strip()
        if normalized in seen_captions:
            rejected["duplicate_caption"] += 1
            continue
        seen_captions.add(normalized)
        tie = hashlib.sha256(f"{seed}:{image_id(row)}".encode()).hexdigest()
        candidates.append((min(score, 250.0), tie, row))

    # Score remains useful but is capped so a few platform-specific viral items
    # cannot dominate the pool. Hash tie-breaking makes selection reproducible.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in candidates[:limit]], rejected


def prepare_pools() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    specs = (
        ("train", Path("data/processed/sft_train.jsonl"), FULL_POOL_TRAIN, SEED),
        ("val", Path("data/processed/sft_val.jsonl"), FULL_POOL_VAL, SEED + 1),
    )
    summary: dict[str, Any] = {}
    for split, source, limit, seed in specs:
        rows, rejected = choose_pool(source, limit, seed)
        pilot_size = PILOT_POOL_TRAIN if split == "train" else PILOT_POOL_VAL
        output_rows = []
        for index, row in enumerate(rows):
            item = dict(row)
            item["_pipeline"] = {
                "phase": "pilot" if index < pilot_size else "full",
                "pool_index": index,
                "source_score": float(row.get("meta", {}).get("score") or 0),
            }
            output_rows.append(item)
        path = DATA_DIR / f"{split}_candidate_pool.jsonl"
        write_jsonl(path, output_rows)
        summary[split] = {
            "source": str(source),
            "source_sha256": file_sha256(source),
            "selected": len(output_rows),
            "pilot_phase": min(pilot_size, len(output_rows)),
            "full_phase": max(0, len(output_rows) - pilot_size),
            "prefilter_rejections": dict(rejected),
        }
    write_json(OUTPUT / "pool_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_vlm(model_path: Path, adapter: Path | None = None) -> tuple[Any, Any, Any]:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model: Any = base
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
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
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(model_device(model))


def greedy_vlm(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    image: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    inputs = prepare_inputs(model, processor, process_vision_info, image, prompt)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def unload(*objects: Any) -> None:
    for value in objects:
        del value
    gc.collect()
    torch.cuda.empty_cache()


def completed_ids(path: Path, include_failed: bool = False) -> set[str]:
    values = set()
    for row in read_jsonl(path):
        if include_failed or not row.get("failed"):
            values.add(image_id(row))
    return values


def rows_for_phase(path: Path, include_full: bool) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if include_full:
        return rows
    return [row for row in rows if row.get("_pipeline", {}).get("phase") == "pilot"]


def normalize_screen(value: dict[str, Any]) -> dict[str, Any]:
    scores = {}
    for key in ("visual_grounding", "naturalness", "humor", "format", "overall"):
        try:
            scores[key] = max(1, min(5, int(round(float(value.get(key, 0))))))
        except (TypeError, ValueError):
            scores[key] = 0
    passed = (
        scores["visual_grounding"] >= 4
        and scores["naturalness"] >= 4
        and scores["humor"] >= 3
        and scores["format"] >= 4
        and scores["overall"] >= 4
    )
    return {
        "scores": scores,
        "passed": passed,
        "reason": normalize_space(value.get("reason"))[:200],
    }


def screen_split(split: str, include_full: bool) -> None:
    pool_path = DATA_DIR / f"{split}_candidate_pool.jsonl"
    output_path = DATA_DIR / f"{split}_screen_7b.jsonl"
    rows = rows_for_phase(pool_path, include_full)
    done = completed_ids(output_path)
    pending = [row for row in rows if image_id(row) not in done]
    if not pending:
        print(f"[screen] {split} complete for requested phase")
        return
    model, processor, vision = load_vlm(ANALYST_MODEL)
    try:
        for row in tqdm(pending, desc=f"screen {split}", dynamic_ncols=True):
            caption = normalize_space(extract_caption(row))
            prompt = SCREEN_PROMPT.format(caption=caption)
            try:
                raw = greedy_vlm(
                    model,
                    processor,
                    vision,
                    str(Path(str(row["image"])).resolve()),
                    prompt,
                    max_new_tokens=192,
                )
                normalized = normalize_screen(extract_json(raw))
                output = {
                    "image": str(Path(str(row["image"])).resolve()),
                    "image_id": image_id(row),
                    "phase": row.get("_pipeline", {}).get("phase"),
                    "caption": caption,
                    "source_score": float(row.get("meta", {}).get("score") or 0),
                    **normalized,
                    "raw_response": raw,
                    "judge_model": str(ANALYST_MODEL),
                }
            except Exception as exc:
                if isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc):
                    raise
                output = {
                    "image": row.get("image"),
                    "image_id": image_id(row),
                    "phase": row.get("_pipeline", {}).get("phase"),
                    "failed": True,
                    "error": repr(exc),
                }
            append_jsonl(output_path, output)
    finally:
        unload(model, processor)


def genericize_description(value: Any) -> tuple[str, bool]:
    original = normalize_space(value).strip(" \"'")
    match = re.search(r"(?<=[.!?])\s+(?=[A-Z])", original)
    if match:
        original = original[: match.start()].strip()
    if original and original[-1] not in ".!?":
        original += "."
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


def clean_cue(value: Any) -> tuple[str, str | None]:
    if isinstance(value, list):
        if len(value) > 1:
            return "", "multiple cues"
        value = value[0] if value else ""
    cue = normalize_space(value).strip(" \"'")
    if not cue:
        return "", None
    if len(cue) > 220 or ";" in cue:
        return "", "too long or multiple"
    for pattern, reason in FORBIDDEN_GUIDANCE_PATTERNS:
        if re.search(pattern, cue, flags=re.IGNORECASE):
            return "", reason
    return cue.rstrip("."), None


def accepted_screen_rows(split: str, include_full: bool) -> list[dict[str, Any]]:
    rows = read_jsonl(DATA_DIR / f"{split}_screen_7b.jsonl")
    allowed = {"pilot", "full"} if include_full else {"pilot"}
    return [
        row
        for row in rows
        if not row.get("failed") and row.get("passed") and row.get("phase") in allowed
    ]


def extract_guidance_split(split: str, include_full: bool) -> None:
    rows = accepted_screen_rows(split, include_full)
    output_path = DATA_DIR / f"{split}_guidance_7b.jsonl"
    done = completed_ids(output_path)
    pending = [row for row in rows if image_id(row) not in done]
    if not pending:
        print(f"[guidance] {split} complete for requested phase")
        return
    model, processor, vision = load_vlm(ANALYST_MODEL)
    try:
        for row in tqdm(pending, desc=f"guidance {split}", dynamic_ncols=True):
            try:
                raw = greedy_vlm(
                    model,
                    processor,
                    vision,
                    str(Path(str(row["image"])).resolve()),
                    GUIDANCE_PROMPT,
                    max_new_tokens=192,
                )
                parsed = extract_json(raw)
                description, genericized = genericize_description(parsed.get("description"))
                if not description:
                    raise ValueError("empty description")
                cue, rejected = clean_cue(parsed.get("humor_cue"))
                output = {
                    "image": str(Path(str(row["image"])).resolve()),
                    "image_id": image_id(row),
                    "phase": row.get("phase"),
                    "description": description,
                    "description_genericized": genericized,
                    "humor_cue": cue,
                    "cue_rejected_reason": rejected,
                    "raw_response": raw,
                    "extractor_model": str(ANALYST_MODEL),
                }
            except Exception as exc:
                if isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc):
                    raise
                output = {
                    "image": row.get("image"),
                    "image_id": image_id(row),
                    "phase": row.get("phase"),
                    "failed": True,
                    "error": repr(exc),
                }
            append_jsonl(output_path, output)
    finally:
        unload(model, processor)


def map_by_id(path: Path, usable_only: bool = True) -> dict[str, dict[str, Any]]:
    values = {}
    for row in read_jsonl(path):
        if usable_only and row.get("failed"):
            continue
        values[image_id(row)] = row
    return values


def write_subset(
    name: str,
    rows: list[dict[str, Any]],
    guidance: dict[str, dict[str, Any]],
) -> None:
    write_jsonl(DATA_DIR / f"{name}.jsonl", rows)
    write_jsonl(DATA_DIR / f"{name}_guidance.jsonl", [guidance[image_id(row)] for row in rows])


def materialize(pilot: bool) -> dict[str, int]:
    train_pool = map_by_id(DATA_DIR / "train_candidate_pool.jsonl")
    val_pool = map_by_id(DATA_DIR / "val_candidate_pool.jsonl")
    train_guidance = map_by_id(DATA_DIR / "train_guidance_7b.jsonl")
    val_guidance = map_by_id(DATA_DIR / "val_guidance_7b.jsonl")
    train_passed = accepted_screen_rows("train", include_full=not pilot)
    val_passed = accepted_screen_rows("val", include_full=not pilot)
    train_ids = [image_id(row) for row in train_passed if image_id(row) in train_guidance]
    val_ids = [image_id(row) for row in val_passed if image_id(row) in val_guidance]

    if pilot:
        if len(train_ids) < MIN_PILOT_TRAIN:
            raise RuntimeError(f"Only {len(train_ids)} pilot train rows passed; need {MIN_PILOT_TRAIN}")
        needed_val = MIN_PILOT_VAL + MIN_PILOT_GATE
        if len(val_ids) < needed_val:
            raise RuntimeError(f"Only {len(val_ids)} pilot val rows passed; need {needed_val}")
        train_ids = train_ids[:PILOT_TRAIN_TARGET]
        val_train_ids = val_ids[:PILOT_VAL_TARGET]
        gate_ids = val_ids[PILOT_VAL_TARGET : PILOT_VAL_TARGET + PILOT_GATE_TARGET]
        write_subset("pilot_train", [train_pool[key] for key in train_ids], train_guidance)
        write_subset("pilot_val", [val_pool[key] for key in val_train_ids], val_guidance)
        write_subset("pilot_gate", [val_pool[key] for key in gate_ids], val_guidance)
        result = {
            "pilot_train": len(train_ids),
            "pilot_val": len(val_train_ids),
            "pilot_gate": len(gate_ids),
        }
    else:
        if len(train_ids) < MIN_FULL_TRAIN:
            raise RuntimeError(f"Only {len(train_ids)} full train rows passed; need {MIN_FULL_TRAIN}")
        if len(val_ids) < MIN_FULL_VAL:
            raise RuntimeError(f"Only {len(val_ids)} full val rows passed; need {MIN_FULL_VAL}")
        train_ids = train_ids[:FULL_TRAIN_TARGET]
        val_ids = val_ids[:FULL_VAL_TARGET]
        write_subset("full_train", [train_pool[key] for key in train_ids], train_guidance)
        write_subset("full_val", [val_pool[key] for key in val_ids], val_guidance)
        result = {"full_train": len(train_ids), "full_val": len(val_ids)}
    write_json(OUTPUT / ("pilot_data_summary.json" if pilot else "full_data_summary.json"), result)
    print(json.dumps(result, indent=2))
    return result


def run_training(config: Path) -> None:
    with config.open("r", encoding="utf-8") as handle:
        import yaml

        resolved = yaml.safe_load(handle)
    final_adapter = Path(resolved["output"]["final_adapter_dir"])
    if (final_adapter / "adapter_config.json").is_file():
        print(f"[train] already complete: {final_adapter}")
        return
    command = [str(PYTHON), "scripts/train_guided_humor_lora.py", "--config", str(config)]
    print("[train]", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def sample_candidates(
    model: Any,
    processor: Any,
    vision: Any,
    image: str,
    prompt: str,
    seed: int,
) -> list[str]:
    inputs = prepare_inputs(model, processor, vision, image, prompt)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=NUM_EVAL_CANDIDATES,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
        )
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    return [
        normalize_space(value)
        for value in processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]


def generate_gate(method: str, adapter: Path | None) -> Path:
    rows = read_jsonl(DATA_DIR / "pilot_gate.jsonl")
    guidance = map_by_id(DATA_DIR / "pilot_gate_guidance.jsonl")
    output_path = OUTPUT / "pilot_evaluation" / f"{method}_candidates.jsonl"
    done = completed_ids(output_path)
    pending = [row for row in rows if image_id(row) not in done]
    if not pending:
        return output_path
    model, processor, vision = load_vlm(BASE_MODEL, adapter=adapter)
    try:
        for index, row in enumerate(tqdm(pending, desc=f"generate {method}", dynamic_ncols=True)):
            context = guidance[image_id(row)]
            prompt = build_training_prompt(
                context["description"],
                context.get("humor_cue", ""),
                DEFAULT_SFT_PROMPT,
            )
            candidates = sample_candidates(
                model,
                processor,
                vision,
                str(Path(str(row["image"])).resolve()),
                prompt,
                SEED + 500_000 + int(row.get("_pipeline", {}).get("pool_index", index)),
            )
            append_jsonl(
                output_path,
                {
                    "image": str(Path(str(row["image"])).resolve()),
                    "image_id": image_id(row),
                    "method": method,
                    "prompt": prompt,
                    "candidates": candidates,
                },
            )
    finally:
        unload(model, processor)
    return output_path


def shuffled(values: list[str], seed: int) -> list[str]:
    result = list(values)
    random.Random(seed).shuffle(result)
    return result


def numbered(values: list[str]) -> str:
    return "\n".join(f"{index}. {value}" for index, value in enumerate(values, 1))


def parse_blind_judgment(raw: str) -> dict[str, Any]:
    value = extract_json(raw)
    winner = normalize_space(value.get("winner_group")).upper()
    if winner not in {"A", "B", "TIE"}:
        raise ValueError(f"bad winner: {winner}")
    result = {
        "best_a_index": int(value["best_a_index"]),
        "best_b_index": int(value["best_b_index"]),
        "usable_a_count": int(value["usable_a_count"]),
        "usable_b_count": int(value["usable_b_count"]),
        "winner_group": winner.lower() if winner == "TIE" else winner,
        "confidence": int(value.get("confidence", 0)),
        "reason": normalize_space(value.get("reason")),
    }
    for key in ("best_a_index", "best_b_index"):
        if not 1 <= result[key] <= NUM_EVAL_CANDIDATES:
            raise ValueError(f"{key} outside range")
    for key in ("usable_a_count", "usable_b_count"):
        if not 0 <= result[key] <= NUM_EVAL_CANDIDATES:
            raise ValueError(f"{key} outside range")
    return result


def judge_gate(base_path: Path, lora_path: Path) -> Path:
    base = map_by_id(base_path)
    lora = map_by_id(lora_path)
    ids = sorted(set(base) & set(lora))
    output_path = OUTPUT / "pilot_evaluation" / "blind_judgments_7b.jsonl"
    done = completed_ids(output_path)
    pending = [key for key in ids if key not in done]
    if not pending:
        return output_path
    position_ids = list(ids)
    random.Random(SEED + 700_000).shuffle(position_ids)
    base_first = set(position_ids[: len(position_ids) // 2])
    model, processor, vision = load_vlm(ANALYST_MODEL)
    try:
        for index, key in enumerate(tqdm(pending, desc="blind pilot judge", dynamic_ncols=True)):
            base_candidates = shuffled(base[key]["candidates"], SEED + 800_000 + index * 2)
            lora_candidates = shuffled(lora[key]["candidates"], SEED + 800_001 + index * 2)
            if key in base_first:
                group_a, group_b = base_candidates, lora_candidates
                mapping = {"A": "base", "B": "lora"}
            else:
                group_a, group_b = lora_candidates, base_candidates
                mapping = {"A": "lora", "B": "base"}
            prompt = BLIND_JUDGE_TEMPLATE.format(
                group_a=numbered(group_a),
                group_b=numbered(group_b),
                n=NUM_EVAL_CANDIDATES,
            )
            try:
                raw = greedy_vlm(
                    model,
                    processor,
                    vision,
                    base[key]["image"],
                    prompt,
                    max_new_tokens=256,
                )
                parsed = parse_blind_judgment(raw)
                winner = parsed["winner_group"]
                winner_method = "tie" if winner == "tie" else mapping[winner]
                usable = {
                    mapping["A"]: parsed["usable_a_count"],
                    mapping["B"]: parsed["usable_b_count"],
                }
                output = {
                    "image": base[key]["image"],
                    "image_id": key,
                    "mapping": mapping,
                    "winner_method": winner_method,
                    "base_usable_count": usable["base"],
                    "lora_usable_count": usable["lora"],
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "raw_response": raw,
                }
            except Exception as exc:
                output = {
                    "image": base[key]["image"],
                    "image_id": key,
                    "failed": True,
                    "error": repr(exc),
                }
            append_jsonl(output_path, output)
    finally:
        unload(model, processor)
    return output_path


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return center - margin, center + margin


def pilot_gate(judgments_path: Path) -> bool:
    rows = read_jsonl(judgments_path)
    valid = [row for row in rows if not row.get("failed")]
    failures = len(rows) - len(valid)
    wins = Counter(row["winner_method"] for row in valid)
    decisive = wins["base"] + wins["lora"]
    low, high = wilson(wins["lora"], decisive)
    p_value = (
        float(binomtest(wins["lora"], decisive, 0.5, alternative="two-sided").pvalue)
        if decisive
        else 1.0
    )
    usable_diff = [
        row["lora_usable_count"] - row["base_usable_count"]
        for row in valid
    ]
    mean_usable_diff = statistics.fmean(usable_diff) if usable_diff else math.nan
    passed = (
        decisive >= 80
        and low > 0.5
        and p_value < 0.05
        and mean_usable_diff >= 0
        and failures / max(len(rows), 1) <= 0.05
    )
    summary = {
        "base_wins": wins["base"],
        "lora_wins": wins["lora"],
        "ties": wins["tie"],
        "failures": failures,
        "decisive": decisive,
        "lora_decisive_rate": wins["lora"] / decisive if decisive else math.nan,
        "lora_wilson_95pct": [low, high],
        "exact_binomial_p": p_value,
        "mean_lora_minus_base_usable": mean_usable_diff,
        "gate_rules": {
            "minimum_decisive": 80,
            "wilson_lower_above": 0.5,
            "p_below": 0.05,
            "usable_difference_at_least": 0,
            "max_failure_rate": 0.05,
        },
        "passed": passed,
    }
    write_json(OUTPUT / "pilot_evaluation" / "gate_summary.json", summary)
    marker = OUTPUT / ("PILOT_PASSED" if passed else "PILOT_FAILED")
    marker.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return passed


def manifest() -> None:
    value = {
        "python": str(PYTHON),
        "base_model": str(BASE_MODEL),
        "analyst_model": str(ANALYST_MODEL),
        "old_adapter_loaded": False,
        "test_split_used": False,
        "quality_screen_sees_caption": True,
        "guidance_extractor_sees_caption": False,
        "base_prompt": DEFAULT_SFT_PROMPT,
        "pilot_gate": "blind Base-vs-fresh-LoRA comparison on held-out sft_val images",
        "full_training_rule": "run only when pilot gate passes",
        "script_sha256": file_sha256(Path(__file__)),
    }
    write_json(OUTPUT / "manifest.json", value)


def run_all() -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"Use required Python: {PYTHON}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest()
    prepare_pools()

    # Phase 1: high-quality pilot data only.
    screen_split("train", include_full=False)
    screen_split("val", include_full=False)
    extract_guidance_split("train", include_full=False)
    extract_guidance_split("val", include_full=False)
    materialize(pilot=True)
    run_training(Path("configs/guided_humor_lora_pilot.yaml"))

    base_candidates = generate_gate("base", adapter=None)
    lora_candidates = generate_gate(
        "pilot_lora",
        adapter=OUTPUT / "pilot_lora" / "final_lora",
    )
    judgments = judge_gate(base_candidates, lora_candidates)
    if not pilot_gate(judgments):
        print("[pipeline] pilot failed the preregistered gate; full training will not run")
        return

    # Phase 2: expand quality screening only after pilot success.
    screen_split("train", include_full=True)
    screen_split("val", include_full=True)
    extract_guidance_split("train", include_full=True)
    extract_guidance_split("val", include_full=True)
    materialize(pilot=False)
    run_training(Path("configs/guided_humor_lora_full.yaml"))
    (OUTPUT / "FULL_TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    print("[pipeline] full guided LoRA training complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "screen-pilot", "guidance-pilot", "materialize-pilot", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_pools()
    elif args.stage == "screen-pilot":
        screen_split("train", include_full=False)
        screen_split("val", include_full=False)
    elif args.stage == "guidance-pilot":
        extract_guidance_split("train", include_full=False)
        extract_guidance_split("val", include_full=False)
    elif args.stage == "materialize-pilot":
        materialize(pilot=True)
    else:
        run_all()


if __name__ == "__main__":
    main()
