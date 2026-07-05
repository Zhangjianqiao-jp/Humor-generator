#!/usr/bin/env python
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_guided_sft_pipeline as core
from src.training.guided_humor_sft_dataset import build_training_prompt
from src.training.sft_dataset import DEFAULT_SFT_PROMPT, clean_generated_caption


OUTPUT = Path("outputs/guided_sft_pipeline")
DATA = OUTPUT / "data"
SEED = 260618
PILOT_TRAIN_POOL = 800
PILOT_VAL_POOL = 320
FULL_TRAIN_POOL = 5000
FULL_VAL_POOL = 1000
TEACHER_CANDIDATES = 4


TEACHER_PROMPT_TEMPLATE = """Use the conservative visual notes only when they agree with the image.
Do not mention the notes. Do not invent unsupported identity, profession, relationship,
emotion, intention, or hidden story.

Image description: {description}
Humor cue: {humor_cue}

{base_prompt}"""


TARGET_AUDIT_TEMPLATE = """You are selecting a high-quality supervised target for humorous image-caption training.

Judge the candidates against the attached image. Do not rewrite them.
Reject unsupported visual facts, guessed identity/profession/relationship/emotion/intention,
generic dialogue, literal non-jokes, broken language, explanations, and multi-part text.

Select the best candidate and score only that candidate.

Candidates:
{candidates}

Return only valid JSON:
{{
  "best_index": 1,
  "visual_grounding": 1,
  "naturalness": 1,
  "humor": 1,
  "format": 1,
  "overall": 1,
  "reason": "brief reason"
}}

Indices are 1-{n}. Scores are integers 1-5. A 4 means clearly good; 5 is rare."""


def source_score(row: dict[str, Any]) -> float:
    return float(row.get("meta", {}).get("score") or 0)


def choose_unique_images(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in core.read_jsonl(path):
        key = core.image_id(row)
        image = Path(str(row.get("image") or ""))
        if not image.is_file():
            continue
        if key not in best or source_score(row) > source_score(best[key]):
            best[key] = row
    ranked = []
    for row in best.values():
        tie = hashlib.sha256(f"{seed}:{core.image_id(row)}".encode()).hexdigest()
        # Source score is only a weak proxy that the image elicited humor. It is
        # capped and never used as the training target quality judgment.
        ranked.append((min(source_score(row), 100.0), tie, row))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in ranked[:limit]]


def prepare_image_pools() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    specs = (
        ("train", Path("data/processed/sft_train.jsonl"), FULL_TRAIN_POOL, PILOT_TRAIN_POOL, SEED),
        ("val", Path("data/processed/sft_val.jsonl"), FULL_VAL_POOL, PILOT_VAL_POOL, SEED + 1),
    )
    summary = {}
    for split, source, limit, pilot_limit, seed in specs:
        rows = choose_unique_images(source, limit, seed)
        prepared = []
        for index, row in enumerate(rows):
            item = copy.deepcopy(row)
            item["_teacher_pipeline"] = {
                "phase": "pilot" if index < pilot_limit else "full",
                "pool_index": index,
                "source_score_proxy": source_score(row),
            }
            prepared.append(item)
        path = DATA / f"teacher_{split}_image_pool.jsonl"
        core.write_jsonl(path, prepared)
        summary[split] = {
            "source": str(source),
            "selected_unique_images": len(prepared),
            "pilot": min(pilot_limit, len(prepared)),
            "full_extra": max(0, len(prepared) - pilot_limit),
        }
    core.write_json(OUTPUT / "teacher_pool_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def phase_rows(split: str, include_full: bool) -> list[dict[str, Any]]:
    rows = core.read_jsonl(DATA / f"teacher_{split}_image_pool.jsonl")
    if include_full:
        return rows
    return [row for row in rows if row["_teacher_pipeline"]["phase"] == "pilot"]


def completed(path: Path) -> set[str]:
    return {
        core.image_id(row)
        for row in core.read_jsonl(path)
        if not row.get("failed")
    }


def extract_guidance(split: str, include_full: bool) -> None:
    rows = phase_rows(split, include_full)
    output = DATA / f"teacher_{split}_guidance.jsonl"
    done = completed(output)
    pending = [row for row in rows if core.image_id(row) not in done]
    if not pending:
        print(f"[teacher-guidance] {split} complete")
        return
    model, processor, vision = core.load_vlm(core.ANALYST_MODEL)
    try:
        for row in tqdm(pending, desc=f"teacher guidance {split}", dynamic_ncols=True):
            try:
                raw = core.greedy_vlm(
                    model,
                    processor,
                    vision,
                    str(Path(str(row["image"])).resolve()),
                    core.GUIDANCE_PROMPT,
                    max_new_tokens=192,
                )
                parsed = core.extract_json(raw)
                description, genericized = core.genericize_description(parsed.get("description"))
                if not description:
                    raise ValueError("empty description")
                cue, rejected = core.clean_cue(parsed.get("humor_cue"))
                value = {
                    "image": str(Path(str(row["image"])).resolve()),
                    "image_id": core.image_id(row),
                    "phase": row["_teacher_pipeline"]["phase"],
                    "description": description,
                    "description_genericized": genericized,
                    "humor_cue": cue,
                    "cue_rejected_reason": rejected,
                    "extractor_model": str(core.ANALYST_MODEL),
                    "raw_response": raw,
                }
            except Exception as exc:
                if isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc):
                    raise
                value = {
                    "image": row.get("image"),
                    "image_id": core.image_id(row),
                    "phase": row["_teacher_pipeline"]["phase"],
                    "failed": True,
                    "error": repr(exc),
                }
            core.append_jsonl(output, value)
    finally:
        core.unload(model, processor)


def guidance_map(split: str) -> dict[str, dict[str, Any]]:
    return core.map_by_id(DATA / f"teacher_{split}_guidance.jsonl")


def sample_teacher(
    model: Any,
    processor: Any,
    vision: Any,
    image: str,
    prompt: str,
    seed: int,
) -> list[str]:
    inputs = core.prepare_inputs(model, processor, vision, image, prompt)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=TEACHER_CANDIDATES,
            max_new_tokens=core.MAX_NEW_TOKENS,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.05,
        )
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [clean_generated_caption(value, prompt=prompt) for value in decoded]


def generate_teacher_candidates(split: str, include_full: bool) -> None:
    rows = phase_rows(split, include_full)
    guidance = guidance_map(split)
    output = DATA / f"teacher_{split}_candidates.jsonl"
    done = completed(output)
    pending = [
        row
        for row in rows
        if core.image_id(row) in guidance and core.image_id(row) not in done
    ]
    if not pending:
        print(f"[teacher-generate] {split} complete")
        return
    model, processor, vision = core.load_vlm(core.ANALYST_MODEL)
    try:
        for row in tqdm(pending, desc=f"teacher candidates {split}", dynamic_ncols=True):
            key = core.image_id(row)
            context = guidance[key]
            prompt = TEACHER_PROMPT_TEMPLATE.format(
                description=context["description"],
                humor_cue=context.get("humor_cue", ""),
                base_prompt=DEFAULT_SFT_PROMPT,
            )
            try:
                candidates = sample_teacher(
                    model,
                    processor,
                    vision,
                    str(Path(str(row["image"])).resolve()),
                    prompt,
                    SEED + 100_000 + int(row["_teacher_pipeline"]["pool_index"]),
                )
                value = {
                    "image": str(Path(str(row["image"])).resolve()),
                    "image_id": key,
                    "phase": row["_teacher_pipeline"]["phase"],
                    "prompt": prompt,
                    "candidates": candidates,
                    "teacher_model": str(core.ANALYST_MODEL),
                }
            except Exception as exc:
                if isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc):
                    raise
                value = {
                    "image": row.get("image"),
                    "image_id": key,
                    "phase": row["_teacher_pipeline"]["phase"],
                    "failed": True,
                    "error": repr(exc),
                }
            core.append_jsonl(output, value)
    finally:
        core.unload(model, processor)


def audit_result(parsed: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
    try:
        best_index = int(parsed.get("best_index", 0))
    except (TypeError, ValueError):
        best_index = 0
    scores = {}
    for key in ("visual_grounding", "naturalness", "humor", "format", "overall"):
        try:
            scores[key] = max(0, min(5, int(round(float(parsed.get(key, 0))))))
        except (TypeError, ValueError):
            scores[key] = 0
    valid_index = 1 <= best_index <= len(candidates)
    caption = candidates[best_index - 1].strip() if valid_index else ""
    words = re.findall(r"\b[\w'-]+\b", caption)
    mechanical_ok = (
        bool(caption)
        and 3 <= len(words) <= 20
        and len(caption) <= 140
        and "\n" not in caption
        and ";" not in caption
    )
    passed = (
        valid_index
        and mechanical_ok
        and scores["visual_grounding"] >= 4
        and scores["naturalness"] >= 4
        and scores["humor"] >= 3
        and scores["format"] >= 4
        and scores["overall"] >= 4
    )
    return {
        "best_index": best_index,
        "caption": caption,
        "scores": scores,
        "mechanical_ok": mechanical_ok,
        "passed": passed,
        "reason": core.normalize_space(parsed.get("reason"))[:200],
    }


def audit_teacher_targets(split: str, include_full: bool) -> None:
    allowed = {"pilot", "full"} if include_full else {"pilot"}
    candidates = [
        row
        for row in core.read_jsonl(DATA / f"teacher_{split}_candidates.jsonl")
        if not row.get("failed") and row.get("phase") in allowed
    ]
    output = DATA / f"teacher_{split}_targets.jsonl"
    done = completed(output)
    pending = [row for row in candidates if core.image_id(row) not in done]
    if not pending:
        print(f"[teacher-audit] {split} complete")
        return
    model, processor, vision = core.load_vlm(core.ANALYST_MODEL)
    try:
        for row in tqdm(pending, desc=f"teacher audit {split}", dynamic_ncols=True):
            candidate_block = "\n".join(
                f"{index}. {core.normalize_space(value)}"
                for index, value in enumerate(row["candidates"], 1)
            )
            prompt = TARGET_AUDIT_TEMPLATE.format(
                candidates=candidate_block,
                n=len(row["candidates"]),
            )
            try:
                raw = core.greedy_vlm(
                    model,
                    processor,
                    vision,
                    row["image"],
                    prompt,
                    max_new_tokens=192,
                )
                selected = audit_result(core.extract_json(raw), row["candidates"])
                value = {
                    "image": row["image"],
                    "image_id": core.image_id(row),
                    "phase": row["phase"],
                    **selected,
                    "candidates": row["candidates"],
                    "auditor_model": str(core.ANALYST_MODEL),
                    "raw_response": raw,
                }
            except Exception as exc:
                if isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc):
                    raise
                value = {
                    "image": row.get("image"),
                    "image_id": core.image_id(row),
                    "phase": row.get("phase"),
                    "failed": True,
                    "error": repr(exc),
                }
            core.append_jsonl(output, value)
    finally:
        core.unload(model, processor)


def replace_target(row: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(row)
    replaced = False
    for message in value.get("messages", []):
        if message.get("role") != "assistant":
            continue
        message["content"] = [{"type": "text", "text": target["caption"]}]
        replaced = True
        break
    if not replaced:
        value.setdefault("messages", []).append(
            {"role": "assistant", "content": [{"type": "text", "text": target["caption"]}]}
        )
    value["meta"] = {
        **value.get("meta", {}),
        "teacher_model": str(core.ANALYST_MODEL),
        "teacher_audit_scores": target["scores"],
        "original_caption_not_used": True,
    }
    return value


def target_map(split: str, include_full: bool) -> dict[str, dict[str, Any]]:
    allowed = {"pilot", "full"} if include_full else {"pilot"}
    return {
        core.image_id(row): row
        for row in core.read_jsonl(DATA / f"teacher_{split}_targets.jsonl")
        if not row.get("failed") and row.get("passed") and row.get("phase") in allowed
    }


def materialize(pilot: bool) -> dict[str, int]:
    include_full = not pilot
    train_pool = {core.image_id(row): row for row in phase_rows("train", include_full)}
    val_pool = {core.image_id(row): row for row in phase_rows("val", include_full)}
    train_targets = target_map("train", include_full)
    val_targets = target_map("val", include_full)
    train_guidance = guidance_map("train")
    val_guidance = guidance_map("val")

    train_ids = [
        key for key in train_pool if key in train_targets and key in train_guidance
    ]
    val_ids = [
        key for key in val_pool if key in val_targets and key in val_guidance
    ]
    if pilot:
        minimum_val = core.MIN_PILOT_VAL + core.MIN_PILOT_GATE
        if len(train_ids) < core.MIN_PILOT_TRAIN:
            raise RuntimeError(f"Only {len(train_ids)} teacher train targets passed")
        if len(val_ids) < minimum_val:
            raise RuntimeError(f"Only {len(val_ids)} teacher val targets passed")
        train_ids = train_ids[: core.PILOT_TRAIN_TARGET]
        internal_val = val_ids[: core.PILOT_VAL_TARGET]
        gate = val_ids[
            core.PILOT_VAL_TARGET : core.PILOT_VAL_TARGET + core.PILOT_GATE_TARGET
        ]
        groups = {
            "pilot_train": (train_ids, train_pool, train_targets, train_guidance),
            "pilot_val": (internal_val, val_pool, val_targets, val_guidance),
            "pilot_gate": (gate, val_pool, val_targets, val_guidance),
        }
    else:
        if len(train_ids) < core.MIN_FULL_TRAIN:
            raise RuntimeError(f"Only {len(train_ids)} full teacher train targets passed")
        if len(val_ids) < core.MIN_FULL_VAL:
            raise RuntimeError(f"Only {len(val_ids)} full teacher val targets passed")
        train_ids = train_ids[: core.FULL_TRAIN_TARGET]
        internal_val = val_ids[: core.FULL_VAL_TARGET]
        groups = {
            "full_train": (train_ids, train_pool, train_targets, train_guidance),
            "full_val": (internal_val, val_pool, val_targets, val_guidance),
        }

    summary = {}
    for name, (ids, pool, targets, guidance) in groups.items():
        training_rows = [replace_target(pool[key], targets[key]) for key in ids]
        core.write_jsonl(DATA / f"{name}.jsonl", training_rows)
        core.write_jsonl(DATA / f"{name}_guidance.jsonl", [guidance[key] for key in ids])
        summary[name] = len(ids)
    core.write_json(
        OUTPUT / ("teacher_pilot_data_summary.json" if pilot else "teacher_full_data_summary.json"),
        summary,
    )
    print(json.dumps(summary, indent=2))
    return summary


def teacher_stats(split: str, include_full: bool) -> dict[str, Any]:
    allowed = {"pilot", "full"} if include_full else {"pilot"}
    targets = [
        row
        for row in core.read_jsonl(DATA / f"teacher_{split}_targets.jsonl")
        if not row.get("failed") and row.get("phase") in allowed
    ]
    return {
        "audited": len(targets),
        "passed": sum(bool(row.get("passed")) for row in targets),
        "failed_parse": sum(bool(row.get("failed")) for row in targets),
        "score_distributions": {
            key: dict(
                Counter(str(row.get("scores", {}).get(key, 0)) for row in targets)
            )
            for key in ("visual_grounding", "naturalness", "humor", "format", "overall")
        },
    }


def process_teacher_data(include_full: bool) -> None:
    for split in ("train", "val"):
        extract_guidance(split, include_full)
        generate_teacher_candidates(split, include_full)
        audit_teacher_targets(split, include_full)
    core.write_json(
        OUTPUT / ("teacher_full_quality_summary.json" if include_full else "teacher_pilot_quality_summary.json"),
        {
            split: teacher_stats(split, include_full)
            for split in ("train", "val")
        },
    )


def manifest() -> None:
    core.write_json(
        OUTPUT / "teacher_pipeline_manifest.json",
        {
            "base_generator": str(core.BASE_MODEL),
            "teacher_and_auditor": str(core.ANALYST_MODEL),
            "old_adapter_loaded_for_training": False,
            "training_target": "7B best-of-4 synthetic caption, image-conditionally audited",
            "original_hic_caption_used_as_target": False,
            "guidance_extractor_sees_target_caption": False,
            "test_split_used": False,
            "pilot_gate": "held-out sft_val, blind Base 3B vs fresh pilot LoRA",
            "full_training": "starts only if pilot gate passes",
            "base_prompt": DEFAULT_SFT_PROMPT,
        },
    )


def run_all() -> None:
    if Path(sys.executable).resolve() != core.PYTHON.resolve():
        raise RuntimeError(f"Use required Python: {core.PYTHON}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest()
    prepare_image_pools()

    process_teacher_data(include_full=False)
    materialize(pilot=True)
    core.run_training(Path("configs/guided_humor_lora_pilot.yaml"))

    base_candidates = core.generate_gate("base", adapter=None)
    lora_candidates = core.generate_gate(
        "pilot_lora",
        adapter=OUTPUT / "pilot_lora" / "final_lora",
    )
    judgments = core.judge_gate(base_candidates, lora_candidates)
    if not core.pilot_gate(judgments):
        print("[teacher-pipeline] pilot failed; full data generation/training stopped")
        return

    process_teacher_data(include_full=True)
    materialize(pilot=False)
    core.run_training(Path("configs/guided_humor_lora_full.yaml"))
    (OUTPUT / "FULL_TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    print("[teacher-pipeline] full training complete")


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
