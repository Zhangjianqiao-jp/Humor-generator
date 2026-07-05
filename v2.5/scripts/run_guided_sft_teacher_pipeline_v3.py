#!/usr/bin/env python
"""Two-pass cue extraction plus boolean target auditing.

The first conservative pass often leaves every cue empty. When that happens,
this entry point performs a second image-only pass focused narrowly on allowed
visible incongruities, then applies the same deterministic safety filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_guided_sft_pipeline as core
import scripts.run_guided_sft_teacher_pipeline_v2 as v2

pipeline = v2.pipeline


CUE_RECHECK_PROMPT = """Inspect the attached image for at most one clearly visible incongruity.

The conservative literal description from a separate image-only pass is:
{description}

Check these categories carefully:
1. a clearly unusual physical detail;
2. a large and obvious size difference;
3. two visible actions or poses that strongly contrast;
4. a composition relationship, overlap, or alignment between visible subjects or objects;
5. a visible subject/object placement or role mismatch.

Return only valid JSON:
{{
  "humor_cue": "one neutral visible observation, or an empty string"
}}

Strict rules:
- The cue is a factual visual observation, not a joke or final caption.
- Mention only objects, positions, sizes, actions, or composition clearly visible in the image.
- Do not guess identity, age, profession, social role, relationship, emotion, intention,
  motivation, thought, cause, or hidden story.
- Do not use metaphor, fictional characters, pop culture, or "looks like/as if".
- Do not infer stealing, guarding, escaping, pretending, planning, helping, competing,
  driving, speaking, or trying to do something.
- If one category is unmistakably present, return the single strongest one.
- If none is reliable, return an empty string.

Valid: "The miniature meal is much smaller than the hand holding it."
Valid: "A dog is positioned behind a car steering wheel."
Valid: "One person is standing while the surrounding people are crouching."
Invalid: "The dog is driving to work."

Return JSON only."""


def extract_guidance_two_pass(split: str, include_full: bool) -> None:
    rows = pipeline.phase_rows(split, include_full)
    output = pipeline.DATA / f"teacher_{split}_guidance.jsonl"
    done = pipeline.completed(output)
    pending = [row for row in rows if core.image_id(row) not in done]
    if not pending:
        print(f"[teacher-guidance-v3] {split} complete")
        return
    model, processor, vision = core.load_vlm(core.ANALYST_MODEL)
    try:
        for row in tqdm(pending, desc=f"teacher guidance v3 {split}", dynamic_ncols=True):
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
                recheck_raw = ""
                recheck_rejected = None
                if not cue:
                    recheck_raw = core.greedy_vlm(
                        model,
                        processor,
                        vision,
                        str(Path(str(row["image"])).resolve()),
                        CUE_RECHECK_PROMPT.format(description=description),
                        max_new_tokens=128,
                    )
                    recheck = core.extract_json(recheck_raw)
                    recheck_cue, recheck_rejected = core.clean_cue(recheck.get("humor_cue"))
                    if recheck_cue:
                        cue = recheck_cue
                        rejected = None
                value = {
                    "image": str(Path(str(row["image"])).resolve()),
                    "image_id": core.image_id(row),
                    "phase": row["_teacher_pipeline"]["phase"],
                    "description": description,
                    "description_genericized": genericized,
                    "humor_cue": cue,
                    "cue_rejected_reason": rejected or recheck_rejected,
                    "cue_source": "first_pass" if parsed.get("humor_cue") and cue else (
                        "recheck" if cue else "empty"
                    ),
                    "extractor_model": str(core.ANALYST_MODEL),
                    "raw_response": raw,
                    "raw_cue_recheck_response": recheck_raw,
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


pipeline.extract_guidance = extract_guidance_two_pass


if __name__ == "__main__":
    pipeline.run_all()
