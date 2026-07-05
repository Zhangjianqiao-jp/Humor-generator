from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.training.sft_dataset import DEFAULT_SFT_PROMPT, HumorSFTDataset, extract_image_path


def context_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    image_id = row.get("image_id")
    if image_id:
        keys.append(f"id:{image_id}")
    image = row.get("image") or extract_image_path(row)
    if image:
        keys.extend((f"image:{image}", f"stem:{Path(str(image)).stem}"))
    return keys


def load_guidance(path: Path) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"Guidance file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("failed") or not str(row.get("description") or "").strip():
                continue
            for key in context_keys(row):
                contexts[key] = row
    if not contexts:
        raise ValueError(f"No usable guidance rows in {path}")
    return contexts


def build_training_prompt(
    description: str,
    humor_cue: str,
    base_prompt: str = DEFAULT_SFT_PROMPT,
) -> str:
    """Keep the original caption request verbatim at the end of the prompt."""
    return (
        f"Image description: {' '.join(description.split())}\n"
        f"Humor cue: {' '.join(humor_cue.split())}\n\n"
        f"{base_prompt}"
    )


class GuidedHumorSFTDataset(HumorSFTDataset):
    """Image SFT with independently extracted description and one visible cue."""

    def __init__(
        self,
        *args: Any,
        guidance_jsonl: Path,
        require_guidance: bool = True,
        **kwargs: Any,
    ) -> None:
        self.guidance_jsonl = guidance_jsonl
        self.guidance = load_guidance(guidance_jsonl)
        self.require_guidance = require_guidance
        super().__init__(*args, **kwargs)

    def _find_guidance(self, row: dict[str, Any]) -> dict[str, Any] | None:
        for key in context_keys(row):
            context = self.guidance.get(key)
            if context is not None:
                return context
        return None

    def _validate_row(
        self,
        raw_row: dict[str, Any],
        index: int,
        validate_images: bool,
    ) -> tuple[dict[str, Any] | None, str]:
        row, reason = super()._validate_row(raw_row, index, validate_images)
        if row is None:
            return None, reason
        guidance = self._find_guidance(row)
        if guidance is None:
            if self.require_guidance:
                return None, "missing_guidance"
            guidance = {}
        row["description"] = " ".join(str(guidance.get("description") or "").split())
        row["humor_cue"] = " ".join(str(guidance.get("humor_cue") or "").split())
        row["guidance_source"] = {
            "path": str(self.guidance_jsonl),
            "model": guidance.get("extractor_model"),
        }
        row["prompt"] = self.prompt_for_row(row)
        return row, ""

    def prompt_for_row(self, row: dict[str, Any]) -> str:
        base_prompt = self.sft_prompt
        if not self.normalize_prompt:
            base_prompt = str(row.get("original_prompt") or "").strip() or self.sft_prompt
        return build_training_prompt(
            description=str(row.get("description") or ""),
            humor_cue=str(row.get("humor_cue") or ""),
            base_prompt=base_prompt,
        )
