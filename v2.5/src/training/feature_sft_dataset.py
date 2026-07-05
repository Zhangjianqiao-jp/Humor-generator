from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.analysis.guided_prompting import build_guided_prompt
from src.training.sft_dataset import HumorSFTDataset, extract_image_path


def _keys_for_context(row: dict[str, Any]) -> list[str]:
    keys = []
    image_id = row.get("image_id")
    if image_id:
        keys.append(f"id:{image_id}")
    image = row.get("image") or row.get("raw_image") or extract_image_path(row)
    if image:
        keys.append(f"image:{image}")
        keys.append(f"stem:{Path(str(image)).stem}")
    return keys


def load_context_map(path: Path) -> dict[str, dict[str, Any]]:
    context_map: dict[str, dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Feature context JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("failed"):
                continue
            for key in _keys_for_context(row):
                context_map[key] = row
    if not context_map:
        raise ValueError(f"Feature context JSONL produced zero usable rows: {path}")
    return context_map


class FeatureHumorSFTDataset(HumorSFTDataset):
    """SFT dataset that injects VLM visual facts or structured humor guidance."""

    def __init__(
        self,
        *args: Any,
        context_jsonl: Path,
        feature_method: str,
        require_context: bool = True,
        **kwargs: Any,
    ) -> None:
        self.context_jsonl = context_jsonl
        self.context_by_key = load_context_map(context_jsonl)
        self.feature_method = feature_method
        self.require_context = require_context
        super().__init__(*args, **kwargs)

    def _find_context(self, row: dict[str, Any]) -> dict[str, Any] | None:
        for key in _keys_for_context(row):
            context = self.context_by_key.get(key)
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

        context = self._find_context(row)
        if context is None:
            if self.require_context:
                return None, "missing_context"
            context = {}

        row["image_description"] = context.get("image_description", "")
        row["visual_facts"] = context.get("visual_facts") or context.get("humor_points", {})
        row["structured_humor"] = context.get("structured_humor", {})
        row["context_source"] = {
            "context_jsonl": str(self.context_jsonl),
            "extractor_model": context.get("extractor_model"),
        }
        row["prompt"] = self.prompt_for_row(row)
        return row, ""

    def prompt_for_row(self, row: dict[str, Any]) -> str:
        if self.normalize_prompt:
            base_prompt = self.sft_prompt
        else:
            base_prompt = str(row.get("original_prompt") or "").strip() or self.sft_prompt
        return build_guided_prompt(
            method=self.feature_method,
            image_description=str(row.get("image_description") or ""),
            visual_facts=row.get("visual_facts") or {},
            structured_humor=row.get("structured_humor") or {},
            base_prompt=base_prompt,
        )

