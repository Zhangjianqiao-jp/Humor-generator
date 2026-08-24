"""Multimodal preference data utilities for offline DPO.

The pair format keeps a single image/prompt context and two candidate captions.
Both candidates are encoded independently so that image tokens, chat-template
tokens, and answer masking exactly match the SFT data path.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from src.training.sft_dataset import resolve_image_path
from src.utils.io import read_jsonl


class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Path,
        image_root: Path | None = None,
        skip_missing_images: bool = False,
        max_samples: int | None = None,
    ) -> None:
        raw_rows = read_jsonl(path)
        self.rows: list[dict[str, Any]] = []
        skipped: list[str] = []
        for index, raw in enumerate(raw_rows):
            required = ("image", "image_id", "prompt", "chosen", "rejected")
            missing = [key for key in required if not str(raw.get(key) or "").strip()]
            if missing:
                raise ValueError(f"{path} row {index} is missing required keys: {missing}")
            image = resolve_image_path(str(raw["image"]), image_root)
            if not image.exists():
                if skip_missing_images:
                    skipped.append("missing_image")
                    continue
                raise FileNotFoundError(f"{path} row {index} references missing image: {image}")
            try:
                with Image.open(image) as handle:
                    handle.verify()
            except (OSError, UnidentifiedImageError) as exc:
                if skip_missing_images:
                    skipped.append("unreadable_image")
                    continue
                raise ValueError(f"{path} row {index} has unreadable image: {image}") from exc
            row = dict(raw)
            row["image"] = str(image)
            self.rows.append(row)
            if max_samples is not None and len(self.rows) >= max_samples:
                break
        if not self.rows:
            raise ValueError(f"{path} produced zero valid preference pairs.")
        print(f"[data] {path}: valid={len(self.rows)}/{len(raw_rows)}, skipped={len(skipped)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class ImageBalancedPreferenceDataset(torch.utils.data.Dataset):
    """One deterministic or random preference pair per image per epoch."""

    def __init__(self, dataset: PreferenceDataset, seed: int, randomize: bool) -> None:
        self.dataset = dataset
        self.randomize = randomize
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            grouped[str(row["image_id"])].append(index)
        self.groups = [grouped[key] for key in sorted(grouped)]
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        candidates = self.groups[index]
        if not self.randomize:
            return self.dataset[candidates[0]]
        chosen = int(torch.randint(len(candidates), (1,), generator=self.generator).item())
        return self.dataset[candidates[chosen]]


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed supervised-token log probabilities and token counts."""
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = torch.gather(
        torch.log_softmax(shifted_logits, dim=-1),
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    return (token_logps * mask).sum(dim=-1), mask.sum(dim=-1)


class DPOCollator:
    def __init__(
        self,
        processor: Any,
        max_seq_len: int,
        require_reference: bool = True,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
    ) -> None:
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.require_reference = require_reference
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels

    @staticmethod
    def _messages(row: dict[str, Any], answer: str | None = None) -> list[dict[str, Any]]:
        image_content: dict[str, Any] = {"type": "image", "image": row["image"]}
        if row.get("image_min_pixels") is not None:
            image_content["min_pixels"] = int(row["image_min_pixels"])
        if row.get("image_max_pixels") is not None:
            image_content["max_pixels"] = int(row["image_max_pixels"])
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": row["prompt"]},
                ],
            }
        ]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    def _template(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _vision_info(conversations: list[list[dict[str, Any]]]) -> tuple[Any, Any]:
        try:
            from qwen_vl_utils import process_vision_info

            return process_vision_info(conversations)
        except ImportError as exc:
            raise RuntimeError("qwen-vl-utils is required for multimodal DPO collation.") from exc

    def __call__(self, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        n_pairs = len(pairs)
        if n_pairs == 0:
            raise ValueError("DPO collator received an empty batch.")
        prepared_pairs = []
        for pair in pairs:
            prepared = dict(pair)
            prepared["image_min_pixels"] = self.image_min_pixels
            prepared["image_max_pixels"] = self.image_max_pixels
            prepared_pairs.append(prepared)
        rows = [(pair, pair["chosen"]) for pair in prepared_pairs] + [(pair, pair["rejected"]) for pair in prepared_pairs]
        full_messages = [self._messages(pair, answer) for pair, answer in rows]
        prompt_messages = [self._messages(pair) for pair, _ in rows]
        texts = [self._template(messages, add_generation_prompt=False) for messages in full_messages]
        prompt_texts = [self._template(messages, add_generation_prompt=True) for messages in prompt_messages]
        image_inputs, video_inputs = self._vision_info(full_messages)
        prompt_image_inputs, prompt_video_inputs = self._vision_info(prompt_messages)
        encoded = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        prompt_encoded = self.processor(
            text=prompt_texts,
            images=prompt_image_inputs,
            videos=prompt_video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        seq_len = labels.shape[1]
        for i in range(labels.shape[0]):
            prompt_len = min(int(prompt_encoded["attention_mask"][i].sum().item()), seq_len)
            labels[i, :prompt_len] = -100
            labels[i, encoded["attention_mask"][i] == 0] = -100
            if int((labels[i] != -100).sum().item()) == 0:
                raise ValueError(f"DPO target was fully truncated for image={rows[i][0]['image']}")
        encoded["labels"] = labels
        if self.require_reference:
            reference = []
            for pair in pairs:
                values = pair.get("reference_logps")
                if not isinstance(values, dict) or "chosen" not in values or "rejected" not in values:
                    raise ValueError(
                        "DPO training requires frozen reference_logps. Run "
                        "scripts/precompute_dpo_reference_logps.py first."
                    )
                reference.append((float(values["chosen"]), float(values["rejected"])))
            encoded["reference_chosen_logps"] = torch.tensor([x[0] for x in reference], dtype=torch.float32)
            encoded["reference_rejected_logps"] = torch.tensor([x[1] for x in reference], dtype=torch.float32)
        encoded["num_pairs"] = n_pairs
        return encoded


def model_inputs_from_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    excluded = {"labels", "reference_chosen_logps", "reference_rejected_logps", "num_pairs"}
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
        if key not in excluded
    }
