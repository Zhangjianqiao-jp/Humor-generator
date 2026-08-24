from __future__ import annotations

import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from src.utils.io import read_jsonl, write_jsonl


DEFAULT_SFT_PROMPT = (
    "Generate one short, natural, image-specific humorous caption for this image. "
    "Do not explain."
)

BAD_TEXT_VALUES = {"", "nan", "null", "none"}


def _first_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text")
        return None
    if isinstance(content, list):
        for item in content:
            text = _first_text(item)
            if text is not None:
                return text
    return None


def _first_image(content: Any) -> str | None:
    if isinstance(content, dict):
        if content.get("type") == "image":
            return content.get("image")
        return None
    if isinstance(content, list):
        for item in content:
            image = _first_image(item)
            if image is not None:
                return image
    return None


def _is_invalid_text(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in BAD_TEXT_VALUES


def extract_original_prompt(row: dict[str, Any]) -> str | None:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return _first_text(message.get("content"))
    return row.get("prompt")


def extract_caption(row: dict[str, Any]) -> str | None:
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            return _first_text(message.get("content"))
    for key in ("caption", "gold_caption", "text", "answer"):
        if key in row:
            return row[key]
    return None


def extract_image_path(row: dict[str, Any]) -> str | None:
    if row.get("image"):
        return row["image"]
    for message in row.get("messages", []):
        image = _first_image(message.get("content"))
        if image is not None:
            return image
    return row.get("image_path")


def resolve_image_path(image: str, image_root: Path | None) -> Path:
    path = Path(image).expanduser()
    if path.is_absolute() or image_root is None:
        return path
    return image_root / path


def clean_generated_caption(
    text: str,
    prompt: str | None = None,
    preserve_newlines: bool = False,
) -> str:
    text = text.strip()
    if prompt and prompt in text:
        text = text.split(prompt, maxsplit=1)[-1].strip()
    for prefix in ("Caption:", "caption:", "CAPTION:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    text = "\n".join(lines) if preserve_newlines else (lines[0] if lines else "")
    for prefix in ("- ", "* "):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


class HumorSFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Path,
        processor: Any | None = None,
        max_seq_len: int = 512,
        image_root: Path | None = None,
        max_caption_chars: int = 240,
        skip_missing_images: bool = False,
        normalize_prompt: bool = True,
        sft_prompt: str = DEFAULT_SFT_PROMPT,
        min_supervised_tokens: int = 3,
        missing_image_report_path: Path | None = None,
        max_samples: int | None = None,
        validate_images: bool = True,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
    ) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self.original_count = 0
        self.skipped_count = 0
        self.skip_reasons: Counter[str] = Counter()
        self.missing_image_count = 0
        self.unreadable_image_count = 0
        self.truncated_sample_count = 0
        self.low_supervision_warning_count = 0
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.image_root = image_root
        self.max_caption_chars = max_caption_chars
        self.skip_missing_images = skip_missing_images
        self.normalize_prompt = normalize_prompt
        self.sft_prompt = sft_prompt.strip()
        self.min_supervised_tokens = min_supervised_tokens
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels

        raw_rows = read_jsonl(path)
        self.original_count = len(raw_rows)
        missing_rows: list[dict[str, Any]] = []

        for index, raw_row in enumerate(raw_rows):
            row, reason = self._validate_row(raw_row, index, validate_images)
            if row is not None:
                self.rows.append(row)
                if max_samples is not None and len(self.rows) >= max_samples:
                    break
                continue

            self.skipped_count += 1
            self.skip_reasons[reason] += 1
            if reason == "missing_image":
                missing_rows.append(raw_row)

        self.missing_image_count = self.skip_reasons["missing_image"]
        self.unreadable_image_count = self.skip_reasons["unreadable_image"]
        if missing_image_report_path and missing_rows:
            write_jsonl(missing_image_report_path, missing_rows)

        self._print_validation_summary(max_samples=max_samples)
        if len(self.rows) == 0:
            raise ValueError(f"{path} produced zero valid SFT samples.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def _validate_row(
        self,
        raw_row: dict[str, Any],
        index: int,
        validate_images: bool,
    ) -> tuple[dict[str, Any] | None, str]:
        raw_image = extract_image_path(raw_row)
        if _is_invalid_text(raw_image):
            return None, "missing_image_field"

        image_path = resolve_image_path(str(raw_image), self.image_root)
        if not image_path.exists():
            if not self.skip_missing_images:
                raise FileNotFoundError(
                    f"{self.path} row {index} references a missing image: {image_path}. "
                    "Set data.skip_missing_images: true to skip missing images."
                )
            return None, "missing_image"

        if validate_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                if not self.skip_missing_images:
                    raise ValueError(f"{self.path} row {index} has an unreadable image: {image_path}") from exc
                return None, "unreadable_image"

        caption = extract_caption(raw_row)
        if _is_invalid_text(caption):
            return None, "empty_caption"
        caption = str(caption).strip()
        if len(caption) > self.max_caption_chars:
            return None, "caption_too_long"

        original_prompt = extract_original_prompt(raw_row)
        if original_prompt is None:
            original_prompt = ""

        return (
            {
                "image": str(image_path),
                "raw_image": str(raw_image),
                "image_id": raw_row.get("image_id") or image_path.stem,
                "original_prompt": str(original_prompt).strip(),
                "prompt": self.prompt_for_row({"original_prompt": original_prompt}),
                "caption": caption,
                "meta": raw_row.get("meta", {}),
                "source_index": index,
            },
            "",
        )

    def _print_validation_summary(self, max_samples: int | None = None) -> None:
        suffix = f" (capped at {max_samples})" if max_samples is not None else ""
        print(
            f"[data] {self.path}: valid={len(self.rows)}/{self.original_count}{suffix}, "
            f"skipped={self.skipped_count}"
        )
        if self.skip_reasons:
            reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(self.skip_reasons.items()))
            print(f"[data] skipped reasons: {reasons}")

    def prompt_for_row(self, row: dict[str, Any]) -> str:
        if self.normalize_prompt:
            return self.sft_prompt
        prompt = str(row.get("original_prompt") or "").strip()
        return prompt or self.sft_prompt

    def build_user_message(self, row: dict[str, Any]) -> dict[str, Any]:
        image_content: dict[str, Any] = {"type": "image", "image": row["image"]}
        # qwen-vl-utils resizes the image before the processor expands the
        # image placeholder.  Keeping this budget on the message is essential:
        # truncating expanded image tokens later makes Qwen token alignment fail.
        if self.image_min_pixels is not None:
            image_content["min_pixels"] = int(self.image_min_pixels)
        if self.image_max_pixels is not None:
            image_content["max_pixels"] = int(self.image_max_pixels)
        return {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": self.prompt_for_row(row)},
            ],
        }

    def build_full_messages(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            self.build_user_message(row),
            {"role": "assistant", "content": [{"type": "text", "text": row["caption"]}]},
        ]

    def build_prompt_messages(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        return [self.build_user_message(row)]

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        if self.processor is None:
            raise RuntimeError("HumorSFTDataset.collate_fn requires a processor.")
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        prompt = messages[0]["content"][-1]["text"]
        if add_generation_prompt:
            return f"User: <image>\n{prompt}\nAssistant:"
        answer = messages[1]["content"][0]["text"]
        return f"User: <image>\n{prompt}\nAssistant: {answer}"

    def _process_vision_info(self, conversations: list[list[dict[str, Any]]]) -> tuple[Any, Any]:
        try:
            from qwen_vl_utils import process_vision_info

            return process_vision_info(conversations)
        except ImportError:
            images = [Image.open(row["image"]).convert("RGB") for row in conversations_to_rows(conversations)]
            return images, None

    def collate_fn(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if self.processor is None:
            raise RuntimeError("HumorSFTDataset.collate_fn requires a processor.")

        full_messages = [self.build_full_messages(row) for row in examples]
        prompt_messages = [self.build_prompt_messages(row) for row in examples]
        texts = [
            self._apply_chat_template(messages, add_generation_prompt=False)
            for messages in full_messages
        ]
        prompt_texts = [
            self._apply_chat_template(messages, add_generation_prompt=True)
            for messages in prompt_messages
        ]
        image_inputs, video_inputs = self._process_vision_info(full_messages)
        prompt_image_inputs, prompt_video_inputs = self._process_vision_info(prompt_messages)

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
        supervised_counts: list[int] = []
        prompt_lengths: list[int] = []
        truncated_flags: list[bool] = []
        attention_mask = encoded["attention_mask"]
        seq_len = labels.shape[1]

        for i, prompt_attention_mask in enumerate(prompt_encoded["attention_mask"]):
            prompt_len = min(int(prompt_attention_mask.sum().item()), seq_len)
            prompt_lengths.append(prompt_len)
            labels[i, :prompt_len] = -100
            labels[i, attention_mask[i] == 0] = -100

            nonpad_len = int(attention_mask[i].sum().item())
            is_truncated = nonpad_len >= self.max_seq_len
            truncated_flags.append(is_truncated)
            if is_truncated:
                self.truncated_sample_count += 1

            supervised_count = int((labels[i] != -100).sum().item())
            supervised_counts.append(supervised_count)
            if supervised_count == 0:
                raise ValueError(
                    "Supervised target was fully truncated or masked. "
                    f"image={examples[i]['image']} max_seq_len={self.max_seq_len} "
                    f"prompt_len={prompt_len} full_len={nonpad_len}"
                )
            if supervised_count < self.min_supervised_tokens:
                self.low_supervision_warning_count += 1
                if self.low_supervision_warning_count <= 5:
                    warnings.warn(
                        "Supervised target has very few tokens: "
                        f"tokens={supervised_count}, image={examples[i]['image']}, "
                        f"caption={examples[i]['caption']!r}",
                        RuntimeWarning,
                    )

        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[encoded["input_ids"] == pad_token_id] = -100

        encoded["labels"] = labels
        encoded["supervised_token_counts"] = torch.tensor(supervised_counts, dtype=torch.long)
        encoded["prompt_lengths"] = torch.tensor(prompt_lengths, dtype=torch.long)
        encoded["truncated_flags"] = torch.tensor(truncated_flags, dtype=torch.bool)
        encoded["metadata"] = examples
        return encoded

    def print_debug_samples(self, n: int = 3) -> None:
        for idx, row in enumerate(self.rows[:n]):
            print("=" * 80)
            print(f"[sample {idx}] image: {row['image']}")
            print(f"[sample {idx}] image_id: {row.get('image_id')}")
            print(f"[sample {idx}] score: {row.get('meta', {}).get('score')}")
            print(f"[sample {idx}] original prompt: {row.get('original_prompt', '')}")
            print(f"[sample {idx}] normalized prompt used: {self.prompt_for_row(row)}")
            print(f"[sample {idx}] assistant caption: {row['caption']}")

    def print_debug_batch(
        self,
        batch: dict[str, Any],
        examples: list[dict[str, Any]],
        n: int = 3,
    ) -> None:
        tokenizer = self.processor.tokenizer
        limit = min(n, len(examples))
        for i in range(limit):
            labels = batch["labels"][i]
            input_ids = batch["input_ids"][i]
            attention_mask = batch["attention_mask"][i]
            supervised_mask = labels != -100
            supervised_ids = input_ids[supervised_mask]
            full_ids = input_ids[attention_mask.bool()]
            decoded_full = tokenizer.decode(
                full_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            decoded_target = tokenizer.decode(
                supervised_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            print("=" * 80)
            print(f"[collator {i}] image: {examples[i]['image']}")
            print(f"[collator {i}] original prompt: {examples[i].get('original_prompt', '')}")
            print(f"[collator {i}] normalized prompt used: {self.prompt_for_row(examples[i])}")
            print(f"[collator {i}] raw caption: {examples[i]['caption']}")
            print(f"[collator {i}] num_supervised_tokens: {int(supervised_mask.sum().item())}")
            print(f"[collator {i}] prompt_len: {int(batch['prompt_lengths'][i].item())}")
            print(f"[collator {i}] truncated: {bool(batch['truncated_flags'][i].item())}")
            print(f"[collator {i}] decoded full input:\n{decoded_full}")
            print(f"[collator {i}] decoded supervised target only:\n{decoded_target}")


def conversations_to_rows(conversations: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for messages in conversations:
        image = extract_image_path({"messages": messages})
        rows.append({"image": image})
    return rows
