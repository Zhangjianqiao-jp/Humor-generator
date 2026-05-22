from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.utils.io import read_jsonl


class HumorSFTDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, processor: Any, max_seq_len: int) -> None:
        self.rows = read_jsonl(path)
        self.processor = processor
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def _format_text(self, row: dict[str, Any]) -> str:
        messages = row["messages"]
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        prompt = messages[0]["content"][-1]["text"]
        answer = messages[1]["content"][0]["text"]
        return f"User: <image>\n{prompt}\nAssistant: {answer}"

    def collate_fn(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [self._format_text(row) for row in examples]
        images = [Image.open(row["image"]).convert("RGB") for row in examples]
        encoded = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoded["labels"] = labels
        return encoded
