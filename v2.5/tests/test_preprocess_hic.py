from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data.preprocess_hic import PreprocessConfig, preprocess_hic_dataset
from src.utils.io import read_jsonl
from src.training.sft_dataset import HumorSFTDataset, clean_generated_caption


def test_generated_text_cleaning_can_preserve_planner_schema() -> None:
    plan = "ANCHOR: visible object\nCONTRAST: violated expectation\nANGLE: dry escalation"
    assert clean_generated_caption(plan) == "ANCHOR: visible object"
    assert clean_generated_caption(plan, preserve_newlines=True) == plan

def test_preprocess_filters_deduplicates_and_splits(tmp_path: Path) -> None:
    image_csv = tmp_path / "images.csv"
    caption_csv = tmp_path / "captions.csv"
    output_dir = tmp_path / "processed"

    pd.DataFrame(
        [
            {"image_id": "1", "image_path": "a.jpg"},
            {"image_id": "2", "image_path": "b.jpg"},
            {"image_id": "3", "image_path": "c.jpg"},
        ]
    ).to_csv(image_csv, index=False)
    pd.DataFrame(
        [
            {"image_id": "1", "caption": "a funny caption", "score": 0.9},
            {"image_id": "1", "caption": "A funny caption", "score": 0.8},
            {"image_id": "2", "caption": "", "score": 0.9},
            {"image_id": "3", "caption": "another funny caption", "score": 0.7},
        ]
    ).to_csv(caption_csv, index=False)

    counts = preprocess_hic_dataset(
        PreprocessConfig(
            image_csv=image_csv,
            caption_csv=caption_csv,
            image_base_dir=None,
            output_dir=output_dir,
            image_id_col="image_id",
            image_path_col="image_path",
            image_url_col=None,
            caption_image_id_col="image_id",
            caption_col="caption",
            score_col="score",
            rank_percentile_threshold=0.0,
            train_ratio=0.34,
            val_ratio=0.33,
            test_ratio=0.33,
            seed=7,
        )
    )

    rows = (
        read_jsonl(output_dir / "sft_train.jsonl")
        + read_jsonl(output_dir / "sft_val.jsonl")
        + read_jsonl(output_dir / "sft_test.jsonl")
    )
    assert counts == {"train": 0, "val": 0, "test": 2}
    assert len(rows) == 2
    assert {row["image_id"] for row in rows} == {"1", "3"}
    assert all(row["meta"]["version"] == "v1.5" for row in rows)


def test_sft_dataset_filters_missing_images(tmp_path: Path) -> None:
    data_path = tmp_path / "sft.jsonl"
    report_path = tmp_path / "missing.jsonl"
    rows = [
        {
            "image": str(tmp_path / "missing.jpg"),
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "prompt"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "caption"}]},
            ],
        }
    ]
    from src.utils.io import write_jsonl

    write_jsonl(data_path, rows)
    with pytest.raises(ValueError, match="produced zero valid SFT samples"):
        HumorSFTDataset(
            data_path,
            processor=None,
            max_seq_len=128,
            skip_missing_images=True,
            missing_image_report_path=report_path,
        )

    assert read_jsonl(report_path)[0]["image"].endswith("missing.jpg")

class _FakeTokenizer:
    pad_token_id = 0


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if add_generation_prompt:
            return "USER IMAGE PROMPT ASSISTANT:"
        return "USER IMAGE PROMPT ASSISTANT: funny caption"

    def __call__(self, text, images=None, videos=None, padding=True, truncation=True, max_length=None, return_tensors=None):
        import torch

        if "funny caption" not in text[0]:
            return {
                "input_ids": torch.tensor([[11, 12, 13]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }
        return {
            "input_ids": torch.tensor([[11, 12, 13, 21, 22]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        }


def test_sft_dataset_masks_prompt_tokens(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(image_path)
    data_path = tmp_path / "sft.jsonl"
    from src.utils.io import write_jsonl

    write_jsonl(
        data_path,
        [
            {
                "image": str(image_path),
                "messages": [
                    {"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": "prompt"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "funny caption"}]},
                ],
            }
        ],
    )

    dataset = HumorSFTDataset(data_path, processor=_FakeProcessor(), max_seq_len=32)
    batch = dataset.collate_fn([dataset[0]])

    assert batch["labels"][0, :3].tolist() == [-100, -100, -100]
    assert batch["labels"][0, 3:].tolist() == [21, 22]
