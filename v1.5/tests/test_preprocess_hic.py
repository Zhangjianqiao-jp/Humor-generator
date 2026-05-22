from pathlib import Path

import pandas as pd

from src.data.preprocess_hic import PreprocessConfig, preprocess_hic_dataset
from src.training.sft_dataset import HumorSFTDataset
from src.utils.io import read_jsonl


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
    dataset = HumorSFTDataset(
        data_path,
        processor=None,
        max_seq_len=128,
        skip_missing_images=True,
        missing_image_report_path=report_path,
    )

    assert len(dataset) == 0
    assert read_jsonl(report_path)[0]["image"].endswith("missing.jpg")
