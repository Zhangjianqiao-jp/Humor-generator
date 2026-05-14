from pathlib import Path

import pandas as pd

from src.data.build_sft_dataset import build_sft_dataset
from src.utils.io import read_jsonl


def test_build_sft_from_fake_csv(tmp_path: Path) -> None:
    image_csv = tmp_path / "images.csv"
    caption_csv = tmp_path / "captions.csv"
    out_dir = tmp_path / "processed"

    pd.DataFrame([
        {"image_id": "1", "image_path": "a.jpg"},
        {"image_id": "2", "image_path": "b.jpg"},
    ]).to_csv(image_csv, index=False)
    pd.DataFrame([
        {"image_id": "1", "caption": "a funny cat", "score": 0.9},
        {"image_id": "1", "caption": "", "score": 0.2},
        {"image_id": "2", "caption": "a funny dog", "score": 0.8},
    ]).to_csv(caption_csv, index=False)

    build_sft_dataset(image_csv, caption_csv, out_dir, "image_id", "image_path", None, "image_id", "caption", "score", threshold=0.0)
    combined = read_jsonl(out_dir / "sft_train.jsonl") + read_jsonl(out_dir / "sft_val.jsonl") + read_jsonl(out_dir / "sft_test.jsonl")
    assert len(combined) == 2
    assert all("messages" in r for r in combined)
