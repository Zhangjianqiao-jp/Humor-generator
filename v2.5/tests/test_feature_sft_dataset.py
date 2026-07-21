from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from src.training.feature_sft_dataset import FeatureHumorSFTDataset
from src.training.sft_dataset import DEFAULT_SFT_PROMPT
from src.utils.io import write_jsonl


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)


def test_hic_compact_json_sft_prompt_uses_row_key_context_and_strips_gold(tmp_path: Path) -> None:
    image_path = tmp_path / "dog.jpg"
    _write_image(image_path)
    train_path = tmp_path / "train.jsonl"
    context_path = tmp_path / "context.jsonl"
    gold_caption = "the dog finally got his license"
    other_caption = "a dog forgot the parking brake"
    gold_key = f"dog::{hashlib.sha1(gold_caption.encode('utf-8')).hexdigest()[:12]}"
    other_key = f"dog::{hashlib.sha1(other_caption.encode('utf-8')).hexdigest()[:12]}"

    write_jsonl(
        train_path,
        [
            {
                "image_id": "dog",
                "image": str(image_path),
                "caption": gold_caption,
            }
        ],
    )
    write_jsonl(
        context_path,
        [
            {
                "row_key": gold_key,
                "image_id": "dog",
                "image": str(image_path),
                "gold_caption": gold_caption,
                "analysis": {
                    "literal_image_description": "A dog sits behind a steering wheel.",
                    "humor_type": "role_mismatch",
                    "humor_point": f"The gold caption says {gold_caption}, because the dog is framed as a driver.",
                    "visual_anchors": [
                        {
                            "label": "dog at steering wheel",
                            "role": f"sets up {gold_caption}",
                            "evidence": "dog is behind the wheel",
                        }
                    ],
                    "required_viewpoints": ["relation_crop"],
                    "primary_viewpoint": "relation_crop",
                    "needs_external_knowledge": False,
                },
            },
            {
                "row_key": other_key,
                "image_id": "dog",
                "image": str(image_path),
                "gold_caption": other_caption,
                "analysis": {
                    "literal_image_description": "A dog sits behind a steering wheel.",
                    "humor_type": "wrong_context",
                    "humor_point": f"The gold caption says {other_caption}.",
                    "visual_anchors": [],
                    "required_viewpoints": ["relation_crop"],
                    "primary_viewpoint": "relation_crop",
                    "needs_external_knowledge": False,
                },
            },
        ],
    )

    dataset = FeatureHumorSFTDataset(
        path=train_path,
        context_jsonl=context_path,
        feature_method="hic-compact-json",
        processor=None,
        normalize_prompt=True,
        sft_prompt=DEFAULT_SFT_PROMPT,
    )

    prompt = dataset[0]["prompt"]
    assert prompt.endswith(DEFAULT_SFT_PROMPT)
    assert "<joke_annotations>" in prompt
    assert "role_mismatch" in prompt
    assert "wrong_context" not in prompt
    assert gold_caption not in prompt
    assert other_caption not in prompt
    assert "the target joke" in prompt
    assert "Use the compact JSON as joke clues, not wording to copy." in prompt
