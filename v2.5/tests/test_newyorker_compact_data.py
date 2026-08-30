from __future__ import annotations

from pathlib import Path
import re

from scripts.audit_newyorker_compact_sft import audit
from scripts.build_newyorker_compact_sft import compact_text, descriptions
from scripts.preflight_cpu_lora_sft_step import make_dataset
from src.training.sft_dataset import HumorSFTDataset, extract_caption, extract_original_prompt
from src.utils.io import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
COMPACT_DIR = ROOT / "data/processed/newyorker_compact_sft_v2"
DPO_DIR = ROOT / "data/processed/newyorker_compact_dpo"


def test_independent_raw_source_audit_passes() -> None:
    report = audit(
        ROOT / "data/raw/newyorker_caption_ranking",
        ROOT / "data/processed/newyorker_top3pct_sft",
        COMPACT_DIR,
    )
    assert report["status"] == "pass"
    assert report["selection"] == "exact per-cartoon top 3% reconstructed from raw ranking CSV"
    assert report["split_image_disjoint"] is True
    assert report["gold_caption_prompt_leakage"] == 0
    assert report["verified_unique_images"] == 127


def test_compact_text_keeps_a_complete_first_sentence() -> None:
    text = "A king is sitting on his throne while a servant reads a scroll. A second sentence."
    assert compact_text(text) == "A king is sitting on his throne while a servant reads a scroll."


def test_cpu_preflight_preserves_configured_original_prompt() -> None:
    config = {
        "data": {
            "train_path": str(COMPACT_DIR / "caption_train.jsonl"),
            "image_root": None,
            "max_seq_len": 768,
            "max_caption_chars": 220,
            "min_supervised_tokens": 3,
            "image_min_pixels": None,
            "image_max_pixels": 200704,
            "normalize_prompt": False,
        }
    }
    dataset = make_dataset(config, processor=None)
    assert "Humor plan:" in dataset[0]["prompt"]
    assert dataset[0]["prompt"] == dataset[0]["original_prompt"]


def test_compact_plans_are_schema_valid_and_caption_free() -> None:
    description_by_contest = descriptions(ROOT / "data/raw/newyorker_caption_ranking")
    expected_images = {"train": 79, "validation": 24, "test": 24}
    for split, expected_count in expected_images.items():
        planner = read_jsonl(COMPACT_DIR / f"planner_{split}.jsonl")
        assert len(planner) == expected_count
        plans_by_image: dict[str, str] = {}
        for row in planner:
            plan = extract_caption(row)
            assert plan is not None
            lines = plan.splitlines()
            assert len(lines) == 3
            assert [line.split(":", 1)[0] for line in lines] == ["ANCHOR", "CONTRAST", "ANGLE"]
            assert row["meta"]["label_source"] == "release_gpt4o_description_only"
            source = description_by_contest[int(row["meta"]["contest_number"])]
            assert lines[0] == f"ANCHOR: {compact_text(str(source['canny']))}"
            assert lines[1] == f"CONTRAST: {compact_text(str(source['uncanny']))}"
            plans_by_image[str(row["image_id"])] = plan
            for line in lines[:2]:
                value = line.split(":", 1)[1].strip()
                assert not re.search(
                    r"\b(a|an|the|of|in|on|with|while|and|to|for|from|at|as|by|that|which|is|are)$",
                    value,
                    re.IGNORECASE,
                )

        for row in read_jsonl(COMPACT_DIR / f"caption_{split}.jsonl"):
            prompt = extract_original_prompt(row)
            caption = extract_caption(row)
            assert prompt is not None and caption is not None
            # The full reference caption must never be exposed in the compact plan.
            assert caption.strip().casefold() not in prompt.casefold()
            assert row["meta"]["compact_label_source"] == "release_gpt4o_description_only"
            assert prompt.endswith(plans_by_image[str(row["image_id"])])


def test_splits_are_image_disjoint_and_dpo_pairs_are_valid() -> None:
    split_images: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        caption_rows = read_jsonl(COMPACT_DIR / f"caption_{split}.jsonl")
        dpo_rows = read_jsonl(DPO_DIR / f"dpo_{split}.jsonl")
        split_images[split] = {str(row["image_id"]) for row in caption_rows}
        assert {str(row["image_id"]) for row in dpo_rows} == split_images[split]
        assert all(int(row["meta"]["rank_gap"]) > 0 for row in dpo_rows)
        assert all(row["chosen"].casefold() != row["rejected"].casefold() for row in dpo_rows)
    assert not (split_images["train"] & split_images["validation"])
    assert not (split_images["train"] & split_images["test"])
    assert not (split_images["validation"] & split_images["test"])


def test_image_budget_is_kept_in_qwen_vision_message() -> None:
    dataset = HumorSFTDataset(
        COMPACT_DIR / "caption_train.jsonl",
        processor=None,
        validate_images=False,
        image_max_pixels=200704,
    )
    user_message = dataset.build_user_message(dataset[0])
    image_content = user_message["content"][0]
    assert image_content["type"] == "image"
    assert image_content["max_pixels"] == 200704
