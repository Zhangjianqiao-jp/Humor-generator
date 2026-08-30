from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verify_sft_generations import verify_generation_rows


def test_planner_generation_requires_exact_schema_and_unique_images() -> None:
    report = verify_generation_rows(
        [
            {
                "image_id": "one",
                "prompt": "prompt",
                "candidates": ["ANCHOR: object\nCONTRAST: oddity\nANGLE: reversal"],
            },
            {
                "image_id": "two",
                "prompt": "prompt",
                "candidates": ["ANCHOR: action\nCONTRAST: mismatch\nANGLE: deadpan"],
            },
        ],
        "planner",
    )

    assert report["rows"] == report["unique_images"] == 2
    assert report["schema_valid"] is True


def test_caption_generation_rejects_prompt_leakage() -> None:
    with pytest.raises(ValueError, match="leaks the prompt"):
        verify_generation_rows(
            [{"image_id": "one", "prompt": "secret prompt", "candidates": ["secret prompt result"]}],
            "captioner",
        )


def test_planner_generation_rejects_field_placeholder_echo() -> None:
    with pytest.raises(ValueError, match="echoes a planner placeholder"):
        verify_generation_rows(
            [
                {
                    "image_id": "one",
                    "candidates": [
                        "ANCHOR: object\nCONTRAST: oddity\nANGLE: concise punchline direction"
                    ],
                }
            ],
            "planner",
        )


def test_caption_generation_rejects_duplicate_images() -> None:
    rows = [
        {"image_id": "one", "candidates": ["caption one"]},
        {"image_id": "one", "candidates": ["caption two"]},
    ]
    with pytest.raises(ValueError, match="Duplicate image_id"):
        verify_generation_rows(rows, "captioner")


def test_generation_verifier_runs_as_cli(tmp_path) -> None:
    input_path = tmp_path / "planner.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "image_id": "one",
                "prompt": "prompt",
                "candidates": ["ANCHOR: object\nCONTRAST: oddity\nANGLE: deadpan"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "verify_sft_generations.py"),
            "--input-jsonl",
            str(input_path),
            "--kind",
            "planner",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["schema_valid"] is True
