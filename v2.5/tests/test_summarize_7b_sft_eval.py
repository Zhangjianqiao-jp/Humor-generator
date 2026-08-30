from pathlib import Path

from scripts.summarize_7b_sft_eval import generation_metrics
from src.utils.io import write_jsonl


def test_generation_metrics_counts_and_template_rate(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    write_jsonl(
        path,
        [
            {"candidates": ["POV: a cat files taxes.", "The cat files taxes."]},
            {"candidates": ["Meanwhile, the dog audits."]},
        ],
    )
    metrics = generation_metrics(path)
    assert metrics["num_images"] == 2
    assert metrics["num_captions"] == 3
    assert metrics["generic_template_rate"] == 2 / 3
