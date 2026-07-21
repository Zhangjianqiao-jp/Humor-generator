from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_region_ablation_runner_passes_limit_to_generation() -> None:
    script = (ROOT / "scripts" / "run_hic_region_annotation_ablation.sh").read_text(encoding="utf-8")
    generation_block = script.split('"$PY" scripts/generate_guided_lora_candidates.py', 1)[1].split(
        "done",
        1,
    )[0]

    assert '--limit "$LIMIT"' in generation_block

