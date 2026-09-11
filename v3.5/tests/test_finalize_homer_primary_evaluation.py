from __future__ import annotations

import pytest

from scripts.finalize_homer_primary_evaluation import compute


def _records() -> list[dict]:
    rows = []
    for system in ("latent_bridge", "text_homer_context_replay"):
        for image in ("i1", "i2"):
            for trial in (0, 1):
                for group in ("1-9", "200-209", "1000-1009"):
                    for reference_index in range(5):
                        rows.append(
                            {
                                "system_id": system,
                                "image_id": image,
                                "trial": trial,
                                "official_reference_group": group,
                                "reference_group": f"{group}/gt_caption{reference_index + 1}",
                                "candidate_count": 5,
                                "winning_caption_count": 5 if system == "latent_bridge" else 0,
                            }
                        )
    return rows


def test_compute_produces_paired_pass_at_k() -> None:
    result = compute(_records(), bootstrap_seed=7, bootstrap_replicates=20)
    assert len(result["summaries"]) == 2 * 3 * 3
    assert len(result["paired_comparison"]) == 3 * 3
    assert all(row["pass_at_k"] == 1.0 for row in result["summaries"] if row["system_id"] == "latent_bridge")
    assert all(row["mean_difference"] == 1.0 for row in result["paired_comparison"])


def test_compute_rejects_incomplete_image_grid() -> None:
    rows = _records()[:-1]
    with pytest.raises(ValueError, match="incomplete HOMER grid"):
        compute(rows, bootstrap_seed=7, bootstrap_replicates=20)
