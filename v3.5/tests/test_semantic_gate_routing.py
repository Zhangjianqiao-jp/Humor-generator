from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _write_phase_a3_fixture(tmp_path: Path, *, update_norm: float) -> tuple[Path, Path]:
    config = tmp_path / "config.yaml"
    config.write_text(
        """loss:
  semantic_objective: channel_balanced_v3
training:
  seed: 7
gate:
  min_channel_validation_gap: 0.0
  min_channel_fraction_gap_gt_0: 0.60
  min_channel_retrieval_at_1: 0.125
  min_nll_improvement: 0.02
  max_relative_update_norm: 0.25
  equal_channel_weight_tolerance: 0.0001
"""
    )
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    validation = {
        "total": 1.0,
        "matched_minus_shuffled_logp": 0.0,
        "fraction_gap_gt_0": 0.5,
        "caption_nll": 1.0,
        "mean_gate": 1 / 3,
        "mean_relative_update_norm": update_norm,
        "info_nce": 0.5,
        "info_nce_retrieval_at_1": 0.2,
    }
    for channel in ("conflict", "local", "global"):
        validation.update(
            {
                f"matched_minus_shuffled_logp_{channel}": -0.01,
                f"fraction_gap_gt_0_{channel}": 0.5,
                f"caption_nll_{channel}": 1.0,
                f"info_nce_retrieval_at_1_{channel}": 0.2,
                f"mean_channel_weight_{channel}": 1 / 3,
            }
        )
    (metrics_dir / "metrics.jsonl").write_text(
        json.dumps({"epoch": 1, "validation": validation}) + "\n"
    )
    details = []
    for index in range(24):
        details.append(
            {
                "cluster_id": f"cluster-{index}",
                "matched_minus_shuffled_logp_conflict": -0.01,
                "matched_minus_shuffled_logp_local": -0.01,
                "matched_minus_shuffled_logp_global": -0.01,
            }
        )
    (metrics_dir / "validation_epoch_1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in details)
    )
    return config, metrics_dir / "metrics.jsonl"


def _run_gate(tmp_path: Path, *, update_norm: float) -> dict:
    config, metrics = _write_phase_a3_fixture(tmp_path, update_norm=update_norm)
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_semantic_training_gate.py"),
            "--config",
            str(config),
            "--metrics",
            str(metrics),
            "--output",
            str(report),
            "--report-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout
    return json.loads(report.read_text())


def test_phase_a3_point_fail_is_inconclusive_not_method_no_go(tmp_path: Path) -> None:
    report = _run_gate(tmp_path, update_norm=0.10)
    assert report["status"] == "pilot_inconclusive"
    assert report["engineering_gate_pass"] is True


def test_phase_a3_engineering_invariant_is_hard_stop(tmp_path: Path) -> None:
    report = _run_gate(tmp_path, update_norm=0.40)
    assert report["status"] == "hard_no_go"
    assert report["engineering_gate_pass"] is False
