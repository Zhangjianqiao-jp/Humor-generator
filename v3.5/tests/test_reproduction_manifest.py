from __future__ import annotations

import json
from pathlib import Path

from humor_generator_v35.reproduction import audit_reproduction


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_preserves_released_count_discrepancy() -> None:
    manifest = json.loads((ROOT / "manifests/homer_official_assets.json").read_text())
    joke = manifest["joke_corpus"]
    assert joke["paper_declared_rows"] == 335_570
    assert joke["physical_lines"] == 335_570
    assert joke["csv_records"] == 335_569
    assert joke["physical_lines"] == joke["csv_records"] + 1


def test_gate_accepts_pinned_local_qwen_substitution() -> None:
    import yaml

    config = yaml.safe_load((ROOT / "configs/homer_text_reproduction.yaml").read_text())
    result = audit_reproduction(config, ROOT)
    assert result.ready
    assert not result.failures
    assert any("project substitution" in warning for warning in result.warnings)
    assert any("335,569 records" in warning for warning in result.warnings)
