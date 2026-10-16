from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.finalize_homer_auxiliary_models import parse_specs, validate_rows


def test_parse_specs_requires_unique_model_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate evaluator model"):
        parse_specs([f"m={tmp_path / 'a'}", f"m={tmp_path / 'b'}"])


def test_validate_rows_fails_closed_on_incomplete_packets(tmp_path: Path) -> None:
    path = tmp_path / "judge.jsonl"
    prompt_hash = hashlib.sha256(b"p").hexdigest()
    path.write_text(
        json.dumps({"packet_id": "only", "decision": "A", "model": "m", "temperature": 0, "prompt_sha256": prompt_hash, "source_provenance_sha256": "b" * 64}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="packet set mismatch"):
        validate_rows(
            path,
            model_id="m",
            expected_ids={"only", "missing"},
            prompt_hash=prompt_hash,
            source_provenance_sha256="b" * 64,
        )
