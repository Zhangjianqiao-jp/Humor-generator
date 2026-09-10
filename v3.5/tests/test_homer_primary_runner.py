from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.run_homer_primary_openai import load_existing, load_packets, run


def _provenance(prompt_hash: str) -> dict:
    return {
        "evaluator": {
            "model_id": "gpt-5-chat-latest",
            "temperature": 0,
            "prompt_sha256": prompt_hash,
        }
    }


def _packet(prompt: str, packet_id: str = "p1") -> dict:
    return {
        "packet_id": packet_id,
        "system_prompt": prompt,
        "user_prompt": "Cartoon description: c\nCaption A : a\nCaption B : b",
        "response_contract": "exactly one token: A or B",
        "protocol": "homer_official_humorAI_pairwise",
    }


def test_runner_packet_validation_checks_prompt_hash_and_contract(tmp_path: Path) -> None:
    prompt = "official prompt"
    packets = tmp_path / "packets.jsonl"
    packets.write_text(json.dumps(_packet(prompt)) + "\n", encoding="utf-8")
    loaded = load_packets(packets, expected_system_prompt_hash=hashlib.sha256(prompt.encode()).hexdigest())
    assert [row["packet_id"] for row in loaded] == ["p1"]


def test_runner_requires_key_before_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = "official prompt"
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    packets = tmp_path / "packets.jsonl"
    provenance = tmp_path / "provenance.json"
    output = tmp_path / "judgments.jsonl"
    packets.write_text(json.dumps(_packet(prompt)) + "\n", encoding="utf-8")
    provenance.write_text(json.dumps(_provenance(prompt_hash)) + "\n", encoding="utf-8")
    monkeypatch.delenv("OPEN_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="no API request sent"):
        run(
            type(
                "Args",
                (),
                {
                    "provenance": provenance,
                    "packets": packets,
                    "output": output,
                    "max_requests": 1,
                    "max_workers": 1,
                    "delay_seconds": 0.0,
                    "max_retries": 1,
                    "retry_base_seconds": 0.0,
                },
            )()
        )
    assert not output.exists()


def test_existing_judgments_are_provenance_checked(tmp_path: Path) -> None:
    path = tmp_path / "judgments.jsonl"
    path.write_text(
        json.dumps({"packet_id": "p1", "decision": "A", "model": "wrong", "temperature": 0, "prompt_sha256": "a" * 64}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model does not match"):
        load_existing(path, model_id="gpt-5-chat-latest", temperature=0, prompt_hash="a" * 64)
