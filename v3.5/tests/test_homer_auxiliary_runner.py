from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_homer_auxiliary_openai import load_existing_auxiliary, run
from scripts.run_homer_account_agent_batches import run_batch_with_recovery, write_checkpoint


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, str]:
    prompt = "official prompt"
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    packets = tmp_path / "packets.jsonl"
    packets.write_text(
        json.dumps(
            {
                "packet_id": "p1",
                "system_prompt": prompt,
                "user_prompt": "Cartoon description: c\nCaption A : a\nCaption B : b",
                "response_contract": "exactly one token: A or B",
                "protocol": "homer_official_humorAI_pairwise",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps({"evaluator": {"model_id": "gpt-5-chat-latest", "temperature": 0, "prompt_sha256": prompt_hash}}),
        encoding="utf-8",
    )
    return packets, provenance, prompt_hash


def test_auxiliary_zero_request_is_local_smoke(tmp_path: Path) -> None:
    packets, provenance, _ = _write_inputs(tmp_path)
    output = tmp_path / "judgments.jsonl"
    args = type(
        "Args",
        (),
        {
            "packets": packets,
            "provenance": provenance,
            "output": output,
            "model_id": "gpt-5.5",
            "max_requests": 0,
            "max_workers": 1,
            "delay_seconds": 0.0,
            "max_retries": 1,
            "retry_base_seconds": 0.0,
        },
    )()
    result = run(args)
    assert result["status"] == "partial"
    assert result["judgments"] == 0
    assert not output.exists()


def test_auxiliary_requires_key_only_when_request_is_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packets, provenance, _ = _write_inputs(tmp_path)
    output = tmp_path / "judgments.jsonl"
    monkeypatch.delenv("OPEN_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = type(
        "Args",
        (),
        {
            "packets": packets,
            "provenance": provenance,
            "output": output,
            "model_id": "gpt-5.5",
            "max_requests": 1,
            "max_workers": 1,
            "delay_seconds": 0.0,
            "max_retries": 1,
            "retry_base_seconds": 0.0,
        },
    )()
    with pytest.raises(RuntimeError, match="no API request sent"):
        run(args)
    assert not output.exists()


def test_auxiliary_existing_rows_are_model_and_source_checked(tmp_path: Path) -> None:
    path = tmp_path / "judgments.jsonl"
    path.write_text(
        json.dumps(
            {
                "packet_id": "p1",
                "decision": "A",
                "model": "wrong",
                "temperature": 0,
                "prompt_sha256": "a" * 64,
                "source_provenance_sha256": "b" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model does not match"):
        load_existing_auxiliary(
            path,
            model_id="gpt-5.5",
            temperature=0,
            prompt_hash="a" * 64,
            source_provenance_sha256="b" * 64,
        )


def test_account_runner_splits_repeatedly_truncated_batch_and_reuses_children(tmp_path: Path) -> None:
    packets = [
        {"packet_id": f"p{i}", "user_prompt": f"prompt-{i}", "system_prompt": "rubric"}
        for i in range(4)
    ]
    state = tmp_path / "state"
    state.mkdir()
    args = type(
        "Args",
        (),
        {
            "codex_bin": "codex",
            "model": "gpt-5.6-terra",
            "call_timeout_seconds": 1.0,
        },
    )()
    schema = state / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    parent = state / "shard_000000_000004.json"
    for index in range(3):
        parent.with_name(parent.name + f".invalid{index}").write_text("truncated", encoding="utf-8")
    write_checkpoint(
        state / "shard_000000_000002.json",
        [{"packet_id": f"p{i}", "decision": "A"} for i in range(2)],
    )
    write_checkpoint(
        state / "shard_000002_000004.json",
        [{"packet_id": f"p{i}", "decision": "B"} for i in range(2, 4)],
    )
    rows, evidence = run_batch_with_recovery(
        args=args,
        schema_path=schema,
        state_dir=state,
        shard_name="shard",
        packets=packets,
        start=0,
        end=4,
        rubric="rubric",
    )
    assert [row["packet_id"] for row in rows] == ["p0", "p1", "p2", "p3"]
    assert evidence == parent
    payload = json.loads(parent.read_text(encoding="utf-8"))
    assert len(payload["decisions"]) == 4
    assert len(list(state.glob(parent.name + ".invalid*"))) == 3
