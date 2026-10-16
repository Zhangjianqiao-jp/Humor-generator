from __future__ import annotations

import json

from scripts.homer_evaluation_controller import (
    PROMPT_SHA256,
    SOURCE_PROVENANCE_SHA256,
    account_environment,
    checkpoint_progress,
    format_status,
    finalization_complete,
    launch_allowed,
    progress,
    row_valid,
)
from scripts.run_homer_account_agent_batches import (
    classify_codex_failure,
    parse_retry_after,
    sanitized_environment,
    validate_output,
)


def test_account_environment_does_not_forward_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-forward")
    monkeypatch.setenv("OPEN_API_KEY", "must-not-forward")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-forward")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-forward")
    monkeypatch.setenv("MY_SERVICE_API_KEY", "must-not-forward")
    environment = account_environment()
    assert "OPENAI_API_KEY" not in environment
    assert "OPEN_API_KEY" not in environment
    assert "CODEX_API_KEY" not in environment
    assert "ANTHROPIC_API_KEY" not in environment
    assert "MY_SERVICE_API_KEY" not in environment


def test_runner_environment_does_not_forward_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("PROVIDER_API_KEY", "must-not-forward")
    monkeypatch.setenv("UNRELATED_SETTING", "kept")
    environment = sanitized_environment()
    assert "PROVIDER_API_KEY" not in environment
    assert environment["UNRELATED_SETTING"] == "kept"


def test_judgment_row_contract_is_fail_closed() -> None:
    valid = {
        "packet_id": "packet-1",
        "decision": "A",
        "model": "gpt-5.5",
        "temperature": 0,
        "prompt_sha256": PROMPT_SHA256,
        "source_provenance_sha256": SOURCE_PROVENANCE_SHA256,
    }
    assert row_valid(valid, "gpt-5.5", "packet-1")
    assert not row_valid({**valid, "decision": "BLAH"}, "gpt-5.5", "packet-1")
    assert not row_valid({**valid, "model": "gpt-5.4"}, "gpt-5.5", "packet-1")
    assert not row_valid({**valid, "packet_id": "packet-2"}, "gpt-5.5", "packet-1")


def test_status_is_context_free_and_machine_readable() -> None:
    status = {
        "updated_at": "now",
        "canonical_packets": 10,
        "models": {
            "gpt-5.5": {"rows": 10, "total": 10, "complete": True, "blocked": None},
            "gpt-5.6-sol": {"rows": 4, "total": 10, "complete": False},
        },
        "terra_reaudit": {"shard_a": True, "shard_b": False},
    }
    text = format_status(status)
    assert "packets=10" in text
    assert "gpt-5.5: 10/10 (complete)" in text
    assert "gpt-5.6-sol: 4/10 (running)" in text
    assert "terra re-audit: 1/2 shards complete" in text


def test_existing_output_reuse_is_fail_closed() -> None:
    packets = [{"packet_id": "packet-1"}, {"packet_id": "packet-2"}]
    row = {
        "packet_id": "packet-1",
        "decision": "A",
        "model": "gpt-5.5",
        "temperature": 0,
        "prompt_sha256": PROMPT_SHA256,
        "source_provenance_sha256": SOURCE_PROVENANCE_SHA256,
    }
    try:
        validate_output([row], packets, model="gpt-5.5")
    except RuntimeError as exc:
        assert "count" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("partial final output must not be reused")

    valid = [row, {**row, "packet_id": "packet-2", "decision": "B"}]
    validate_output(valid, packets, model="gpt-5.5")
    try:
        validate_output([{**valid[0], "model": "gpt-5.6-sol"}, valid[1]], packets, model="gpt-5.5")
    except RuntimeError as exc:
        assert "model" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("stale model output must not be reused")


def test_checkpoint_progress_does_not_double_count_adaptive_parent_and_children(tmp_path) -> None:
    root = tmp_path
    shard = root / "multimodel_agent_shards" / "shard_000000"
    shard.mkdir(parents=True)
    packets = [{"packet_id": f"p{i}"} for i in range(4)]
    (shard / "public_packets.jsonl").write_text(
        "".join(json.dumps(packet) + "\n" for packet in packets), encoding="utf-8"
    )
    state = root / "gpt55_account_batches"
    state.mkdir()
    decisions = lambda start, end: {  # noqa: E731 - compact fixture helper
        "decisions": [
            {"packet_id": f"p{i}", "decision": "A" if i < 2 else "B"}
            for i in range(start, end)
        ]
    }
    (state / "shard_000000_000000_000004.json").write_text(json.dumps(decisions(0, 4)), encoding="utf-8")
    (state / "shard_000000_000000_000002.json").write_text(json.dumps(decisions(0, 2)), encoding="utf-8")
    (state / "shard_000000_000002_000004.json").write_text(json.dumps(decisions(2, 4)), encoding="utf-8")
    progress = checkpoint_progress(root, "gpt-5.5")
    assert progress["rows"] == 4
    assert progress["batches"] == 1


def test_progress_keeps_terra_gate_after_promotion_moves_reaudit_files(tmp_path, monkeypatch) -> None:
    import scripts.homer_evaluation_controller as controller

    shard = tmp_path / "shard_001201_002400"
    shards = [shard / "public_packets.jsonl"]
    monkeypatch.setattr(controller, "validate_public_layout", lambda root: ([shard], 1, "c" * 64))
    monkeypatch.setattr(controller, "public_ids", lambda path: ["p0"])
    monkeypatch.setattr(controller, "shard_count", lambda path: 1)
    monkeypatch.setattr(controller, "valid_output", lambda *args, **kwargs: True)
    monkeypatch.setattr(controller, "checkpoint_progress", lambda *args, **kwargs: {"rows": 0, "batches": 0, "active": None})
    monkeypatch.setattr(controller, "terra_approved", lambda root: True)
    value = progress(tmp_path)
    assert value["terra_reaudit"] == {name: True for name in controller.SUSPECT_TERRA if name == shard.name}
    assert value["terra_reaudit_complete"] is True


def test_finalization_sentinel_is_fail_closed(tmp_path) -> None:
    output_dir = tmp_path / "multimodel_agent_final"
    output_dir.mkdir()
    summary = output_dir / "multimodel_summary.json"
    markdown = output_dir / "multimodel_summary.md"
    summary.write_text('{"status":"complete_auxiliary_multimodel"}\n', encoding="utf-8")
    markdown.write_text("partial", encoding="utf-8")
    assert not finalization_complete(tmp_path, 10)

    summary.write_text(
        '{"status":"complete_auxiliary_multimodel","packet_count":10,'
        '"evaluator_model_ids":["gpt-5.5","gpt-5.6-sol","gpt-5.6-terra"],'
        '"evaluators":[{"model_id":"gpt-5.5","metrics":{"summaries":[1],"paired_comparison":[1]}},'
        '{"model_id":"gpt-5.6-sol","metrics":{"summaries":[1],"paired_comparison":[1]}},'
        '{"model_id":"gpt-5.6-terra","metrics":{"summaries":[1],"paired_comparison":[1]}}],'
        '"majority":{"metrics":{"paired_comparison":[1]}}}\n',
        encoding="utf-8",
    )
    assert finalization_complete(tmp_path, 10)
    summary.write_text(
        summary.read_text(encoding="utf-8").replace('"packet_count":10', '"packet_count":9'),
        encoding="utf-8",
    )
    assert not finalization_complete(tmp_path, 10)


def test_codex_failure_classification_pauses_expensive_retries() -> None:
    unsupported = classify_codex_failure("The 'gpt-5.6-sol' model is not supported when using Codex with a ChatGPT account.")
    assert unsupported == {
        "kind": "model_unsupported",
        "retryable": False,
        "retry_after_epoch": None,
        "retry_after": None,
    }
    expired = classify_codex_failure("Provided authentication token is expired (token_expired).")
    assert expired["kind"] == "auth_expired"
    quota_message = "You've hit your usage limit; try again at Oct 13th, 2026 1:43 PM."
    quota = classify_codex_failure(quota_message)
    assert quota["kind"] == "quota_exhausted"
    assert quota["retryable"] is True
    epoch, timestamp = parse_retry_after(quota_message)
    assert isinstance(epoch, float)
    assert timestamp is not None

    transient = classify_codex_failure("503 Service Unavailable from evaluator")
    assert transient["kind"] == "transient_error"
    assert transient["retryable"] is True
    assert transient["retry_after_epoch"] > 0


def test_restart_budget_is_separate_from_historical_launch_count() -> None:
    state = {
        "no_progress_launches": {"gpt-5.5": 2},
        "restart_limit_models": {},
    }
    assert not launch_allowed(state, "gpt-5.5", 2)
    assert state["restart_limit_models"]["gpt-5.5"]["kind"] == "restart_limit_exceeded"
    state["no_progress_launches"]["gpt-5.5"] = 0
    state["restart_limit_models"].pop("gpt-5.5")
    assert launch_allowed(state, "gpt-5.5", 2)
