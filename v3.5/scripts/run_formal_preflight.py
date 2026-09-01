#!/usr/bin/env python3
"""Fail-closed static/interface/data preflight before any formal GPU work."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str]) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        argv, cwd=ROOT, text=True, capture_output=True, env=environment
    )
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/preflight/formal_preflight.json"
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    python = str(ROOT / ".venv/bin/python")
    commands = [
        [python, "-m", "compileall", "-q", "src", "scripts", "tests"],
        [python, "scripts/check_environment.py"],
        [python, "scripts/check_v35_isolation.py"],
        [python, "scripts/verify_frozen_artifacts.py"],
        [python, "scripts/train_bridge.py", "--help"],
        [python, "scripts/real_trace_bridge_smoke.py", "--help"],
        [python, "scripts/generate_formal_baseline.py", "--help"],
        [python, "scripts/formal_pipeline_monitor.py", "--help"],
        [python, "scripts/verify_clustered_dataset.py"],
        [python, "scripts/check_trace_completion.py"],
    ]
    if not args.skip_tests:
        commands.insert(1, [python, "-m", "pytest", "-q"])
    results = []
    failure: str | None = None
    for argv in commands:
        result = run(argv)
        results.append(result)
        if result["returncode"] != 0:
            failure = "command_failed"
            break
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
    if dirty and not args.allow_dirty:
        failure = failure or "dirty_tracked_worktree"
    dataset_summary = ROOT / "results/preflight/dataset_audit/dataset_audit_summary.json"
    trace_summary = ROOT / "results/preflight/trace_audit/trace_audit_summary.json"
    evidence = {}
    for name, path in (("dataset", dataset_summary), ("traces", trace_summary)):
        if not path.is_file():
            failure = failure or f"missing_{name}_summary"
            continue
        payload = json.loads(path.read_text())
        evidence[name] = {"path": str(path), "sha256": sha256(path), "summary": payload}
        if name == "dataset" and not (
            payload.get("status") == "pass"
            and payload.get("rows_checked") == payload.get("rows_passed") == 2846
            and payload.get("unique_images_checked") == payload.get("unique_images_passed") == 949
            and payload.get("source_inputs_checked") == payload.get("source_inputs_passed") == 367
        ):
            failure = failure or "dataset_summary_contract_failed"
        if name == "traces" and not (
            payload.get("trace_records_checked") == payload.get("trace_records_passed") == 666
            and payload.get("invalid_trace_records") == 0
        ):
            failure = failure or "trace_summary_contract_failed"
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "pass" if failure is None else "fail",
        "failure": failure,
        "git_commit": commit,
        "tracked_worktree_dirty": dirty,
        "tests_skipped": args.skip_tests,
        "commands": results,
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failure is not None:
        raise SystemExit(f"formal preflight failed: {failure}")


if __name__ == "__main__":
    main()
