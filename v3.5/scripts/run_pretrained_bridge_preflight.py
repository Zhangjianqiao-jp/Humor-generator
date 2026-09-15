#!/usr/bin/env python3
"""Current-route CPU preflight for the sealed HOMER + bridge experiment.

This intentionally does not invoke the historical 2,846-row/666-trace audit.
The current route is the immutable 362-contest adapter-free public release;
the historical audit remains a separately reported provenance check.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
CONFIG = ROOT / "configs/pilot/cross_attention_caption_pretrained_public.yaml"


def run(argv: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
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
        "--output", type=Path,
        default=ROOT / "results/preflight/pretrained_bridge_preflight.json",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    commands = [
        [str(PYTHON), "-m", "compileall", "-q", "src", "scripts", "tests"],
        [str(PYTHON), "scripts/homer_public_code_smoke.py"],
        [str(PYTHON), "scripts/verify_homer_public_release_362.py"],
        [str(PYTHON), "scripts/verify_homer_evaluation_assets.py"],
        [
            str(PYTHON), "scripts/check_pretrained_route.py",
            "--require-data-ready", "--require-context-ready",
            "--require-bridge-data-ready",
        ],
        [
            str(PYTHON), "scripts/verify_pretrained_bridge_inputs.py",
            "--config", str(CONFIG),
            "--output", str(ROOT / "results/preflight/pretrained_bridge_inputs_public.json"),
        ],
        [str(PYTHON), "scripts/engineering_smoke.py"],
        [str(PYTHON), "scripts/validate_project_memory.py"],
    ]
    if not args.skip_tests:
        commands.insert(1, [str(PYTHON), "-m", "pytest", "-q"])

    reports: list[dict[str, Any]] = []
    for command in commands:
        report = run(command)
        reports.append(report)
        if report["returncode"] != 0:
            break
    failure = next(
        (item for item in reports if item["returncode"] != 0), None
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "pass" if failure is None else "fail",
        "failure_command": None if failure is None else failure["argv"],
        "git_commit": commit,
        "route": "homer_public_code_bridge",
        "data_version": "homer_pretrained_7b_public_release_362",
        "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "historical_audit_intentionally_separate": True,
        "commands": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
