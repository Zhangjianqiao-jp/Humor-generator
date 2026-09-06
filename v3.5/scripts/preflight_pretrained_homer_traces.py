#!/usr/bin/env python3
"""CPU-only preflight for the adapter-free 362-contest trace job.

This deliberately does not load Qwen or request CUDA.  It checks the sealed
population, prompt-stage smoke, environment/isolation contracts and the
formal clean-tree requirement before a GPU job is submitted.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str]) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, env=env)
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    python = str(ROOT / ".venv/bin/python")
    commands = [
        [python, "-m", "py_compile", "scripts/cache_pretrained_homer_traces.py", "scripts/verify_pretrained_homer_traces.py"],
        [python, "scripts/homer_public_code_smoke.py"],
        [python, "scripts/verify_homer_public_release_362.py"],
        [python, "scripts/check_environment.py"],
        [python, "scripts/check_v35_isolation.py"],
        [python, "scripts/verify_frozen_artifacts.py"],
    ]
    results = []
    failure: str | None = None
    for argv in commands:
        result = run(argv)
        results.append(result)
        if result["returncode"] != 0:
            failure = "command_failed"
            break
    repo = ROOT.parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "--", "v3.5"], cwd=repo
    ).returncode != 0
    if dirty and not args.allow_dirty:
        failure = failure or "dirty_tracked_worktree"
    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pass" if failure is None else "fail",
        "failure": failure,
        "git_commit": commit,
        "tracked_worktree_dirty": dirty,
        "population_manifest": "manifests/homer_population_public_release_362.json",
        "population_manifest_sha256": sha256(ROOT / "manifests/homer_population_public_release_362.json"),
        "trace_input_manifest": "data/processed/homer_pretrained_7b_public_release_362/trace_inputs.jsonl",
        "trace_input_manifest_sha256": sha256(ROOT / "data/processed/homer_pretrained_7b_public_release_362/trace_inputs.jsonl"),
        "commands": results,
        "gpu_forward_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
