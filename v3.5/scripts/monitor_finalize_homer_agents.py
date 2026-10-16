#!/usr/bin/env python3
"""Monitor account-agent HOMER judgments and finalize once complete.

This script does not run any judge model. It only watches for already produced
public-shard judgment files, runs the public shard merger, and when all requested
models are complete, runs the private-map finalizer inside the local project.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


EXPECTED_TOTAL = 35_250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--max-iterations", type=int, default=144)
    parser.add_argument("--log", required=True, type=Path)
    return parser.parse_args()


def run(cmd: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.write(proc.stdout)
        fh.write(proc.stderr)
    return proc


def read_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    python = project_root / ".venv" / "bin" / "python"
    merged = args.root / "multimodel_agent_merged"
    final = args.root / "multimodel_agent_final"
    progress_path = merged / "merge_progress.json"

    model_specs = [
        "gpt-5.5=judge_gpt55.jsonl",
        "gpt-5.6-sol=judge_gpt56sol.jsonl",
        "gpt-5.6-terra=judge_gpt56terra.jsonl",
    ]
    judgment_args = [
        "gpt-5.5=" + str(merged / "judge_gpt55.jsonl"),
        "gpt-5.6-sol=" + str(merged / "judge_gpt56sol.jsonl"),
        "gpt-5.6-terra=" + str(merged / "judge_gpt56terra.jsonl"),
    ]

    for idx in range(args.max_iterations):
        merge_cmd = [
            str(python),
            str(project_root / "scripts" / "merge_homer_agent_shards.py"),
            "--packets",
            str(args.root / "homer_primary" / "homer_primary_prompts.jsonl"),
            "--shard-dir",
            str(args.root / "multimodel_agent_shards"),
            "--output-dir",
            str(merged),
        ]
        for spec in model_specs:
            merge_cmd.extend(["--models", spec])
        proc = run(merge_cmd, args.log)
        progress = read_progress(progress_path)
        with args.log.open("a", encoding="utf-8") as fh:
            fh.write(f"\niteration={idx} returncode={proc.returncode} progress={json.dumps(progress, ensure_ascii=False)}\n")

        counts = {
            model: info.get("rows", 0)
            for model, info in progress.get("models", {}).items()
        }
        if all(counts.get(model, 0) == EXPECTED_TOTAL for model in ["gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra"]):
            final_cmd = [
                str(python),
                str(project_root / "scripts" / "finalize_homer_auxiliary_models.py"),
                "--private-mapping",
                str(args.root / "homer_primary" / "homer_primary_private_mapping.jsonl"),
                "--provenance",
                str(args.root / "homer_primary" / "provenance.json"),
                "--output-dir",
                str(final),
                "--bootstrap-replicates",
                "2000",
            ]
            for spec in judgment_args:
                final_cmd.extend(["--judgments", spec])
            final_proc = run(final_cmd, args.log)
            return final_proc.returncode

        time.sleep(args.interval_seconds)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
