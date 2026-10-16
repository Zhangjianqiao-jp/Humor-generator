#!/usr/bin/env python3
"""Keep the no-API-key HOMER account-agent evaluation resumable.

The supervisor never judges packets itself.  It only restarts a bounded
runner after a transient account/quota failure, promotes the audited Terra
shards, and runs the fail-closed merge/finalizer once all three evaluators have
complete packet sets.  Existing processes are detected before a new runner is
started, so this can be launched while a runner is already active.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


SUSPECT_TERRA = {
    "shard_001201_002400",
    "shard_006001_010800",
    "shard_010801_015600",
    "shard_025201_030000",
    "shard_030001_034800",
}


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n")


def process_has(marker: str) -> bool:
    current = os.getpid()
    try:
        rows = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    except subprocess.CalledProcessError:
        return False
    for row in rows:
        fields = row.strip().split(None, 1)
        if len(fields) == 2 and int(fields[0]) != current and marker in fields[1]:
            return True
    return False


def packet_count(shard: Path) -> int:
    return sum(1 for line in (shard / "public_packets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())


def complete(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    try:
        return sum(1 for line in path.open(encoding="utf-8") if line.strip()) == expected
    except OSError:
        return False


def all_shards_complete(shard_dir: Path, filename: str) -> bool:
    shards = sorted(p for p in shard_dir.glob("shard_*") if p.is_dir())
    return bool(shards) and all(complete(shard / filename, packet_count(shard)) for shard in shards)


def promote_terra(shard_dir: Path, log: Path) -> bool:
    """Atomically promote only a complete, re-audited Terra set."""
    for name in sorted(SUSPECT_TERRA):
        shard = shard_dir / name
        source = shard / "judge_gpt56terra_reaudit.jsonl"
        if not complete(source, packet_count(shard)):
            return False
    for name in sorted(SUSPECT_TERRA):
        shard = shard_dir / name
        source = shard / "judge_gpt56terra_reaudit.jsonl"
        target = shard / "judge_gpt56terra.jsonl"
        backup = shard / "judge_gpt56terra.heuristic_invalid.jsonl"
        if target.exists() and not backup.exists():
            target.rename(backup)
            log_line(log, f"backed up heuristic Terra shard {target} -> {backup}")
        if source.exists():
            source.rename(target)
            log_line(log, f"promoted audited Terra shard {source} -> {target}")
    audit = shard_dir.parent / "terra_reaudit_promotion.json"
    audit.write_text(
        json.dumps(
            {
                "status": "promoted",
                "suspect_shards": sorted(SUSPECT_TERRA),
                "replacement": "judge_gpt56terra_reaudit.jsonl",
                "backup": "judge_gpt56terra.heuristic_invalid.jsonl",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def run_child(command: list[str], log: Path) -> None:
    log_line(log, "starting " + " ".join(command))
    with log.open("a", encoding="utf-8") as handle:
        subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=int, default=120)
    parser.add_argument("--max-restarts", type=int, default=100)
    parser.add_argument("--log", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    python = project / ".venv" / "bin" / "python"
    root = args.root.resolve()
    shards = root / "multimodel_agent_shards"
    log = args.log.resolve()
    gpt55 = root / "gpt55_batch_runner.py"
    sol = root / "sol_batch_runner.py"
    generic = project / "scripts" / "run_homer_account_agent_batches.py"
    terra_state = root / "terra_reaudit_batches"
    specs: list[tuple[str, str, list[str]]] = [
        ("gpt-5.5", "gpt55_batch_runner.py", [str(python), str(gpt55)]),
        ("gpt-5.6-sol", "sol_batch_runner.py", [str(python), str(sol)]),
        (
            "gpt-5.6-terra",
            "run_homer_account_agent_batches.py",
            [
                str(python), str(generic), "--root", str(root), "--model", "gpt-5.6-terra",
                "--output-name", "judge_gpt56terra_reaudit.jsonl", "--state-dir", str(terra_state),
                "--batch-size", "100", "--shards", *sorted(SUSPECT_TERRA),
            ],
        ),
    ]
    restarts = 0
    while restarts <= args.max_restarts:
        # Always inspect the re-audit files when they exist.  The legacy
        # heuristic Terra files are complete-looking, so checking only the
        # target filename would incorrectly skip the promotion step.
        terra_reaudit_present = any(
            (shards / name / "judge_gpt56terra_reaudit.jsonl").exists()
            for name in SUSPECT_TERRA
        )
        if terra_reaudit_present or not all_shards_complete(shards, "judge_gpt56terra.jsonl"):
            promote_terra(shards, log)
        # A complete-looking legacy target is not sufficient: while any
        # re-audit source remains, Terra is deliberately *not* considered
        # complete.  This prevents finalization from consuming heuristic
        # rows if the other evaluators finish first.
        terra_ok = (
            all_shards_complete(shards, "judge_gpt56terra.jsonl")
            and not any(
                (shards / name / "judge_gpt56terra_reaudit.jsonl").exists()
                for name in SUSPECT_TERRA
            )
        )
        done = (
            all_shards_complete(shards, "judge_gpt55.jsonl")
            and all_shards_complete(shards, "judge_gpt56sol.jsonl")
            and terra_ok
        )
        if done:
            log_line(log, "all evaluator shard outputs complete; running merge")
            merged = root / "multimodel_agent_merged"
            final = root / "multimodel_agent_final"
            merge = [
                str(python), str(project / "scripts" / "merge_homer_agent_shards.py"),
                "--packets", str(root / "homer_primary" / "homer_primary_prompts.jsonl"),
                "--shard-dir", str(shards), "--output-dir", str(merged), "--require-complete",
            ]
            for spec in ("gpt-5.5=judge_gpt55.jsonl", "gpt-5.6-sol=judge_gpt56sol.jsonl", "gpt-5.6-terra=judge_gpt56terra.jsonl"):
                merge.extend(["--models", spec])
            with log.open("a", encoding="utf-8") as handle:
                result = subprocess.run(merge, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                log_line(log, f"merge failed with {result.returncode}")
                return result.returncode
            final_cmd = [
                str(python), str(project / "scripts" / "finalize_homer_auxiliary_models.py"),
                "--private-mapping", str(root / "homer_primary" / "homer_primary_private_mapping.jsonl"),
                "--provenance", str(root / "homer_primary" / "provenance.json"), "--output-dir", str(final),
                "--bootstrap-replicates", "2000",
            ]
            for spec in (
                f"gpt-5.5={merged / 'judge_gpt55.jsonl'}",
                f"gpt-5.6-sol={merged / 'judge_gpt56sol.jsonl'}",
                f"gpt-5.6-terra={merged / 'judge_gpt56terra.jsonl'}",
            ):
                final_cmd.extend(["--judgments", spec])
            with log.open("a", encoding="utf-8") as handle:
                result = subprocess.run(final_cmd, stdout=handle, stderr=subprocess.STDOUT)
            log_line(log, f"finalizer exit={result.returncode}")
            return result.returncode

        for model, marker, command in specs:
            if model == "gpt-5.5" and all_shards_complete(shards, "judge_gpt55.jsonl"):
                continue
            if model == "gpt-5.6-sol" and all_shards_complete(shards, "judge_gpt56sol.jsonl"):
                continue
            if model == "gpt-5.6-terra" and terra_ok:
                continue
            if not process_has(marker):
                restarts += 1
                run_child(command, log)
                log_line(log, f"restart_count={restarts} model={model}")
                if restarts > args.max_restarts:
                    return 2
        time.sleep(max(10, args.interval_seconds))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
