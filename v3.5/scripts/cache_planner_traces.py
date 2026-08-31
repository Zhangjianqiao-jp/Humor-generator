#!/usr/bin/env python3
"""Generate validated HOMER Planner text and exactly aligned hidden states."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import plan_to_record, read_jsonl, save_trace
from humor_generator_v35.homer.contracts import parse_conflicts, validate_plan
from humor_generator_v35.homer.prompts import (
    conflict_messages,
    global_imagination_messages,
    local_imagination_messages,
)
from humor_generator_v35.latent.state_capture import AlignedMessageStates
from humor_generator_v35.qwen_backend import QwenBackend


MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit() -> str:
    repo = ROOT.parent
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "v3.5"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "formal traces require a committed, clean v3.5 tree; commit the listed changes first:\n"
            + status
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("could not resolve a full Git commit")
    return commit


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/latent_bridge_v35")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/cache/planner_traces_homer_strict_v35",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--retry-round", type=int, default=0)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    if args.attempts < 1:
        raise ValueError("attempts must be positive")
    if args.retry_round < 0:
        raise ValueError("retry-round must be non-negative")

    provenance = {
        "git_commit": repository_commit(),
        "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
        "homer_prompts_sha256": sha256(ROOT / "src/humor_generator_v35/homer/prompts.py"),
        "adapter_manifest_sha256": sha256(ROOT / "manifests/frozen_7b_adapters.json"),
    }

    rows: dict[str, dict] = {}
    for split in args.splits:
        for row in read_jsonl(args.dataset / f"{split}.jsonl"):
            rows.setdefault(row["cluster_id"], {**row, "split": split})
    selected = [rows[key] for key in sorted(rows, key=lambda x: int(x.rsplit("_", 1)[1]))]
    if args.max_clusters is not None:
        selected = selected[: args.max_clusters]

    backend = QwenBackend.load(
        MODEL,
        revision=REVISION,
        adapter=ROOT / "artifacts/checkpoints/planner_sft",
        load_in_4bit=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.jsonl"
    existing = {item["cluster_id"]: item for item in read_jsonl(index_path)} if index_path.exists() else {}
    failures: list[dict] = []
    completed_success = len(existing)
    with index_path.open("a", encoding="utf-8") as index:
        for offset, row in enumerate(selected):
            cluster = row["cluster_id"]
            if cluster in existing:
                continue
            last_error = ""
            last_outputs: dict[str, str] = {}
            for attempt in range(args.attempts):
                attempt_outputs: dict[str, str] = {}
                seed = (
                    args.seed
                    + args.retry_round * 10_000_000
                    + offset * args.attempts * 3
                    + attempt * 3
                )
                temperature = 0.0 if attempt == 0 else 0.7
                top_p = 1.0 if attempt == 0 else 0.95
                try:
                    conflict_text, conflict_states, conflict_alignment = backend.generate_and_verify_states(
                        conflict_messages(row["standard_description"]), max_new_tokens=384, seed=seed,
                        temperature=temperature, top_p=top_p,
                    )
                    # Strict conflict parsing precedes both association calls.
                    conflicts = parse_conflicts(conflict_text)
                    conflict_states = AlignedMessageStates(
                        conflict_states.token_ids, conflict_states.states, conflict_text
                    )
                    attempt_outputs["conflict"] = conflict_text
                except Exception as exc:
                    last_error = f"conflict/alignment: {exc}"
                    last_outputs = attempt_outputs
                    continue
                normalized = " ".join(
                    f"{i}. {pair.render()}" for i, pair in enumerate(conflicts, 1)
                )
                try:
                    local_text, local_states, local_alignment = backend.generate_and_verify_states(
                        local_imagination_messages(row["standard_description"], normalized),
                        max_new_tokens=512, seed=seed + 1,
                        temperature=temperature, top_p=top_p,
                    )
                    attempt_outputs["local"] = local_text
                    local_states = AlignedMessageStates(
                        local_states.token_ids, local_states.states, local_text
                    )
                    global_text, global_states, global_alignment = backend.generate_and_verify_states(
                        global_imagination_messages(row["image"], normalized),
                        max_new_tokens=512, seed=seed + 2,
                        temperature=temperature, top_p=top_p,
                    )
                    attempt_outputs["global"] = global_text
                    global_states = AlignedMessageStates(
                        global_states.token_ids, global_states.states, global_text
                    )
                    plan = validate_plan(
                        row["standard_description"], normalized, local_text, global_text
                    )
                except Exception as exc:
                    last_error = f"association/alignment: {exc}"
                    last_outputs = attempt_outputs
                    continue
                trace_path = args.output / "states" / f"{cluster}.pt"
                trace_hash = save_trace(trace_path, {
                    "conflict": conflict_states,
                    "local": local_states,
                    "global": global_states,
                })
                alignment = {
                    name: {
                        "replay": report.replay.__dict__,
                        "processed_score_token_accuracy": report.processed_score_token_accuracy,
                        "raw_head_token_accuracy_diagnostic": report.raw_head_token_accuracy,
                        "emitted_token_mean_logprob": report.emitted_token_mean_logprob,
                        "sampling_mode": report.sampling_mode,
                        "communication_state_definition": report.communication_state_definition,
                    }
                    for name, report in {
                        "conflict": conflict_alignment,
                        "local": local_alignment,
                        "global": global_alignment,
                    }.items()
                }
                record = {
                    "schema_version": 2,
                    "cluster_id": cluster,
                    "split": row["split"],
                    "trace_path": str(trace_path.relative_to(ROOT)),
                    "trace_sha256": trace_hash,
                    "planner_model": MODEL,
                    "planner_revision": REVISION,
                    "planner_adapter": "planner_sft",
                    "provenance": provenance,
                    "seed": seed,
                    "attempt": attempt,
                    "retry_round": args.retry_round,
                    "generation": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "repetition_penalty": 1.0,
                    },
                    "plan": plan_to_record(plan),
                    "alignment": alignment,
                }
                index.write(json.dumps(record, ensure_ascii=False) + "\n")
                index.flush()
                completed_success += 1
                if completed_success % 10 == 0 or completed_success == len(selected):
                    progress = {
                        "requested": len(selected),
                        "completed": completed_success,
                        "failed": len(failures),
                        "last_cluster": cluster,
                    }
                    (args.output / "progress.json").write_text(
                        json.dumps(progress, indent=2) + "\n"
                    )
                    print(json.dumps(progress), flush=True)
                break
            else:
                failures.append({
                    "cluster_id": cluster,
                    "error": last_error,
                    "last_outputs": last_outputs,
                })
    (args.output / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"requested": len(selected), "failed": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
