#!/usr/bin/env python3
"""Generate validated HOMER Planner text and exactly aligned hidden states."""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v3.data.traces import plan_to_record, read_jsonl, save_trace
from humor_generator_v3.homer.contracts import parse_conflicts, validate_plan
from humor_generator_v3.homer.prompts import (
    conflict_messages,
    global_imagination_messages,
    local_imagination_messages,
)
from humor_generator_v3.qwen_backend import QwenBackend


MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/latent_bridge_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "data/cache/planner_traces")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    if args.attempts < 1:
        raise ValueError("attempts must be positive")

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
                seed = args.seed + offset * args.attempts * 3 + attempt * 3
                try:
                    conflict_text, conflict_states, conflict_alignment = backend.generate_and_verify_states(
                        conflict_messages(row["standard_description"]), max_new_tokens=384, seed=seed
                    )
                    # Strict conflict parsing precedes both association calls.
                    conflicts = parse_conflicts(conflict_text)
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
                        max_new_tokens=512,
                        seed=seed + 1,
                    )
                    attempt_outputs["local"] = local_text
                    global_text, global_states, global_alignment = backend.generate_and_verify_states(
                        global_imagination_messages(row["image"], normalized),
                        max_new_tokens=512,
                        seed=seed + 2,
                    )
                    attempt_outputs["global"] = global_text
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
                    name: report.replay.__dict__
                    for name, report in {
                        "conflict": conflict_alignment,
                        "local": local_alignment,
                        "global": global_alignment,
                    }.items()
                }
                record = {
                    "schema_version": 1,
                    "cluster_id": cluster,
                    "split": row["split"],
                    "trace_path": str(trace_path.relative_to(ROOT)),
                    "trace_sha256": trace_hash,
                    "planner_model": MODEL,
                    "planner_revision": REVISION,
                    "planner_adapter": "planner_sft",
                    "seed": seed,
                    "attempt": attempt,
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
