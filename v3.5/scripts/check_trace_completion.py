#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import hashlib

import torch

from humor_generator_v35.data.traces import load_trace, plan_from_record


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    dataset = ROOT / "data/processed/latent_bridge_v35"
    cache = ROOT / "data/cache/planner_traces_homer_strict_v35"
    required = {
        row["cluster_id"]
        for split in ("train", "validation")
        for row in read_jsonl(dataset / f"{split}.jsonl")
    }
    records = read_jsonl(cache / "index.jsonl")
    available = {record["cluster_id"] for record in records}
    failures = json.loads((cache / "failures.json").read_text())
    missing = sorted(required - available)
    extra = sorted(available - required)
    invalid = []
    record_checks = []
    current_hashes = {
        "trace_input_manifest_sha256": sha256(dataset / "trace_inputs.jsonl"),
        "homer_prompts_sha256": sha256(ROOT / "src/humor_generator_v35/homer/prompts.py"),
        "adapter_manifest_sha256": sha256(ROOT / "manifests/frozen_7b_adapters.json"),
    }
    provenance_values: set[tuple[str, str, str, str]] = set()
    for record in records:
        cluster_id = record.get("cluster_id")
        check = {"cluster_id": cluster_id, "status": "fail"}
        try:
            if record.get("schema_version") != 3:
                raise ValueError("formal trace must use provenance schema version 3")
            provenance = record.get("provenance", {})
            required_provenance = {
                "git_commit", "trace_input_manifest_sha256", "homer_prompts_sha256",
                "adapter_manifest_sha256",
            }
            if set(provenance) != required_provenance:
                raise ValueError("missing or extra trace provenance fields")
            if not re.fullmatch(r"[0-9a-f]{40}", provenance["git_commit"]):
                raise ValueError("invalid trace Git commit")
            for key in required_provenance - {"git_commit"}:
                if not re.fullmatch(r"[0-9a-f]{64}", provenance[key]):
                    raise ValueError(f"invalid provenance hash: {key}")
                if provenance[key] != current_hashes[key]:
                    raise ValueError(f"stale trace provenance hash: {key}")
            provenance_values.add(tuple(provenance[key] for key in sorted(required_provenance)))
            plan_from_record(record["plan"])
            trace_path = (ROOT / record["trace_path"]).resolve(strict=True)
            if ROOT not in trace_path.parents:
                raise ValueError("trace path escapes v3.5 root")
            if not re.fullmatch(r"[0-9a-f]{64}", record.get("trace_sha256", "")):
                raise ValueError("invalid trace SHA-256")
            loaded = load_trace(trace_path, expected_sha256=record["trace_sha256"])
            if any(not torch.isfinite(item.states).all() for item in loaded.values()):
                raise ValueError("trace contains non-finite hidden states")
            if any((item.token_ids < 0).any() for item in loaded.values()):
                raise ValueError("trace contains negative token IDs")
            if record.get("split") not in {"train", "validation"}:
                raise ValueError(f"invalid split {record.get('split')!r}")
            if set(record.get("alignment", {})) != {"conflict", "local", "global"}:
                raise ValueError("missing channel alignment evidence")
            repairs = record.get("validator_repair")
            repaired_channels = set()
            if repairs is not None:
                if repairs.get("policy_version") != "validator-feedback-format-only-v1":
                    raise ValueError("unknown validator repair policy")
                if not re.fullmatch(r"[0-9a-f]{64}", repairs.get("repair_prompt_sha256", "")):
                    raise ValueError("invalid validator repair prompt hash")
                current_repair_hash = sha256(ROOT / "src/humor_generator_v35/homer/repair.py")
                if repairs["repair_prompt_sha256"] != current_repair_hash:
                    raise ValueError("stale validator repair prompt hash")
                repaired_channels = set(repairs.get("channels", {}))
            for channel, evidence in record["alignment"].items():
                if channel in repaired_channels:
                    if evidence.get("state_capture_mode") != "teacher_forced_replay_after_validator_repair":
                        raise ValueError(f"invalid repaired replay mode for {channel}")
                    if evidence.get("exact_text_roundtrip") is not True:
                        raise ValueError(f"failed repaired text replay for {channel}")
                    if evidence.get("assistant_token_count", 0) < 1:
                        raise ValueError(f"empty repaired replay for {channel}")
                else:
                    replay = evidence.get("replay", {})
                    if replay.get("mean_cosine", 0) < 0.98 or replay.get("min_cosine", 0) < 0.90:
                        raise ValueError(f"failed causal replay evidence for {channel}")
                    if evidence.get("sampling_mode") not in {"greedy", "sample"}:
                        raise ValueError(f"missing sampling mode for {channel}")
                if evidence.get("communication_state_definition") != "teacher_forced_post_token":
                    raise ValueError(f"invalid communication state definition for {channel}")
            check.update({
                "status": "pass",
                "split": record["split"],
                "trace_sha256": record["trace_sha256"],
                "channel_tokens": {
                    name: int(item.token_ids.numel()) for name, item in loaded.items()
                },
                "hidden_width": {
                    name: int(item.states.shape[-1]) for name, item in loaded.items()
                },
            })
        except Exception as exc:
            check["error"] = str(exc)
            invalid.append({"cluster_id": cluster_id, "error": str(exc)})
        record_checks.append(check)
    output_dir = ROOT / "results/preflight/trace_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trace_checks.jsonl").open("w") as handle:
        for check in record_checks:
            handle.write(json.dumps(check, ensure_ascii=False) + "\n")
    report = {
        "required_clusters": len(required),
        "available_clusters": len(available),
        "failure_records": len(failures),
        "missing_clusters": len(missing),
        "extra_clusters": len(extra),
        "invalid_trace_records": len(invalid),
        "trace_records_checked": len(record_checks),
        "trace_records_passed": sum(item["status"] == "pass" for item in record_checks),
        "unique_provenance_sets": len(provenance_values),
    }
    print(json.dumps(report, indent=2))
    (output_dir / "trace_audit_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if failures or missing or extra or invalid or len(records) != len(available) or not provenance_values:
        raise SystemExit("formal trace gate failed")


if __name__ == "__main__":
    main()
