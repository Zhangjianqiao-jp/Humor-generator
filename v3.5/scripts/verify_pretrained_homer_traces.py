#!/usr/bin/env python3
"""Fail-closed validator for the adapter-free public-release Planner traces."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import load_trace, read_jsonl, plan_from_record
from humor_generator_v35.homer.contracts import parse_associations, parse_conflicts


MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DATA_VERSION = "homer_pretrained_7b_public_release_362"
DATASET = ROOT / "data/processed/homer_pretrained_7b_public_release_362"
DEFAULT_CACHE = ROOT / "data/cache/homer_pretrained_7b_planner_traces"
PROMPTS = ROOT / "src/humor_generator_v35/homer/official_prompts.py"
MODEL_MANIFEST = ROOT / "manifests/local_qwen2_5_vl_7b.json"
CHANNELS = {"conflict", "global", "local"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_input_rows(dataset: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(dataset / "trace_inputs.jsonl")
    if len(rows) != 362:
        raise RuntimeError(f"expected 362 sealed trace inputs, found {len(rows)}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        cluster = str(row.get("cluster_id", ""))
        if not cluster or cluster in result:
            raise RuntimeError(f"duplicate/empty input cluster: {cluster!r}")
        result[cluster] = row
    return result


def verify(cache: Path, dataset: Path = DATASET) -> dict[str, Any]:
    expected = read_input_rows(dataset)
    input_manifest = dataset / "trace_inputs.jsonl"
    input_hash = sha256(input_manifest)
    prompt_hash = sha256(PROMPTS)
    model_manifest_hash = sha256(MODEL_MANIFEST)
    index_path = cache / "index.jsonl"
    failures_path = cache / "failures.json"
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    if not index_path.is_file():
        errors.append(f"missing trace index: {index_path}")
    else:
        records = read_jsonl(index_path)
    failures: list[Any] = []
    if failures_path.is_file():
        try:
            failures = json.loads(failures_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"failures.json is invalid JSON: {exc}")
    elif records:
        errors.append("trace cache is missing failures.json")
    if not isinstance(failures, list):
        errors.append("failures.json must contain a list")
        failures = []

    seen: set[str] = set()
    passed = 0
    record_errors: list[dict[str, str]] = []
    provenance_sets: set[tuple[str, ...]] = set()
    for record in records:
        cluster = str(record.get("cluster_id", ""))
        try:
            if cluster in seen:
                raise ValueError("duplicate cluster record")
            seen.add(cluster)
            if cluster not in expected:
                raise ValueError("cluster is not in sealed public-release trace inputs")
            row = expected[cluster]
            if record.get("schema_version") != 4:
                raise ValueError("trace schema_version must be 4")
            if record.get("data_version") != DATA_VERSION:
                raise ValueError("trace data_version mismatch")
            if int(record.get("contest_number")) != int(row["contest_number"]):
                raise ValueError("contest_number mismatch")
            for field in ("image", "image_sha256", "split", "standard_description"):
                if str(record.get(field)) != str(row.get(field)):
                    raise ValueError(f"input field mismatch: {field}")
            if record.get("planner_model") != MODEL:
                raise ValueError("Planner model mismatch")
            if record.get("planner_revision") != REVISION:
                raise ValueError("Planner revision mismatch")
            if record.get("planner_adapter") is not None:
                raise ValueError("pretrained trace must have planner_adapter=null")
            if record.get("prompt_track") != "public_code_exact_text":
                raise ValueError("trace does not use public_code_exact_text prompt track")
            if set(record.get("trace_channels", [])) != CHANNELS:
                raise ValueError("trace channel set mismatch")
            if record.get("trace_stage_order") != [
                "extractor.conflict", "imaginator.global", "imaginator.local"
            ]:
                raise ValueError("trace stage order mismatch")
            provenance = record.get("provenance", {})
            expected_provenance = {
                "git_commit", "trace_input_manifest_sha256",
                "homer_official_prompts_sha256", "model_manifest_sha256", "data_version",
            }
            if set(provenance) != expected_provenance:
                raise ValueError("trace provenance schema mismatch")
            if not re.fullmatch(r"[0-9a-f]{40}", str(provenance["git_commit"])):
                raise ValueError("invalid trace Git commit")
            if provenance["trace_input_manifest_sha256"] != input_hash:
                raise ValueError("trace input manifest hash mismatch")
            if provenance["homer_official_prompts_sha256"] != prompt_hash:
                raise ValueError("official prompt hash mismatch")
            if provenance["model_manifest_sha256"] != model_manifest_hash:
                raise ValueError("model manifest hash mismatch")
            if provenance["data_version"] != DATA_VERSION:
                raise ValueError("provenance data version mismatch")
            provenance_sets.add(tuple(str(provenance[key]) for key in sorted(expected_provenance)))

            plan = plan_from_record(record["plan"])
            if len(plan.conflicts) < 2:
                raise ValueError("plan contains fewer than two conflict pairs")
            if not plan.local_chains or not plan.global_chains:
                raise ValueError("plan is missing local/global association chains")
            # The parsed plan is not sufficient to replay the remaining
            # public-code stages: summary, retrieval and selection consume the
            # original Planner response strings.  Require those exact bytes
            # in every current trace so a bridge run cannot silently fall back
            # to the historical prompt/dataclass approximation.
            planner_outputs = record.get("planner_outputs")
            if not isinstance(planner_outputs, dict) or set(planner_outputs) != CHANNELS:
                raise ValueError("planner_outputs must contain raw conflict/global/local responses")
            if any(not isinstance(value, str) or not value.strip() for value in planner_outputs.values()):
                raise ValueError("planner_outputs contains an empty/non-string response")
            if record.get("planner_outputs_sha256") != json_sha256(planner_outputs):
                raise ValueError("planner_outputs_sha256 does not match raw Planner responses")
            parsed_conflicts = parse_conflicts(planner_outputs["conflict"])
            parse_associations(planner_outputs["global"], view="global")
            parse_associations(planner_outputs["local"], view="local")
            if tuple(item.render() for item in parsed_conflicts) != tuple(
                item.render() for item in plan.conflicts
            ):
                raise ValueError("parsed conflict plan does not match the raw conflict response")
            trace_value = str(record.get("trace_path", ""))
            trace_path = (ROOT / trace_value).resolve()
            if ROOT not in trace_path.parents:
                raise ValueError("trace path escapes v3.5 root")
            trace_hash = str(record.get("trace_sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", trace_hash):
                raise ValueError("invalid trace sha256")
            loaded = load_trace(trace_path, expected_sha256=trace_hash)
            if set(loaded) != CHANNELS:
                raise ValueError("loaded trace channel set mismatch")
            for channel, aligned in loaded.items():
                if aligned.token_ids.shape[0] != 1:
                    raise ValueError(f"{channel}: batch dimension must be 1")
                if aligned.token_ids.shape[1] < 1:
                    raise ValueError(f"{channel}: empty token sequence")
                if aligned.states.shape[:2] != aligned.token_ids.shape:
                    raise ValueError(f"{channel}: hidden-state/token shape mismatch")
                if aligned.states.shape[-1] != 3584:
                    raise ValueError(f"{channel}: expected hidden width 3584")
                if not torch.isfinite(aligned.states).all():
                    raise ValueError(f"{channel}: non-finite hidden state")
                if (aligned.token_ids < 0).any():
                    raise ValueError(f"{channel}: negative token ID")
                if not aligned.semantics.strip() or aligned.semantics in {
                    "unspecified", "state_used_to_predict_corresponding_token",
                }:
                    raise ValueError(f"{channel}: missing generated semantics")
            alignment = record.get("alignment", {})
            if set(alignment) != CHANNELS:
                raise ValueError("alignment evidence does not cover all channels")
            for channel, evidence in alignment.items():
                if evidence.get("communication_state_definition") != "teacher_forced_post_token":
                    raise ValueError(f"{channel}: unexpected communication state definition")
                replay = evidence.get("replay", {})
                if float(replay.get("mean_cosine", 0.0)) < 0.98:
                    raise ValueError(f"{channel}: cached replay mean cosine below 0.98")
                if float(replay.get("min_cosine", 0.0)) < 0.90:
                    raise ValueError(f"{channel}: cached replay min cosine below 0.90")
            passed += 1
        except Exception as exc:
            record_errors.append({"cluster_id": cluster, "error": str(exc)})

    missing = sorted(set(expected) - seen)
    extra = sorted(seen - set(expected))
    if missing:
        errors.append(f"missing {len(missing)} traces; first={missing[:5]}")
    if extra:
        errors.append(f"extra {len(extra)} traces; first={extra[:5]}")
    if failures:
        errors.append(f"cache reports {len(failures)} generation failures")
    errors.extend(f"{item['cluster_id']}: {item['error']}" for item in record_errors)
    if len(records) != len(seen):
        errors.append("trace index contains duplicate records")
    report = {
        "status": "pass" if not errors and len(records) == 362 and passed == 362 else "fail",
        "data_version": DATA_VERSION,
        "model": MODEL,
        "revision": REVISION,
        "expected_records": len(expected),
        "records": len(records),
        "passed_records": passed,
        "missing_records": len(missing),
        "extra_records": len(extra),
        "failure_records": len(failures),
        "invalid_records": len(record_errors),
        "provenance_sets": len(provenance_sets),
        "trace_input_manifest": str(input_manifest),
        "trace_input_manifest_sha256": input_hash,
        "cache": str(cache),
        "errors": errors,
    }
    return report


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.cache.resolve(), args.dataset.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
