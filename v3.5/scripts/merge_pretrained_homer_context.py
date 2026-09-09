#!/usr/bin/env python3
"""Strictly merge disjoint adapter-free HOMER context shards.

The context generator writes one JSONL record after each successful image, so
walltime-safe shard jobs can be resumed independently.  This merger is
fail-closed: it never overwrites an existing output, rejects duplicate or
foreign clusters, checks every context against the sealed trace index, and
records all shard inputs in the resulting manifest.
"""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import read_jsonl
from humor_generator_v35.training.public_bridge import (
    PUBLIC_BRIDGE_PROMPT_TRACK,
    load_context_index,
)


MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DATA_VERSION = "homer_pretrained_7b_public_release_362"
DATASET = ROOT / "data/processed/homer_pretrained_7b_public_release_362"
TRACE_INDEX = ROOT / "data/cache/homer_pretrained_7b_planner_traces/index.jsonl"
PROMPT_SOURCE = ROOT / "src/humor_generator_v35/homer/official_prompts.py"
MODEL_MANIFEST = ROOT / "manifests/local_qwen2_5_vl_7b.json"
POPULATION_MANIFEST = ROOT / "manifests/homer_population_public_release_362.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RuntimeError("could not resolve a full Git commit")
    return value


def _index_path(value: Path) -> Path:
    path = value / "index.jsonl" if value.is_dir() else value
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_records(path: Path) -> list[dict[str, Any]]:
    # The production loader validates the public context schema and path
    # serialization.  Keep the raw dictionaries for lossless re-emission.
    load_context_index(path)
    return read_jsonl(path)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing merged context: {output}")

    dataset = DATASET.resolve()
    trace_index = TRACE_INDEX.resolve()
    input_manifest = dataset / "trace_inputs.jsonl"
    dataset_manifest = dataset / "manifest.json"
    for required in (input_manifest, dataset_manifest, trace_index, PROMPT_SOURCE,
                     MODEL_MANIFEST, POPULATION_MANIFEST):
        if not required.is_file():
            raise FileNotFoundError(required)

    rows = read_jsonl(input_manifest)
    if len(rows) != 362:
        raise RuntimeError(f"sealed trace inputs must contain 362 rows, got {len(rows)}")
    expected = {str(row["cluster_id"]): row for row in rows}
    traces = {str(row["cluster_id"]): row for row in read_jsonl(trace_index)}
    if set(expected) != set(traces) or len(traces) != 362:
        raise RuntimeError("trace index and sealed inputs do not have the same 362 clusters")
    trace_hash = sha256(trace_index)
    input_hash = sha256(input_manifest)
    prompt_hash = sha256(PROMPT_SOURCE)
    model_hash = sha256(MODEL_MANIFEST)
    population_hash = sha256(POPULATION_MANIFEST)
    dataset_hash = sha256(dataset_manifest)

    merged: dict[str, dict[str, Any]] = {}
    shard_inputs: list[str] = []
    for raw_shard in args.shard:
        shard = raw_shard.resolve()
        index_path = _index_path(shard)
        shard_inputs.append(str(shard.relative_to(ROOT) if ROOT in shard.parents else shard))
        failures_path = index_path.parent / "failures.json"
        if failures_path.is_file():
            failures = json.loads(failures_path.read_text(encoding="utf-8"))
            if failures:
                raise RuntimeError(f"shard {shard} reports {len(failures)} failures")
        manifest_path = index_path.parent / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("data_version") != DATA_VERSION:
                raise RuntimeError(f"shard {shard} has the wrong data version")
            if manifest.get("model") != MODEL or manifest.get("revision") != REVISION:
                raise RuntimeError(f"shard {shard} has the wrong Planner identity")
            if manifest.get("adapter") is not None:
                raise RuntimeError(f"shard {shard} is not adapter-free")
            if manifest.get("trace_index_sha256") != trace_hash:
                raise RuntimeError(f"shard {shard} uses another trace index")

        records = _load_records(index_path)
        for record in records:
            cluster = str(record.get("cluster_id", ""))
            if cluster in merged:
                raise RuntimeError(f"duplicate context cluster across shards: {cluster}")
            if cluster not in expected:
                raise RuntimeError(f"foreign context cluster: {cluster}")
            row = expected[cluster]
            for field in ("schema_version", "data_version", "cluster_id", "contest_number",
                          "image", "image_sha256", "split", "standard_description",
                          "context_prompt_track", "trace_path", "trace_sha256"):
                if field == "schema_version" and record.get(field) != 1:
                    raise RuntimeError(f"{cluster}: context schema_version must be 1")
                if field == "data_version" and record.get(field) != DATA_VERSION:
                    raise RuntimeError(f"{cluster}: data_version mismatch")
                if field == "cluster_id" and str(record.get(field)) != cluster:
                    raise RuntimeError(f"{cluster}: cluster_id mismatch")
                if field == "contest_number" and int(record.get(field)) != int(row[field]):
                    raise RuntimeError(f"{cluster}: contest_number mismatch")
                if field in {"image", "image_sha256", "split", "standard_description"} \
                        and str(record.get(field)) != str(row[field]):
                    raise RuntimeError(f"{cluster}: {field} mismatch")
                if field == "context_prompt_track" and record.get(field) != PUBLIC_BRIDGE_PROMPT_TRACK:
                    raise RuntimeError(f"{cluster}: context prompt track mismatch")
            trace = traces[cluster]
            provenance = record.get("provenance", {})
            if provenance.get("trace_index_sha256") != trace_hash:
                raise RuntimeError(f"{cluster}: trace index hash mismatch")
            if provenance.get("trace_input_manifest_sha256") != input_hash:
                raise RuntimeError(f"{cluster}: trace input hash mismatch")
            if provenance.get("population_manifest_sha256") != population_hash:
                raise RuntimeError(f"{cluster}: population manifest hash mismatch")
            if provenance.get("homer_official_prompts_sha256") != prompt_hash:
                raise RuntimeError(f"{cluster}: prompt hash mismatch")
            if provenance.get("model_manifest_sha256") != model_hash:
                raise RuntimeError(f"{cluster}: model manifest hash mismatch")
            if provenance.get("data_version") != DATA_VERSION:
                raise RuntimeError(f"{cluster}: provenance data version mismatch")
            if provenance.get("trace_path") != trace.get("trace_path"):
                raise RuntimeError(f"{cluster}: trace path mismatch")
            if provenance.get("trace_sha256") != trace.get("trace_sha256"):
                raise RuntimeError(f"{cluster}: trace hash mismatch")
            if not isinstance(record.get("summary_raw"), str) or not record["summary_raw"].strip():
                raise RuntimeError(f"{cluster}: empty summary_raw")
            merged[cluster] = record

    missing = sorted(set(expected) - set(merged))
    if missing:
        raise RuntimeError(f"merged context is incomplete: {len(missing)} missing, first={missing[:5]}")
    if len(merged) != 362:
        raise RuntimeError(f"merged context expected 362 records, got {len(merged)}")

    output.mkdir(parents=True, exist_ok=False)
    index_path = output / "index.jsonl"
    ordered = [merged[str(row["cluster_id"])] for row in rows]
    with index_path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "data_version": DATA_VERSION,
        "expected_records": 362,
        "records_in_index": 362,
        "failure_records": 0,
        "model": MODEL,
        "revision": REVISION,
        "adapter": None,
        "prompt_track": PUBLIC_BRIDGE_PROMPT_TRACK,
        "stage_order": [
            "imaginator.summary", "retrieval.offline", "generator.select_conflict",
            "generator.select_entities", "generator.seeded_dfs_path",
        ],
        "dataset": str(dataset.relative_to(ROOT)),
        "dataset_manifest_sha256": dataset_hash,
        "trace_index": str(trace_index.relative_to(ROOT)),
        "trace_index_sha256": trace_hash,
        "trace_input_manifest": str(input_manifest.relative_to(ROOT)),
        "trace_input_manifest_sha256": input_hash,
        "population_manifest_sha256": population_hash,
        "homer_official_prompts_sha256": prompt_hash,
        "model_manifest_sha256": model_hash,
        "git_commit": git_commit(),
        "generation_temperature": 1.0,
        "generation_top_p": 1.0,
        "selection_policy": "official_public_code_selection_plus_seeded_dfs_path",
        "caption_stage": "not run; this is a post-trace context cache",
        "merge_inputs": shard_inputs,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "failures.json").write_text("[]\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "records": 362, "output": str(output)}))


if __name__ == "__main__":
    main()
