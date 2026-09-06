#!/usr/bin/env python3
"""Build hidden-state traces for the frozen, adapter-free 7B Planner route.

This entry point is intentionally independent from ``cache_planner_traces.py``.
The latter is the historical SFT-adapter cache and must never be reused for
the public-release pretrained route.  Here the input population is the sealed
362-contest public release, the Planner is loaded with ``adapter=None``, and
the pinned public HOMER prompt builders are used for the three bridge channels
(conflict, global imagination and local imagination).

The script is resumable, but fail-closed: a malformed channel, a generation
state/token mismatch, a stale input manifest, or a dirty tracked v3.5 tree
aborts rather than creating a partially claimable formal cache.  Retries
repeat the original prompt with a new deterministic seed; no semantic repair
or manual completion is performed in this trace route.
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import plan_to_record, read_jsonl, save_trace
from humor_generator_v35.homer.contracts import (
    parse_associations,
    parse_conflicts,
    validate_plan,
)
from humor_generator_v35.homer.official_prompts import (
    conflict_messages,
    global_imagination_messages,
    local_imagination_messages,
)
from humor_generator_v35.latent.state_capture import AlignedMessageStates
from humor_generator_v35.qwen_backend import QwenBackend


MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DATA_VERSION = "homer_pretrained_7b_public_release_362"
DEFAULT_DATASET = ROOT / "data/processed/homer_pretrained_7b_public_release_362"
DEFAULT_OUTPUT = ROOT / "data/cache/homer_pretrained_7b_planner_traces"
CHANNELS = ("conflict", "global", "local")
PROMPT_SOURCE = ROOT / "src/humor_generator_v35/homer/official_prompts.py"
MODEL_MANIFEST = ROOT / "manifests/local_qwen2_5_vl_7b.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit() -> str:
    """Require a committed v3.5 tree for formal trace provenance."""
    repo = ROOT.parent
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", "v3.5"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "formal pretrained traces require a committed, clean v3.5 tree; "
            "commit intended changes before GPU generation:\n" + status
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("could not resolve a full Git commit")
    return commit


def alignment_record(report: Any) -> dict[str, Any]:
    """Convert dataclass alignment evidence into JSON-safe primitives."""
    if isinstance(report, dict):
        value = report
    else:
        value = asdict(report) if is_dataclass(report) else vars(report)

    def normalize(item: Any) -> Any:
        if is_dataclass(item):
            return {key: normalize(value) for key, value in asdict(item).items()}
        if isinstance(item, dict):
            return {str(key): normalize(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        return item

    return normalize(value)


def load_inputs(dataset: Path) -> list[dict[str, Any]]:
    """Read the sealed trace inputs and verify image/description identity."""
    input_path = dataset / "trace_inputs.jsonl"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    rows = read_jsonl(input_path)
    if len(rows) != 362:
        raise RuntimeError(f"public-release trace input must contain 362 rows; got {len(rows)}")
    ids = [int(row["contest_number"]) for row in rows]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise RuntimeError("trace inputs must be sorted by unique contest_number")
    clusters = [str(row["cluster_id"]) for row in rows]
    if len(set(clusters)) != len(clusters):
        raise RuntimeError("trace inputs contain duplicate cluster_id values")
    for row in rows:
        required = {"cluster_id", "contest_number", "image", "image_sha256", "standard_description", "split"}
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"{row.get('cluster_id')}: missing input fields {missing}")
        image = Path(str(row["image"]))
        if not image.is_absolute():
            image = (ROOT / image).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"{row['cluster_id']}: image is missing: {image}")
        if sha256(image) != str(row["image_sha256"]):
            raise RuntimeError(f"{row['cluster_id']}: image_sha256 mismatch: {image}")
        if not str(row["standard_description"]).strip():
            raise RuntimeError(f"{row['cluster_id']}: empty standard_description")
    return rows


def channel_prompts(row: dict[str, Any], conflict_text: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Construct the public-code prompt for each captured Planner channel."""
    description = str(row["standard_description"])
    image = str((ROOT / row["image"]).resolve())
    if conflict_text is None:
        raise ValueError("global/local prompts require the generated conflict text")
    return {
        "conflict": conflict_messages(description),
        "global": global_imagination_messages(image, conflict_text, provider="qwen"),
        "local": local_imagination_messages(description, conflict_text),
    }


def _trace_path(output: Path, cluster_id: str) -> Path:
    return output / "states" / f"{cluster_id}.pt"


def _existing_records(index_path: Path, input_hash: str) -> dict[str, dict[str, Any]]:
    if not index_path.exists():
        return {}
    records = read_jsonl(index_path)
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        cluster = str(record.get("cluster_id", ""))
        if not cluster or cluster in result:
            raise RuntimeError(f"duplicate or empty cluster in existing trace index: {cluster!r}")
        provenance = record.get("provenance", {})
        if provenance.get("trace_input_manifest_sha256") != input_hash:
            raise RuntimeError(
                f"existing trace {cluster} was built from a different public-release input manifest"
            )
        if record.get("planner_model") != MODEL or record.get("planner_revision") != REVISION:
            raise RuntimeError(f"existing trace {cluster} has a different Planner identity")
        if record.get("planner_adapter") is not None:
            raise RuntimeError(f"existing trace {cluster} is not adapter-free")
        result[cluster] = record
    return result


def _generation_limits(channel: str) -> int:
    # Keep a bounded tensor footprint while leaving room for the official
    # prompt's short conflict/three-successor JSON outputs.
    return 384 if channel == "conflict" else 512


def generate_one(
    backend: QwenBackend,
    row: dict[str, Any],
    *,
    seed: int,
    temperature: float,
    top_p: float,
) -> tuple[dict[str, AlignedMessageStates], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Generate and validate one conflict/local/global trace bundle."""
    outputs: dict[str, str] = {}
    alignments: dict[str, Any] = {}

    conflict_text, conflict_states, conflict_alignment = backend.generate_and_verify_states(
        conflict_messages(str(row["standard_description"])),
        max_new_tokens=_generation_limits("conflict"),
        seed=seed,
        temperature=temperature,
        top_p=top_p,
    )
    parse_conflicts(conflict_text)
    outputs["conflict"] = conflict_text
    alignments["conflict"] = alignment_record(conflict_alignment)

    prompts = channel_prompts(row, conflict_text)
    global_text, global_states, global_alignment = backend.generate_and_verify_states(
        prompts["global"],
        max_new_tokens=_generation_limits("global"),
        seed=seed + 1,
        temperature=temperature,
        top_p=top_p,
    )
    parse_associations(global_text, view="global")
    outputs["global"] = global_text
    alignments["global"] = alignment_record(global_alignment)

    local_text, local_states, local_alignment = backend.generate_and_verify_states(
        prompts["local"],
        max_new_tokens=_generation_limits("local"),
        seed=seed + 2,
        temperature=temperature,
        top_p=top_p,
    )
    parse_associations(local_text, view="local")
    outputs["local"] = local_text
    alignments["local"] = alignment_record(local_alignment)

    plan = validate_plan(
        str(row["standard_description"]), conflict_text, local_text, global_text
    )
    states = {
        "conflict": AlignedMessageStates(
            conflict_states.token_ids, conflict_states.states, conflict_text
        ),
        "global": AlignedMessageStates(
            global_states.token_ids, global_states.states, global_text
        ),
        "local": AlignedMessageStates(local_states.token_ids, local_states.states, local_text),
    }
    return states, plan_to_record(plan), alignments, outputs


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--retry-round", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument("--cluster-ids", nargs="+")
    args = parser.parse_args()

    if args.attempts < 1 or args.retry_round < 0:
        raise ValueError("attempts must be positive and retry-round must be non-negative")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be non-negative and top-p must be in (0,1]")

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    manifest_path = dataset / "trace_inputs.jsonl"
    rows = load_inputs(dataset)
    if args.cluster_ids:
        requested = set(args.cluster_ids)
        unknown = sorted(requested - {str(row["cluster_id"]) for row in rows})
        if unknown:
            raise ValueError(f"unknown cluster IDs: {unknown}")
        rows = [row for row in rows if str(row["cluster_id"]) in requested]
    if args.max_clusters is not None:
        if args.max_clusters < 1:
            raise ValueError("max-clusters must be positive")
        rows = rows[: args.max_clusters]

    # The population verifier is the data gate.  This subprocess also checks
    # the allow-list/image/ranking/description hashes before model loading.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_homer_public_release_362.py")],
        cwd=ROOT,
        check=True,
    )
    commit = repository_commit()
    input_hash = sha256(manifest_path)
    prompt_hash = sha256(PROMPT_SOURCE)
    model_manifest_hash = sha256(MODEL_MANIFEST)
    provenance = {
        "git_commit": commit,
        "trace_input_manifest_sha256": input_hash,
        "homer_official_prompts_sha256": prompt_hash,
        "model_manifest_sha256": model_manifest_hash,
        "data_version": DATA_VERSION,
    }

    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.jsonl"
    existing = _existing_records(index_path, input_hash)
    if existing and any(str(row["cluster_id"]) not in existing for row in rows):
        # A resume is allowed, but all records in the file must belong to the
        # same sealed public release.  The final verifier enforces 362/362.
        pass

    backend = QwenBackend.load(
        MODEL,
        revision=REVISION,
        adapter=None,
        load_in_4bit=True,
        min_visual_tokens=256,
        max_visual_tokens=1280,
    )
    failures: list[dict[str, Any]] = []
    with index_path.open("a", encoding="utf-8") as index:
        for offset, row in enumerate(rows):
            cluster = str(row["cluster_id"])
            if cluster in existing:
                continue
            last_error = ""
            last_outputs: dict[str, str] = {}
            for attempt in range(args.attempts):
                seed = args.seed + args.retry_round * 10_000_000 + offset * args.attempts * 3 + attempt * 3
                try:
                    states, plan, alignments, outputs = generate_one(
                        backend,
                        row,
                        seed=seed,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    trace_path = _trace_path(output, cluster)
                    trace_hash = save_trace(trace_path, states)
                    record = {
                        "schema_version": 4,
                        "data_version": DATA_VERSION,
                        "cluster_id": cluster,
                        "contest_number": int(row["contest_number"]),
                        "dataset": row.get("dataset", "humor_in_ai"),
                        "split": row["split"],
                        "image": row["image"],
                        "image_sha256": row["image_sha256"],
                        "trace_path": str(trace_path.relative_to(ROOT)),
                        "trace_sha256": trace_hash,
                        "planner_model": MODEL,
                        "planner_revision": REVISION,
                        "planner_adapter": None,
                        "prompt_track": "public_code_exact_text",
                        "trace_channels": list(CHANNELS),
                        "trace_stage_order": [
                            "extractor.conflict",
                            "imaginator.global",
                            "imaginator.local",
                        ],
                        "standard_description": row["standard_description"],
                        "generation": {
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "attempt": attempt,
                            "retry_round": args.retry_round,
                            "max_new_tokens": {channel: _generation_limits(channel) for channel in CHANNELS},
                        },
                        "seed": seed,
                        "plan": plan,
                        "alignment": alignments,
                        "provenance": provenance,
                    }
                    index.write(json.dumps(record, ensure_ascii=False) + "\n")
                    index.flush()
                    existing[cluster] = record
                    if len(existing) % 10 == 0 or len(existing) == len(rows):
                        progress = {
                            "requested_in_this_run": len(rows),
                            "completed_total_in_index": len(existing),
                            "failed_in_this_run": len(failures),
                            "last_cluster": cluster,
                        }
                        (output / "progress.json").write_text(
                            json.dumps(progress, ensure_ascii=False, indent=2) + "\n"
                        )
                        print(json.dumps(progress, ensure_ascii=False), flush=True)
                    break
                except Exception as exc:  # bounded retries are recorded below
                    last_error = str(exc)
                    last_outputs = locals().get("outputs", {})
            else:
                failures.append({
                    "cluster_id": cluster,
                    "contest_number": int(row["contest_number"]),
                    "error": last_error,
                    "last_outputs": last_outputs,
                })

    (output / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n"
    )
    trace_manifest = {
        "schema_version": 1,
        "status": "pass"
        if not failures and len(read_jsonl(index_path)) == 362
        else "incomplete",
        "data_version": DATA_VERSION,
        "expected_records": 362,
        "records_in_index": len(read_jsonl(index_path)) if index_path.exists() else 0,
        "failure_records": len(failures),
        "model": MODEL,
        "revision": REVISION,
        "adapter": None,
        "prompt_track": "public_code_exact_text",
        "trace_input_manifest": str(manifest_path.relative_to(ROOT)),
        "trace_input_manifest_sha256": input_hash,
        "homer_official_prompts_sha256": prompt_hash,
        "model_manifest_sha256": model_manifest_hash,
        "git_commit": commit,
        "generation_temperature": args.temperature,
        "generation_top_p": args.top_p,
        "retry_round": args.retry_round,
        "channels": list(CHANNELS),
    }
    (output / "manifest.json").write_text(
        json.dumps(trace_manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "status": "pass" if not failures else "incomplete",
        "requested_in_this_run": len(rows),
        "records_in_index": trace_manifest["records_in_index"],
        "failure_records": len(failures),
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
