#!/usr/bin/env python3
"""Cache the remaining HOMER stages for the adapter-free 362 route.

``cache_pretrained_homer_traces.py`` stores hidden states for the three
Planner channels.  This script deliberately runs afterwards and replays the
official summary, retrieval and Generator-selection stages from the *raw*
cached Planner responses.  It does not regenerate conflict or imagination
text, so the latent bridge and the text baseline consume the same Planner
sample.

The resulting context index is a prerequisite for current bridge training. It
is never created from the historical ``latent_bridge_v35`` rows and it is not
the caption evaluation itself (the caption stage remains a separate run).
"""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import read_jsonl, plan_from_record
from humor_generator_v35.homer.contracts import parse_associations, parse_conflicts
from humor_generator_v35.homer.official_prompts import (
    select_conflict_messages,
    select_entity_messages,
    summary_messages,
    extract_marked,
)
from humor_generator_v35.homer.public_code_pipeline import _entity_names, _json_object, _paths_for_entity
from humor_generator_v35.homer.retrieval import OfficialCodeRetrieval
from humor_generator_v35.homer.repair import (
    REPAIR_POLICY_VERSION,
    assert_lossless_summary_repair,
    assert_reference_only_entity_repair,
    validator_feedback_messages,
)
from humor_generator_v35.qwen_backend import QwenBackend
from humor_generator_v35.training.public_bridge import (
    PUBLIC_BRIDGE_PROMPT_TRACK,
    load_context_index,
    selected_path_text,
)


MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DATA_VERSION = "homer_pretrained_7b_public_release_362"
DEFAULT_DATASET = ROOT / "data/processed/homer_pretrained_7b_public_release_362"
DEFAULT_TRACE_CACHE = ROOT / "data/cache/homer_pretrained_7b_planner_traces"
DEFAULT_OUTPUT = ROOT / "data/cache/homer_pretrained_7b_homer_context"
PROMPT_SOURCE = ROOT / "src/humor_generator_v35/homer/official_prompts.py"
MODEL_MANIFEST = ROOT / "manifests/local_qwen2_5_vl_7b.json"
POPULATION_MANIFEST = ROOT / "manifests/homer_population_public_release_362.json"
REPAIR_MANIFEST_VERSION = "pretrained-homer-context-repair-20260910-r5"
SUPPORTED_REPAIR_MANIFEST_VERSIONS = {
    "pretrained-homer-context-repair-20260910",
    "pretrained-homer-context-repair-20260910-r2",
    "pretrained-homer-context-repair-20260910-r3",
    "pretrained-homer-context-repair-20260910-r4",
    REPAIR_MANIFEST_VERSION,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def json_sha256(value: Any) -> str:
    """Hash structured raw responses independent of dictionary insertion order."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_repair_manifest(
    path: Path, *, source_shard: int | None = None
) -> tuple[dict[str, Any], str]:
    """Load and independently verify the bounded residual-failure manifest.

    The repair job must never silently broaden its scope when a shard failure
    file has changed.  The tracked manifest pins the bounded residual cluster
    IDs and the exact failure-file bytes that justified repairing them.  The
    number of targets is intentionally allowed to shrink after a successful
    retry; requiring the historical count would force us to re-introduce
    already repaired records merely to satisfy a stale guard.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("repair_manifest_version") not in SUPPORTED_REPAIR_MANIFEST_VERSIONS:
        raise ValueError("repair manifest has an unknown version")
    if payload.get("data_version") != DATA_VERSION:
        raise ValueError("repair manifest has the wrong data version")
    if payload.get("model") != MODEL or payload.get("revision") != REVISION:
        raise ValueError("repair manifest Planner identity mismatch")
    if payload.get("adapter") is not None:
        raise ValueError("repair manifest must declare adapter=null")
    if payload.get("policy") != REPAIR_POLICY_VERSION:
        raise ValueError("repair manifest uses an unsupported repair policy")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not 1 <= len(targets) <= 362:
        raise ValueError("repair manifest must contain between 1 and 362 targets")
    ids: list[str] = []
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError("repair manifest target is not an object")
        cluster = str(target.get("cluster_id", "")).strip()
        if not cluster or cluster in ids:
            raise ValueError(f"duplicate/empty repair target: {cluster!r}")
        if not isinstance(target.get("source_shard"), int) or target["source_shard"] not in (0, 1, 2):
            raise ValueError(f"invalid source shard for {cluster}")
        if not isinstance(target.get("previous_error"), str) or not target["previous_error"].strip():
            raise ValueError(f"missing previous error for {cluster}")
        ids.append(cluster)
    failures = payload.get("source_failure_manifests")
    if not isinstance(failures, list) or not failures:
        raise ValueError("repair manifest has no source failure manifests")
    selected_targets = {
        str(target["cluster_id"])
        for target in targets
        if source_shard is None or target.get("source_shard") == source_shard
    }
    if source_shard is not None and not selected_targets:
        raise ValueError(f"repair manifest has no targets for shard {source_shard}")
    seen: dict[str, dict[str, Any]] = {}
    for entry in failures:
        if not isinstance(entry, dict):
            raise ValueError("source failure manifest is not an object")
        entry_path = str(entry.get("path", ""))
        if source_shard is not None and f"_shard_{source_shard}/" not in entry_path:
            continue
        source = ROOT / entry_path
        expected_hash = str(entry.get("sha256", ""))
        if not source.is_file() or len(expected_hash) != 64:
            raise ValueError(f"source failure manifest is unavailable: {source}")
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise ValueError(f"source failure manifest hash mismatch: {source}")
        try:
            rows = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"source failure manifest is invalid JSON: {source}") from exc
        if not isinstance(rows, list):
            raise ValueError(f"source failure manifest is not a list: {source}")
        for row in rows:
            cluster = str(row.get("cluster_id", ""))
            if cluster:
                seen[cluster] = row
    if set(seen) != selected_targets:
        raise ValueError(
            "source failure manifests and repair targets differ: "
            f"source_only={sorted(set(seen) - selected_targets)}, "
            f"target_only={sorted(selected_targets - set(seen))}"
        )
    for target in targets:
        cluster = str(target["cluster_id"])
        if cluster not in selected_targets:
            continue
        row = seen.get(cluster)
        if row is None or str(row.get("error", "")) != str(target["previous_error"]):
            raise ValueError(f"repair manifest no longer matches recorded failure: {cluster}")
    manifest_hash = sha256(path)
    return payload, manifest_hash


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _official_retriever() -> OfficialCodeRetrieval:
    manifest = json.loads((ROOT / "manifests/homer_official_assets.json").read_text())
    corpus = ROOT / manifest["joke_corpus"]["local_path"]
    if sha256(corpus) != manifest["joke_corpus"]["sha256"]:
        raise RuntimeError("HOMER joke corpus hash mismatch")
    import csv
    import nltk
    nltk.data.path.insert(0, str(ROOT / "artifacts/nltk_data"))
    with corpus.open(encoding="utf-8", newline="") as handle:
        jokes = [row["Joke"] for row in csv.DictReader(handle)]
    return OfficialCodeRetrieval(jokes)


def _load_trace_records(trace_index: Path) -> dict[str, dict[str, Any]]:
    if not trace_index.is_file():
        raise FileNotFoundError(trace_index)
    records = read_jsonl(trace_index)
    manifest_path = trace_index.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"trace manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or manifest.get("records_in_index") != 362:
        raise ValueError("current context route requires a complete 362-record trace manifest")
    if manifest.get("data_version") != DATA_VERSION:
        raise ValueError("trace manifest has the wrong data version")
    if manifest.get("model") != MODEL or manifest.get("revision") != REVISION:
        raise ValueError("trace manifest has the wrong Planner identity")
    if manifest.get("adapter") is not None:
        raise ValueError("trace manifest is not adapter-free")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        cluster = str(record.get("cluster_id", ""))
        if not cluster or cluster in result:
            raise ValueError(f"duplicate/empty trace cluster: {cluster!r}")
        if record.get("data_version") != DATA_VERSION:
            raise ValueError(f"trace {cluster} has the wrong data version")
        if record.get("planner_model") != MODEL or record.get("planner_revision") != REVISION:
            raise ValueError(f"trace {cluster} has the wrong Planner identity")
        if record.get("planner_adapter") is not None:
            raise ValueError(f"trace {cluster} is not adapter-free")
        if not str(record.get("trace_path", "")).strip() or len(str(record.get("trace_sha256", ""))) != 64:
            raise ValueError(f"trace {cluster} has incomplete hidden-state provenance")
        raw = record.get("planner_outputs")
        if not isinstance(raw, dict) or set(raw) != {"conflict", "global", "local"}:
            raise ValueError(
                f"trace {cluster} has no complete raw conflict/global/local Planner outputs"
            )
        if any(not isinstance(value, str) or not value.strip() for value in raw.values()):
            raise ValueError(f"trace {cluster} has an empty raw Planner output")
        if record.get("planner_outputs_sha256") != json_sha256(raw):
            raise ValueError(f"trace {cluster} raw Planner output hash mismatch")
        # Validate the same schemas used by trace verification before any
        # post-trace model call is made.
        parse_conflicts(raw["conflict"])
        parse_associations(raw["global"], view="global")
        parse_associations(raw["local"], view="local")
        result[cluster] = record
    if len(result) != 362:
        raise ValueError(f"current context route requires 362 traces, got {len(result)}")
    return result


def _load_rows(dataset: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(dataset / "trace_inputs.jsonl")
    if len(rows) != 362:
        raise ValueError(f"current context route requires 362 trace inputs, got {len(rows)}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        cluster = str(row.get("cluster_id", ""))
        if not cluster or cluster in result:
            raise ValueError(f"duplicate/empty trace input cluster: {cluster!r}")
        result[cluster] = row
    return result


def _call(
    backend: Any,
    stage: str,
    messages: list[dict[str, Any]],
    *,
    seed: int,
    temperature: float,
    top_p: float,
    call_log: list[dict[str, Any]],
) -> str:
    # QwenBackend fixes top_p=1.0 internally (the same setting used by the
    # Planner trace job); keep the requested value in the provenance ledger
    # without passing an unsupported keyword to its public ``generate`` API.
    requested_top_p = top_p
    value = backend.generate(
        messages,
        temperature=temperature,
        max_new_tokens=1000,
        seed=seed,
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{stage} returned an empty response")
    call_log.append({
        "stage": stage,
        "seed": seed,
        "temperature": temperature,
        "top_p": requested_top_p,
        "max_new_tokens": 1000,
    })
    return value.strip()


def build_context_for_record(
    backend: Any,
    row: Mapping[str, Any],
    trace: Mapping[str, Any],
    retriever: OfficialCodeRetrieval | Any,
    *,
    seed: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    enable_validator_repair: bool = False,
    repair_attempts: int = 1,
    repair_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Replay summary/retrieval/selection for one cached Planner trace.

    This pure-ish function is intentionally usable with a fake backend in
    unit tests; the production script supplies the frozen Qwen backend.
    """
    cluster = str(row["cluster_id"])
    raw_outputs: dict[str, str] = {}
    raw_output_records: list[dict[str, Any]] = []
    repair_records: list[dict[str, Any]] = []
    try:
        if cluster != str(trace["cluster_id"]):
            raise ValueError("row/trace cluster mismatch")
        for field in ("contest_number", "image", "image_sha256", "split", "standard_description"):
            if str(row.get(field, "")) != str(trace.get(field, "")):
                raise ValueError(f"{cluster}: row/trace mismatch for {field}")
        if repair_attempts < 1:
            raise ValueError("repair_attempts must be positive")
        description = str(row["standard_description"]).strip()
        raw = trace["planner_outputs"]
        conflict = str(raw["conflict"])
        global_payload = str(raw["global"])
        local_payload = str(raw["local"])
        plan = plan_from_record(trace["plan"])
        if tuple(item.render() for item in parse_conflicts(conflict)) != tuple(
            item.render() for item in plan.conflicts
        ):
            raise ValueError(f"{cluster}: raw conflict and parsed trace plan disagree")
        call_log: list[dict[str, Any]] = []
        base_seed = int(seed) * 1000

        def recorded_call(stage: str, messages: list[dict[str, Any]], call_seed: int) -> str:
            value = _call(
                backend,
                stage,
                messages,
                seed=call_seed,
                temperature=temperature,
                top_p=top_p,
                call_log=call_log,
            )
            raw_outputs[stage] = value
            raw_output_records.append({
                "stage": stage,
                "seed": call_seed,
                "output": value,
                "output_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            })
            return value

        def repair_call(
            *,
            channel: str,
            stage: str,
            messages: list[dict[str, Any]],
            invalid_output: str,
            validation_error: str,
            repair_seed_base: int,
            available: list[str] | None = None,
        ) -> str:
            if not enable_validator_repair:
                raise ValueError(validation_error)
            last_error = validation_error
            for repair_index in range(repair_attempts):
                repair_seed = repair_seed_base + repair_index
                try:
                    repaired = recorded_call(
                        f"{stage}.validator_feedback_repair",
                        validator_feedback_messages(
                            messages,
                            invalid_output=invalid_output,
                            validation_error=validation_error,
                            channel=channel,
                        ),
                        repair_seed,
                    )
                    if channel == "summary":
                        assert_lossless_summary_repair(invalid_output, repaired)
                    elif channel == "entities":
                        assert_reference_only_entity_repair(
                            invalid_output, repaired, available=available or []
                        )
                    else:  # pragma: no cover - guarded by validator_feedback_messages
                        raise ValueError(f"unsupported context repair channel: {channel}")
                    repair_records.append({
                        "policy_version": REPAIR_POLICY_VERSION,
                        "channel": channel,
                        "stage": stage,
                        "original_output": invalid_output,
                        "original_output_sha256": hashlib.sha256(
                            invalid_output.encode("utf-8")
                        ).hexdigest(),
                        "validation_error": validation_error,
                        "repair_output": repaired,
                        "repair_output_sha256": hashlib.sha256(
                            repaired.encode("utf-8")
                        ).hexdigest(),
                        "repair_seed": repair_seed,
                        "repair_attempt": repair_index,
                    })
                    return repaired
                except Exception as exc:
                    last_error = str(exc)
            raise ValueError(f"{cluster}: {stage} validator-feedback repair failed: {last_error}")

        summary_prompt = summary_messages(description, conflict, global_payload, local_payload)
        summary_raw = recorded_call("imaginator.summary", summary_prompt, base_seed + 1)
        try:
            summary = _json_object(summary_raw, stage="summary imagination")
            retrieved = dict(retriever.retrieve(summary, description=description, conflicts=conflict))
            if not retrieved:
                raise ValueError(f"{cluster}: retrieval produced an empty imagination graph")
        except Exception as exc:
            summary_raw = repair_call(
                channel="summary",
                stage="imaginator.summary",
                messages=summary_prompt,
                invalid_output=summary_raw,
                validation_error=str(exc),
                repair_seed_base=base_seed + 101,
            )
            summary = _json_object(summary_raw, stage="repaired summary imagination")
            retrieved = dict(retriever.retrieve(summary, description=description, conflicts=conflict))
            if not retrieved:
                raise ValueError(f"{cluster}: repaired retrieval produced an empty imagination graph")

        selected_raw = recorded_call(
            "generator.select_conflict",
            select_conflict_messages(description, conflict),
            base_seed + 2,
        )
        selected_conflict = extract_marked(selected_raw, "##Conflict Scripts")
        if not selected_conflict:
            raise ValueError(f"{cluster}: selected conflict is empty")
        entity_prompt = select_entity_messages(selected_conflict, list(retrieved))
        entities_raw = recorded_call("generator.select_entities", entity_prompt, base_seed + 3)
        try:
            selected_entities = _entity_names(entities_raw, available=list(retrieved))
        except Exception as exc:
            entities_raw = repair_call(
                channel="entities",
                stage="generator.select_entities",
                messages=entity_prompt,
                invalid_output=entities_raw,
                validation_error=str(exc),
                repair_seed_base=base_seed + 103,
                available=list(retrieved),
            )
            selected_entities = _entity_names(entities_raw, available=list(retrieved))
        import random
        rng = random.Random(int(seed))
        selected_paths: dict[str, list[str]] = {}
        for entity in selected_entities:
            paths = _paths_for_entity(entity, retrieved[entity], successors=3)
            selected_paths[entity] = list(rng.choice(paths))
        context = {
            "schema_version": 1,
            "data_version": DATA_VERSION,
            "cluster_id": cluster,
            "contest_number": int(row["contest_number"]),
            "dataset": row.get("dataset", "humor_in_ai"),
            "split": row["split"],
            "image": row["image"],
            "image_sha256": row["image_sha256"],
            "standard_description": description,
            "summary_imagination": summary,
            "summary_raw": summary_raw,
            "retrieved_imagination": retrieved,
            "selected_conflict": selected_conflict,
            "selection_raw": {
                "conflict": selected_raw,
                "entities": entities_raw,
            },
            "selected_entities": list(selected_entities),
            "selected_paths": selected_paths,
            "free_association_text": "\n".join(
                f"The free-association chain of {entity} is {' -> '.join(selected_paths[entity])}"
                for entity in selected_entities
            ),
            "trace_path": str(trace["trace_path"]),
            "trace_sha256": str(trace["trace_sha256"]),
            "planner_outputs_sha256": json_sha256(raw),
            "call_log": call_log,
            "raw_output_records": raw_output_records,
            "context_prompt_track": PUBLIC_BRIDGE_PROMPT_TRACK,
            "selection_policy": "official_public_code_selection_plus_seeded_dfs_path",
            "repair_records": repair_records,
            "repair_manifest_sha256": repair_manifest_hash,
            "provenance": {},
        }
        expected_paths = selected_path_text(
            SimpleNamespace(
                selected_entities=tuple(selected_entities),
                selected_paths={key: tuple(value) for key, value in selected_paths.items()},
            )
        )
        if context["free_association_text"] != expected_paths:
            raise AssertionError(f"{cluster}: selected path serialization mismatch")
        return context
    except Exception as exc:
        # Preserve every response observed before a residual failure so the
        # next bounded retry can audit it instead of guessing what was sent.
        setattr(exc, "raw_outputs", dict(raw_outputs))
        setattr(exc, "raw_output_records", list(raw_output_records))
        setattr(exc, "repair_records", list(repair_records))
        raise


def _existing(index_path: Path, *, trace_hash: str, input_hash: str) -> dict[str, dict[str, Any]]:
    if not index_path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    # Validate the whole prior file before allowing resume.  Otherwise a
    # malformed old record could be skipped and silently promoted to a
    # complete current context manifest.
    load_context_index(index_path)
    for item in read_jsonl(index_path):
        cluster = str(item.get("cluster_id", ""))
        if not cluster or cluster in result:
            raise ValueError(f"duplicate/empty context cluster: {cluster!r}")
        provenance = item.get("provenance", {})
        if provenance.get("trace_index_sha256") != trace_hash:
            raise ValueError(f"existing context {cluster} uses another trace index")
        if provenance.get("trace_input_manifest_sha256") != input_hash:
            raise ValueError(f"existing context {cluster} uses another input manifest")
        if item.get("data_version") != DATA_VERSION:
            raise ValueError(f"existing context {cluster} has the wrong data version")
        if item.get("context_prompt_track") != PUBLIC_BRIDGE_PROMPT_TRACK:
            raise ValueError(f"existing context {cluster} has the wrong prompt track")
        if provenance.get("homer_official_prompts_sha256") != sha256(PROMPT_SOURCE):
            raise ValueError(f"existing context {cluster} uses another official prompt source")
        if provenance.get("model_manifest_sha256") != sha256(MODEL_MANIFEST):
            raise ValueError(f"existing context {cluster} uses another model manifest")
        result[cluster] = item
    return result


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trace-index", type=Path, default=DEFAULT_TRACE_CACHE / "index.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--retry-round", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument("--cluster-ids", nargs="+")
    parser.add_argument(
        "--repair-shard", type=int, choices=(0, 1, 2),
        help="restrict the tracked repair manifest to one source shard",
    )
    parser.add_argument(
        "--enable-validator-repair",
        action="store_true",
        help="append a lossless validator-feedback turn for summary/entity errors",
    )
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument(
        "--repair-manifest", type=Path,
        help="tracked manifest pinning the bounded residual failures being repaired",
    )
    args = parser.parse_args()
    if (
        args.attempts < 1
        or args.retry_round < 0
        or args.repair_attempts < 1
        or args.max_clusters is not None and args.max_clusters < 1
    ):
        raise ValueError("attempts and max-clusters must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be non-negative and top-p must be in (0,1]")
    repair_manifest: dict[str, Any] | None = None
    repair_manifest_hash: str | None = None
    if args.enable_validator_repair:
        if args.repair_manifest is None:
            raise ValueError("--enable-validator-repair requires --repair-manifest")
        repair_manifest, repair_manifest_hash = _load_repair_manifest(
            args.repair_manifest.resolve(), source_shard=args.repair_shard
        )
    elif args.repair_manifest is not None:
        raise ValueError("--repair-manifest is only valid with --enable-validator-repair")
    if repair_manifest is not None and args.max_clusters is not None:
        raise ValueError("--max-clusters is forbidden for a bounded repair manifest")
    dataset = args.dataset.resolve()
    trace_index = args.trace_index.resolve()
    output = args.output.resolve()
    if dataset == (ROOT / "data/processed/latent_bridge_v35").resolve():
        raise RuntimeError("historical latent_bridge_v35 data is forbidden for current HOMER context")
    rows = _load_rows(dataset)
    traces = _load_trace_records(trace_index)
    if set(rows) != set(traces):
        raise RuntimeError("current trace inputs and trace index have different cluster sets")
    # Keep retry/shard seeds tied to the sealed global trace-input order.
    # Without this map, filtering with --cluster-ids would renumber offsets
    # and produce a different seed schedule from the single-process route.
    all_rows = list(rows.values())
    row_offsets = {str(row["cluster_id"]): offset for offset, row in enumerate(all_rows)}
    selected_rows = all_rows
    repair_targets = (repair_manifest or {}).get("targets", [])
    if args.repair_shard is not None and not repair_manifest:
        raise ValueError("--repair-shard requires --repair-manifest")
    repair_ids = {
        str(target["cluster_id"])
        for target in repair_targets
        if args.repair_shard is None or target.get("source_shard") == args.repair_shard
    }
    if repair_manifest and not repair_ids:
        raise ValueError(f"repair manifest has no targets for shard {args.repair_shard}")
    if args.cluster_ids:
        requested = set(args.cluster_ids)
        unknown = sorted(requested - set(rows))
        if unknown:
            raise ValueError(f"unknown cluster IDs: {unknown}")
        if repair_ids and requested != repair_ids:
            raise ValueError(
                "--cluster-ids must exactly match the tracked repair manifest targets"
            )
        selected_rows = [row for row in selected_rows if str(row["cluster_id"]) in requested]
    elif repair_ids:
        selected_rows = [row for row in selected_rows if str(row["cluster_id"]) in repair_ids]
    if args.max_clusters is not None:
        selected_rows = selected_rows[: args.max_clusters]

    # The data and trace validators are deliberately run before model loading.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_homer_public_release_362.py")],
        cwd=ROOT,
        check=True,
    )
    dataset_manifest = dataset / "manifest.json"
    if not dataset_manifest.is_file():
        raise FileNotFoundError(dataset_manifest)
    input_manifest = dataset / "trace_inputs.jsonl"
    trace_manifest = trace_index.parent / "manifest.json"
    if not trace_manifest.is_file():
        raise FileNotFoundError(trace_manifest)
    commit = git_commit()
    input_hash = sha256(input_manifest)
    trace_hash = sha256(trace_index)
    trace_manifest_payload = json.loads(trace_manifest.read_text(encoding="utf-8"))
    if trace_manifest_payload.get("trace_input_manifest_sha256") != input_hash:
        raise RuntimeError("trace index was built from another trace input manifest")
    prompt_hash = sha256(PROMPT_SOURCE)
    model_hash = sha256(MODEL_MANIFEST)
    population_hash = sha256(POPULATION_MANIFEST)
    existing = _existing(output / "index.jsonl", trace_hash=trace_hash, input_hash=input_hash)
    backend = QwenBackend.load(
        MODEL,
        revision=REVISION,
        adapter=None,
        load_in_4bit=True,
        min_visual_tokens=256,
        max_visual_tokens=1280,
    )
    retriever = _official_retriever()
    failures: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as handle:
        for offset, row in enumerate(selected_rows):
            cluster = str(row["cluster_id"])
            if cluster in existing:
                continue
            last_error = ""
            last_exception: Exception | None = None
            for attempt in range(args.attempts):
                try:
                    seed = (
                        int(args.seed)
                        + int(args.retry_round) * 10_000_000
                        + row_offsets[cluster] * args.attempts
                        + attempt
                    )
                    context = build_context_for_record(
                        backend, row, traces[cluster], retriever,
                        seed=seed,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        enable_validator_repair=args.enable_validator_repair,
                        repair_attempts=args.repair_attempts,
                        repair_manifest_hash=repair_manifest_hash,
                    )
                    context["provenance"] = {
                        "git_commit": commit,
                        "population_manifest_sha256": population_hash,
                        "trace_index_sha256": trace_hash,
                        "trace_input_manifest_sha256": input_hash,
                        "trace_path": str(traces[cluster]["trace_path"]),
                        "trace_sha256": str(traces[cluster]["trace_sha256"]),
                        "planner_outputs_sha256": context["planner_outputs_sha256"],
                        "homer_official_prompts_sha256": prompt_hash,
                        "model_manifest_sha256": model_hash,
                        "data_version": DATA_VERSION,
                        "planner_trace_seed": traces[cluster].get("seed"),
                        "context_seed": seed,
                        "attempt": attempt,
                        "retry_round": args.retry_round,
                        "validator_repair_enabled": args.enable_validator_repair,
                        "validator_repair_policy": (
                            REPAIR_POLICY_VERSION if args.enable_validator_repair else None
                        ),
                        "repair_manifest": (
                            _provenance_path(args.repair_manifest)
                            if args.repair_manifest is not None
                            else None
                        ),
                        "repair_manifest_sha256": repair_manifest_hash,
                    }
                    handle.write(json.dumps(context, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    existing[cluster] = context
                    break
                except Exception as exc:
                    last_error = str(exc)
                    last_exception = exc
            else:
                failures.append({
                    "cluster_id": cluster,
                    "contest_number": int(row["contest_number"]),
                    "error": last_error,
                    "seed_base": (
                        int(args.seed)
                        + int(args.retry_round) * 10_000_000
                        + row_offsets[cluster] * args.attempts
                    ),
                    "retry_round": args.retry_round,
                    "raw_outputs": getattr(last_exception, "raw_outputs", {}),
                    "raw_output_records": getattr(last_exception, "raw_output_records", []),
                    "repair_records": getattr(last_exception, "repair_records", []),
                })
    (output / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    records = read_jsonl(index_path) if index_path.is_file() else []
    manifest = {
        "schema_version": 1,
        "status": "pass" if not failures and len(records) == 362 else "incomplete",
        "data_version": DATA_VERSION,
        "expected_records": 362,
        "records_in_index": len(records),
        "failure_records": len(failures),
        "model": MODEL,
        "revision": REVISION,
        "adapter": None,
        "prompt_track": PUBLIC_BRIDGE_PROMPT_TRACK,
        "stage_order": [
            "imaginator.summary",
            "retrieval.offline",
            "generator.select_conflict",
            "generator.select_entities",
            "generator.seeded_dfs_path",
        ],
        "dataset": str(dataset.relative_to(ROOT)),
        "dataset_manifest_sha256": sha256(dataset_manifest),
        "trace_index": str(trace_index.relative_to(ROOT)),
        "trace_index_sha256": trace_hash,
        "trace_input_manifest": str(input_manifest.relative_to(ROOT)),
        "trace_input_manifest_sha256": input_hash,
        "population_manifest_sha256": population_hash,
        "homer_official_prompts_sha256": prompt_hash,
        "model_manifest_sha256": model_hash,
        "git_commit": commit,
        "generation_temperature": args.temperature,
        "generation_top_p": args.top_p,
        "retry_round": args.retry_round,
        "validator_repair_enabled": args.enable_validator_repair,
        "validator_repair_policy": (
            REPAIR_POLICY_VERSION if args.enable_validator_repair else None
        ),
        "repair_manifest": (
            _provenance_path(args.repair_manifest)
            if args.repair_manifest is not None
            else None
        ),
        "repair_manifest_sha256": repair_manifest_hash,
        "selection_policy": "official_public_code_selection_plus_seeded_dfs_path",
        "caption_stage": "not run; this is a post-trace context cache",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "requested_in_this_run": len(selected_rows),
        "records_in_index": len(records),
        "failure_records": len(failures),
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
