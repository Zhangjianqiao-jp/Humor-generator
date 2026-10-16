#!/usr/bin/env python3
"""Generate a provenance-complete current HOMER text/latent comparison.

This entry point is deliberately separate from the historical A5 generator.
It consumes the sealed adapter-free public-release population, the immutable
HOMER context cache, and a *completed* current cross-attention bridge.  Both
conditions use the same cached description/selected plan and the same seed
schedule; the latent condition removes only the textual plan block and adds
the bridge memory at the receiver decoder layers.

The output uses the Caption-judgement ``generation.schema.json`` contract and
can therefore be passed directly to ``caption-judge adapt``.  It does not
call an LLM judge and never writes a quality score.  The HOMER primary
five-candidate/five-trial schedule is the default; Group-of-10 extension runs
must be requested explicitly.
"""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import load_trace, read_jsonl
from humor_generator_v35.homer.official_prompts import parse_caption_blob
from humor_generator_v35.homer.caption_parsing import parse_caption_strict
from humor_generator_v35.training.selected_plan import (SELECTED_PLAN_POLICY, LEGACY_MEMORY_POLICY, LEGACY_TARGET_POLICY, encode_selected_plan, selected_plan_sha256)
from humor_generator_v35.latent.bridges import TypedLatentBridge
from humor_generator_v35.latent.cross_attention import ReceiverDrivenCrossAttentionBridge
from humor_generator_v35.qwen_backend import QwenBackend, model_device
from humor_generator_v35.training.formal_bridge import load_trace_index
from humor_generator_v35.training.public_bridge import (
    PUBLIC_BRIDGE_PROMPT_TRACK,
    load_context_index,
    public_latent_caption_messages,
    public_text_caption_messages,
    validate_row_context,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def generation_schedule(
    *,
    trials: int,
    candidates: int,
    base_seed: int,
    trial_stride: int = 100,
) -> list[dict[str, int]]:
    """Return an order-invariant, auditable HOMER candidate schedule."""
    if trials < 1 or candidates < 1:
        raise ValueError("trials and candidates must be positive")
    if trial_stride < candidates:
        raise ValueError("trial_stride must exceed the candidate index range")
    schedule = []
    for trial in range(trials):
        for candidate_index in range(candidates):
            schedule.append({
                "trial": trial,
                "candidate_index": candidate_index,
                "generation_seed": int(base_seed + trial * trial_stride + candidate_index),
            })
    seeds = [item["generation_seed"] for item in schedule]
    if len(seeds) != len(set(seeds)):
        raise ValueError("generation schedule contains duplicate seeds")
    return schedule


def load_completed_keys(
    path: Path,
    *,
    condition: str,
    rows: Iterable[dict[str, Any]],
    schedule: Iterable[dict[str, int]],
    split: str,
    expected_contract: dict[str, Any] | None = None,
) -> set[tuple[str, int]]:
    """Validate resumable output before allowing records to be skipped.

    A partially written generation file is an execution artifact, not an
    authority.  Every existing row must belong to this exact condition,
    population and seed schedule, and keys must be unique.  This prevents a
    stale file from silently causing a missing or cross-condition comparison.
    """
    if not path.is_file():
        return set()
    population = {str(row["cluster_id"]): row for row in rows}
    schedule = list(schedule)
    schedule_by_seed = {
        int(item["generation_seed"]): (int(item["trial"]), int(item["candidate_index"]))
        for item in schedule
    }
    completed: set[tuple[str, int]] = set()
    for line_number, item in enumerate(read_jsonl(path), 1):
        if expected_contract is not None and item.get("generation_contract") != expected_contract:
            raise ValueError("existing generation method/config/checkpoint contract mismatch")
        required = {
            "system_id", "image_id", "image_sha256", "split", "generation_seed",
            "trial", "candidate_index", "candidate_count",
        }
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"existing generation row {path}:{line_number} missing {missing}")
        if str(item["system_id"]) != condition:
            raise ValueError(
                f"existing generation row {path}:{line_number} belongs to condition "
                f"{item['system_id']!r}, expected {condition!r}"
            )
        cluster = str(item["image_id"])
        if cluster not in population:
            raise ValueError(f"existing generation row {path}:{line_number} has unknown image {cluster}")
        if str(item["split"]) != split:
            raise ValueError(f"existing generation row {path}:{line_number} has wrong split")
        if str(item["image_sha256"]) != str(population[cluster]["image_sha256"]):
            raise ValueError(f"existing generation row {path}:{line_number} has image hash mismatch")
        seed = int(item["generation_seed"])
        expected_trial_candidate = schedule_by_seed.get(seed)
        if expected_trial_candidate is None:
            raise ValueError(f"existing generation row {path}:{line_number} has unknown seed {seed}")
        trial, candidate_index = expected_trial_candidate
        if (int(item["trial"]), int(item["candidate_index"])) != (trial, candidate_index):
            raise ValueError(f"existing generation row {path}:{line_number} has seed/trial mismatch")
        expected_candidate_count = len({int(x["candidate_index"]) for x in schedule})
        if int(item["candidate_count"]) != expected_candidate_count:
            raise ValueError(f"existing generation row {path}:{line_number} has wrong candidate_count")
        key = (cluster, seed)
        if key in completed:
            raise ValueError(f"duplicate existing generation key at {path}:{line_number}: {key}")
        completed.add(key)
    return completed


def assert_complete_keys(
    completed: set[tuple[str, int]],
    *,
    rows: Iterable[dict[str, Any]],
    schedule: Iterable[dict[str, int]],
) -> None:
    expected = {
        (str(row["cluster_id"]), int(item["generation_seed"]))
        for row in rows
        for item in schedule
    }
    if completed != expected:
        missing = sorted(expected - completed)[:5]
        extra = sorted(completed - expected)[:5]
        raise RuntimeError(
            f"generation key set is incomplete: got={len(completed)} expected={len(expected)} "
            f"missing={missing} extra={extra}"
        )


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_population(path: Path, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path / f"{split}.jsonl")
    if not rows:
        raise ValueError(f"empty public population split: {path / (split + '.jsonl')}")
    required = {
        "row_id", "cluster_id", "dataset", "contest_number", "image", "image_sha256",
        "standard_description",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"population rows missing fields: {missing}")
    seen: set[str] = set()
    for row in rows:
        # The sealed public population stores the HOMER split as
        # ``homer_description_split``; ``split`` is a generation-view field
        # and is not present in the upstream population rows.  Normalize the
        # alias here rather than requiring a mutable data rewrite.
        declared_split = str(row.get("split", row.get("homer_description_split", "")))
        if declared_split != split:
            raise ValueError(
                f"population row {row.get('row_id')} has split={declared_split!r}, expected {split!r}"
            )
        row["split"] = declared_split
        cluster = str(row["cluster_id"])
        if cluster in seen:
            raise ValueError(f"population split has duplicate cluster: {cluster}")
        seen.add(cluster)
        image = resolve_path(row["image"])
        if not image.is_file():
            raise FileNotFoundError(f"missing image for {cluster}: {image}")
        declared = str(row["image_sha256"])
        if sha256(image) != declared:
            raise ValueError(f"image_sha256 mismatch for {cluster}: {image}")
        if not str(row["standard_description"]).strip():
            raise ValueError(f"empty standard_description for {cluster}")
    return rows


def load_bridge(config: dict[str, Any], checkpoint: Path, backend: QwenBackend) -> ReceiverDrivenCrossAttentionBridge:
    if config["experiment"].get("baseline") != "receiver_cross_attention":
        raise ValueError("comparison entry point requires receiver_cross_attention config")
    width = int(backend.model.get_input_embeddings().weight.shape[1])
    bridge_cfg = config["bridge"]
    bridge = ReceiverDrivenCrossAttentionBridge(
        width,
        width,
        layer_indices=[int(value) for value in bridge_cfg["layer_indices"]],
        bottleneck_dim=int(bridge_cfg["bottleneck_dim"]),
        heads=int(bridge_cfg["heads"]),
        gate_init=float(bridge_cfg.get("gate_init", 0.1)),
        channel_fusion=str(bridge_cfg.get("channel_fusion", "learned")),
        projection_mode=str(bridge_cfg.get("projection_mode", "shared")),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "bridge" not in payload:
        raise ValueError(f"bridge checkpoint has no bridge state: {checkpoint}")
    expected = {
        "memory_source": config.get("protocol", {}).get("memory_source", LEGACY_MEMORY_POLICY),
        "training_target_policy": config.get("protocol", {}).get("training_target_policy", LEGACY_TARGET_POLICY),
    }
    historical = {"memory_source": LEGACY_MEMORY_POLICY, "training_target_policy": LEGACY_TARGET_POLICY}
    if payload.get("method_contract", historical) != expected:
        raise ValueError("checkpoint payload method contract mismatch")
    bridge.load_state_dict(payload["bridge"], strict=True)
    bridge.to(model_device(backend.model)).eval()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    return bridge


def validate_checkpoint(checkpoint: Path, *, expected_contract: dict[str, str] | None = None) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    completion = checkpoint.parent / "complete.json"
    run_manifest = checkpoint.parent / "run_manifest.json"
    if not completion.is_file() or not run_manifest.is_file():
        raise ValueError(
            "a scientific comparison requires a completed bridge run with "
            "complete.json and run_manifest.json"
        )
    completion_payload = json.loads(completion.read_text(encoding="utf-8"))
    if completion_payload.get("status") != "complete":
        raise ValueError(f"bridge completion gate is not complete: {completion_payload}")
    run_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    actual_contract = {
        "memory_source": run_payload.get("memory_source", LEGACY_MEMORY_POLICY),
        "training_target_policy": run_payload.get("training_target_policy", LEGACY_TARGET_POLICY),
    }
    if expected_contract is not None and actual_contract != expected_contract:
        raise ValueError("checkpoint method contract mismatch; revised method requires retraining")
    if run_payload.get("current_homer_public_route") is not True:
        raise ValueError("checkpoint was not trained on the current HOMER public route")
    if run_payload.get("policy_trainable_parameters") != 0:
        raise ValueError("checkpoint does not prove a frozen Planner/Generator policy")
    if run_payload.get("baseline") not in {None, "receiver_cross_attention"}:
        raise ValueError("checkpoint baseline is not receiver_cross_attention")
    return {
        "path": str(checkpoint.resolve()),
        "sha256": sha256(checkpoint),
        "completion_sha256": sha256(completion),
        "method_contract": actual_contract,
        "scientific_gate": "not_established_by_training_completion",
        "run_manifest_sha256": sha256(run_manifest),
        "run_manifest": {
            "git_commit": run_payload.get("git_commit"),
            "current_homer_public_route": run_payload.get("current_homer_public_route"),
            "policy_trainable_parameters": run_payload.get("policy_trainable_parameters"),
            "train_clusters": run_payload.get("train_clusters"),
            "validation_clusters": run_payload.get("validation_clusters"),
        },
    }


def load_current_context(
    rows: Iterable[dict[str, Any]],
    *,
    context_index_path: Path,
    trace_index_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contexts = load_context_index(context_index_path)
    traces = load_trace_index(trace_index_path)
    for row in rows:
        cluster = str(row["cluster_id"])
        context = contexts.get(cluster)
        if context is None:
            raise ValueError(f"missing sealed HOMER context for {cluster}")
        validate_row_context(row, context)
        trace = traces.get(cluster)
        if trace is None:
            raise ValueError(f"missing sealed Planner trace for {cluster}")
        if context.trace_path != str(trace.get("trace_path")):
            raise ValueError(f"context/trace path mismatch for {cluster}")
        if context.trace_sha256 != str(trace.get("trace_sha256")):
            raise ValueError(f"context/trace hash mismatch for {cluster}")
    return contexts, traces


def _caption_record(raw: str, *, cluster: str) -> tuple[str, str]:
    caption, explanation = parse_caption_blob(raw)
    if not caption:
        return "[EMPTY OUTPUT]", explanation
    return caption, explanation


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/pilot/cross_attention_caption_selected_plan_v2.yaml",
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=ROOT / "data/processed/homer_pretrained_7b_public_release_362",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--condition", choices=["text_homer_context_replay", "latent_bridge", "no_memory"], required=True,
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Completed current bridge checkpoint; required only for latent_bridge.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260910)
    parser.add_argument("--trial-stride", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol", {}).get("bridge_prompt_track") != PUBLIC_BRIDGE_PROMPT_TRACK:
        raise ValueError("comparison requires the current public HOMER bridge prompt track")
    for key in ("adapter", "planner_adapter", "generator_adapter"):
        if config.get("model", {}).get(key) is not None:
            raise ValueError(f"current public comparison forbids model.{key}")
    if args.trials != 5 or args.candidates != 5:
        # Non-paper schedules are useful for extension diagnostics but must be
        # visibly labelled in provenance instead of being called HOMER primary.
        schedule_track = "extension_custom_schedule"
    else:
        schedule_track = "homer_comparable_5_candidates_5_trials"
    if args.condition == "latent_bridge" and args.checkpoint is None:
        raise ValueError("latent_bridge requires --checkpoint")
    method_contract = {
        "memory_source": config.get("protocol", {}).get("memory_source", LEGACY_MEMORY_POLICY),
        "training_target_policy": config.get("protocol", {}).get("training_target_policy", LEGACY_TARGET_POLICY),
    }
    if method_contract["memory_source"] not in {LEGACY_MEMORY_POLICY, SELECTED_PLAN_POLICY}:
        raise ValueError("unsupported memory source")
    parser_policy = config.get("protocol", {}).get("caption_parser", "official_compatibility_v1")
    if parser_policy not in {"strict_caption_v1", "official_compatibility_v1"}:
        raise ValueError("unsupported caption parser")
    checkpoint_meta = (
        validate_checkpoint(args.checkpoint.resolve(), expected_contract=method_contract)
        if args.checkpoint is not None else None
    )
    rows = load_population(args.dataset.resolve(), args.split)
    if args.max_rows is not None:
        if args.max_rows < 1:
            raise ValueError("max-rows must be positive")
        rows = rows[: args.max_rows]
    context_index_path = resolve_path(config["data"]["context_index"])
    trace_index_path = resolve_path(config["data"]["trace_index"])
    contexts, traces = load_current_context(
        rows, context_index_path=context_index_path, trace_index_path=trace_index_path,
    )

    model_cfg = config["model"]
    backend = QwenBackend.load(
        model_cfg["name"], revision=model_cfg["revision"], adapter=None,
        load_in_4bit=True,
        min_visual_tokens=int(model_cfg["min_visual_tokens"]),
        max_visual_tokens=int(model_cfg["max_visual_tokens"]),
    )
    bridge = (
        load_bridge(config, args.checkpoint.resolve(), backend)
        if args.condition == "latent_bridge" else None
    )
    schedule = generation_schedule(
        trials=args.trials, candidates=args.candidates,
        base_seed=args.base_seed, trial_stride=args.trial_stride,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generation_contract = {
        **method_contract, "caption_parser": parser_policy,
        "config_sha256": sha256(config_path), "max_new_tokens": args.max_new_tokens,
        "context_index_sha256": sha256(context_index_path),
        "trace_index_sha256": sha256(trace_index_path),
        "checkpoint_sha256": None if bridge is None else checkpoint_meta["sha256"],
    }
    completed = load_completed_keys(
        args.output,
        condition=args.condition,
        rows=rows,
        schedule=schedule,
        split=args.split,
        expected_contract=generation_contract,
    )
    total = len(rows) * len(schedule)
    written = len(completed)
    with args.output.open("a", encoding="utf-8") as handle:
        for row in rows:
            cluster = str(row["cluster_id"])
            context = contexts[cluster]
            trace_record = traces[cluster]
            trace_states = None
            if bridge is not None:
                trace_states = (encode_selected_plan(backend, context)
                    if method_contract["memory_source"] == SELECTED_PLAN_POLICY else load_trace(
                        resolve_path(trace_record["trace_path"]),
                        expected_sha256=str(trace_record["trace_sha256"]),
                    ))
                parameter = next(bridge.parameters())
                full_memory = {
                    name: trace_states[name].states.to(
                        device=parameter.device, dtype=parameter.dtype,
                    ) for name in TypedLatentBridge.channel_order
                }
            for item in schedule:
                seed = item["generation_seed"]
                if (cluster, seed) in completed:
                    continue
                messages = (
                    public_text_caption_messages(context)
                    if args.condition == "text_homer_context_replay"
                    else public_latent_caption_messages(context)
                )
                if bridge is None:
                    raw = backend.generate(
                        messages, temperature=1.0, max_new_tokens=args.max_new_tokens, seed=seed,
                    )
                else:
                    assert full_memory is not None
                    with bridge.inject(backend.model, full_memory):
                        raw = backend.generate(
                            messages, temperature=1.0,
                            max_new_tokens=args.max_new_tokens, seed=seed,
                        )
                official_caption, official_explanation = _caption_record(raw, cluster=cluster)
                parsed = parse_caption_strict(raw)
                caption, explanation = official_caption, official_explanation
                if parser_policy == "strict_caption_v1":
                    caption = parsed.caption if parsed.status == "ok" else "[INVALID CAPTION OUTPUT]"
                    explanation = parsed.explanation
                record = {
                    "schema_version": 2,
                    "generation_contract": generation_contract,
                    "selected_plan_sha256": selected_plan_sha256(context),
                    "parse_status": parsed.status,
                    "caption_parser": parser_policy,
                    "official_compatibility_caption": official_caption,
                    "official_compatibility_explanation": official_explanation,
                    "system_id": args.condition,
                    "image_id": cluster,
                    "image_path": str(resolve_path(row["image"])),
                    "image_sha256": str(row["image_sha256"]),
                    "split": str(row["split"]),
                    "target_culture": "en-US",
                    "caption": caption,
                    "generation_seed": seed,
                    "trial": item["trial"],
                    "candidate_index": item["candidate_index"],
                    "candidate_count": args.candidates,
                    "raw_output": raw,
                    "explanation": explanation,
                    "condition": args.condition,
                    "context_prompt_track": PUBLIC_BRIDGE_PROMPT_TRACK,
                    "context_sha256": sha256(context_index_path),
                    "trace_index_sha256": sha256(trace_index_path),
                    "trace_sha256": str(trace_record["trace_sha256"]),
                    "checkpoint_sha256": checkpoint_meta["sha256"] if bridge is not None else None,
                    "sampling": {
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "repetition_penalty": 1.0,
                        "max_new_tokens": args.max_new_tokens,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed.add((cluster, seed))
                written += 1
                print(json.dumps({"completed": written, "total": total, "cluster_id": cluster, "seed": seed}), flush=True)

    assert_complete_keys(completed, rows=rows, schedule=schedule)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "track": schedule_track,
        "condition": args.condition,
        "split": args.split,
        "rows": len(rows),
        "records": len(completed),
        "trials": args.trials,
        "candidates_per_image": args.candidates,
        "seed_schedule": schedule,
        "sampling": {
            "temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0,
            "max_new_tokens": args.max_new_tokens,
        },
        "model": {
            "name": model_cfg["name"],
            "revision": model_cfg["revision"],
            "adapter": None,
        },
        "config_sha256": sha256(config_path),
        "context_index_sha256": sha256(context_index_path),
        "trace_index_sha256": sha256(trace_index_path),
        "checkpoint": checkpoint_meta if bridge is not None else None,
        "git_commit": git_commit(),
        "output_sha256": sha256(args.output),
        "generation_contract": generation_contract,
        "scientific_status": "exploratory_comparison_not_evidence_of_mechanism_success",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
