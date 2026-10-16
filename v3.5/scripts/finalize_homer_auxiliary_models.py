#!/usr/bin/env python3
"""Finalize several auxiliary HOMER evaluator outputs.

The canonical primary evaluator remains pinned to ``gpt-5-chat-latest``. This
script is for a separate, explicitly labelled multi-model track. It validates
that every model judged exactly the same complete public packet set, computes
HOMER Pass@1/3/5 for each model, and (when an odd number of evaluators is
provided) computes a majority-vote result. All condition-level aggregation is
performed locally with the private mapping; the public packet is never written
to the report.

Judgment arguments use ``MODEL_ID=PATH``. Every JSONL row must contain
``packet_id``, ``decision``, ``model``, ``temperature`` and ``prompt_sha256``.
The script fails closed on missing/extra/duplicate packets or provenance
mismatches.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

try:  # package invocation
    from .aggregate_homer_primary_judgments import aggregate, read_jsonl
    from .finalize_homer_primary_evaluation import compute
except ImportError:  # direct CLI invocation
    from aggregate_homer_primary_judgments import aggregate, read_jsonl  # type: ignore
    from finalize_homer_primary_evaluation import compute  # type: ignore


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--judgments expects MODEL_ID=PATH, got {spec!r}")
        model, raw_path = spec.split("=", 1)
        model = model.strip()
        path = Path(raw_path)
        if not model or not path:
            raise ValueError(f"invalid judgment specification: {spec!r}")
        if model in seen:
            raise ValueError(f"duplicate evaluator model: {model}")
        seen.add(model)
        parsed.append((model, path))
    if not parsed:
        raise ValueError("at least one --judgments MODEL_ID=PATH is required")
    return parsed


def validate_rows(
    path: Path,
    *,
    model_id: str,
    expected_ids: set[str],
    prompt_hash: str,
    source_provenance_sha256: str,
) -> dict[str, str]:
    rows = read_jsonl(path)
    decisions: dict[str, str] = {}
    for index, row in enumerate(rows, 1):
        packet_id = row.get("packet_id")
        if not isinstance(packet_id, str) or not packet_id:
            raise ValueError(f"{path}:{index}: packet_id is required")
        if packet_id in decisions:
            raise ValueError(f"{path}:{index}: duplicate packet_id {packet_id}")
        if row.get("decision") not in ("A", "B"):
            raise ValueError(f"{path}:{index}: decision must be exactly A or B")
        if row.get("model") != model_id:
            raise ValueError(f"{path}:{index}: model does not match {model_id!r}")
        if row.get("temperature") != 0:
            raise ValueError(f"{path}:{index}: temperature must be 0")
        if row.get("prompt_sha256") != prompt_hash:
            raise ValueError(f"{path}:{index}: prompt_sha256 does not match packet provenance")
        if row.get("source_provenance_sha256") != source_provenance_sha256:
            raise ValueError(f"{path}:{index}: source provenance hash does not match packet provenance")
        decisions[packet_id] = row["decision"]
    actual = set(decisions)
    missing = expected_ids - actual
    extra = actual - expected_ids
    if missing or extra:
        raise ValueError(f"{path}: packet set mismatch: missing={len(missing)} extra={len(extra)}")
    return decisions


def write_decisions(path: Path, decisions: dict[str, str], *, model_id: str, prompt_hash: str, source_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for packet_id in sorted(decisions):
            handle.write(
                json.dumps(
                    {
                        "packet_id": packet_id,
                        "decision": decisions[packet_id],
                        "model": model_id,
                        "temperature": 0,
                        "prompt_sha256": prompt_hash,
                        "source_provenance_sha256": source_hash,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "model"


def summarize_one(
    *,
    mapping_path: Path,
    judgments_path: Path,
    output_dir: Path,
    label: str,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    records_suffix: str = "records.jsonl",
    expected_images: set[str] | None = None,
) -> dict[str, Any]:
    records_path = output_dir / f"{safe_name(label)}.{records_suffix}"
    summary = aggregate(mapping_path, judgments_path, records_path)
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = compute(records, bootstrap_seed=bootstrap_seed, bootstrap_replicates=bootstrap_replicates, expected_images=expected_images)
    return {
        "model_id": label,
        "judgments_path": str(judgments_path),
        "judgments_sha256": sha256_file(judgments_path),
        "aggregation": summary,
        "metrics": metrics,
    }


def pairwise_agreement(decisions: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    labels = sorted(decisions)
    result: list[dict[str, Any]] = []
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            ids = sorted(set(decisions[left]) & set(decisions[right]))
            agree = sum(decisions[left][packet_id] == decisions[right][packet_id] for packet_id in ids)
            result.append({"left": left, "right": right, "packets": len(ids), "agreement": agree / len(ids) if ids else None, "agree": agree})
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Auxiliary multi-model HOMER evaluation",
        "",
        "This is a separate auxiliary track; it must not be merged into the canonical HOMER table.",
        "",
        f"- Packets per evaluator: {result['packet_count']}",
        f"- Evaluator models: {', '.join(result['evaluator_model_ids'])}",
        f"- Transport: `{result.get('evaluation_transport', 'unspecified')}`",
        f"- Temperature protocol: `{result.get('temperature_protocol', 'unspecified')}`",
        f"- Prompt hash: `{result['prompt_sha256']}`",
        f"- Bootstrap: seed `{result['bootstrap']['seed']}`, replicates `{result['bootstrap']['replicates']}`",
        "",
        "## Evaluator agreement",
        "",
        "| Evaluator A | Evaluator B | Packets | Agreement |",
        "|---|---|---:|---:|",
    ]
    for row in result["pairwise_agreement"]:
        agreement = "n/a" if row["agreement"] is None else f"{row['agreement']:.6f}"
        lines.append(f"| {row['left']} | {row['right']} | {row['packets']} | {agreement} |")
    lines += ["", "## Pass@K by evaluator", "", "| Evaluator | Reference group | K | System | Pass@K | 95% CI |", "|---|---|---:|---|---:|---:|"]
    for evaluator in result["evaluators"]:
        for row in evaluator["metrics"]["summaries"]:
            lines.append(f"| {evaluator['model_id']} | {row['official_reference_group']} | {row['k']} | {row['system_id']} | {row['pass_at_k']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] |")
    if result.get("majority"):
        lines += ["", "## Majority vote", "", "Majority-vote metrics are included as an auxiliary consensus estimate; they are not a replacement for the canonical evaluator.", ""]
        for row in result["majority"]["metrics"]["paired_comparison"]:
            lines.append(f"- `{row['official_reference_group']}`, K={row['k']}: latent−text `{row['mean_difference']:.6f}`; 95% CI [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]")
    lines += ["", "No result is reported unless every evaluator has a complete one-to-one packet judgment set.", ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    population = Path(args.population)
    expected_images = {json.loads(line)["cluster_id"] for line in population.read_text().splitlines() if line.strip()}
    mapping_path = Path(args.private_mapping)
    provenance_path = Path(args.provenance)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    evaluator = provenance.get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError("provenance.evaluator is required")
    prompt_hash = evaluator.get("prompt_sha256")
    if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
        raise ValueError("provenance.evaluator.prompt_sha256 must be a SHA-256 string")
    if evaluator.get("temperature") != 0:
        raise ValueError("source evaluator temperature must be 0")
    mapping = read_jsonl(mapping_path)
    expected_ids = {row.get("packet_id") for row in mapping}
    if not mapping or None in expected_ids or len(expected_ids) != len(mapping):
        raise ValueError("private mapping must be non-empty and have unique packet IDs")
    source_hash = sha256_file(provenance_path)
    specs = parse_specs(args.judgments)
    decisions: dict[str, dict[str, str]] = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_id, path in specs:
        decisions[model_id] = validate_rows(
            path,
            model_id=model_id,
            expected_ids=expected_ids,
            prompt_hash=prompt_hash,
            source_provenance_sha256=source_hash,
        )
    transport_values: set[str] = set()
    temperature_protocols: set[str] = set()
    for _, path in specs:
        for row in read_jsonl(path):
            transport_values.add(str(row.get("evaluation_transport", "unspecified")))
            temperature_protocols.add(str(row.get("temperature_protocol", "unspecified")))
    evaluator_model_ids = [model_id for model_id, _ in specs]
    evaluators = [
        summarize_one(
            mapping_path=mapping_path,
            judgments_path=path,
            output_dir=output_dir,
            label=model_id,
            bootstrap_seed=args.bootstrap_seed + index * 1000,
            bootstrap_replicates=args.bootstrap_replicates,
            expected_images=expected_images,
        )
        for index, (model_id, path) in enumerate(specs)
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete_auxiliary_multimodel",
        "canonical_primary": False,
        "private_mapping_sha256": sha256_file(mapping_path),
        "provenance_sha256": source_hash,
        "prompt_sha256": prompt_hash,
        "packet_count": len(expected_ids),
        "evaluator_model_ids": evaluator_model_ids,
        "evaluation_transport": ",".join(sorted(transport_values)),
        "temperature_protocol": ",".join(sorted(temperature_protocols)),
        "evaluators": evaluators,
        "pairwise_agreement": pairwise_agreement(decisions),
        "bootstrap": {"seed": args.bootstrap_seed, "replicates": args.bootstrap_replicates, "unit": "image"},
    }
    if len(specs) >= 3 and len(specs) % 2 == 1:
        majority = {
            packet_id: "A" if sum(decisions[model][packet_id] == "A" for model, _ in specs) > len(specs) / 2 else "B"
            for packet_id in expected_ids
        }
        majority_path = output_dir / "majority_judgments.jsonl"
        write_decisions(majority_path, majority, model_id="majority-vote", prompt_hash=prompt_hash, source_hash=source_hash)
        result["majority"] = summarize_one(
            mapping_path=mapping_path,
            judgments_path=majority_path,
            output_dir=output_dir,
            label="majority-vote",
            bootstrap_seed=args.bootstrap_seed + 9000,
            bootstrap_replicates=args.bootstrap_replicates,
            expected_images=expected_images,
        )
    else:
        result["majority"] = None
    output_json = output_dir / "multimodel_summary.json"
    output_md = output_dir / "multimodel_summary.md"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    return {"status": result["status"], "packet_count": result["packet_count"], "evaluators": [item["model_id"] for item in evaluators], "output_json": str(output_json), "output_md": str(output_md)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize auxiliary multi-model HOMER judgments")
    parser.add_argument("--population", type=Path, default=Path(__file__).resolve().parents[1] / "data/processed/homer_pretrained_7b_public_release_362/test.jsonl")
    parser.add_argument("--private-mapping", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--judgments", required=True, action="append", metavar="MODEL_ID=PATH")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20250308)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
