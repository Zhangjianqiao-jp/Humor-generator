#!/usr/bin/env python3
"""Build an exact-HOMER primary evaluator packet from v3.5 generations.

The released HOMER evaluator compares each generated caption (A) with each of
five human reference captions (B), separately for the ``1-9``, ``200-209`` and
``1000-1009`` groups.  This script only prepares those pairwise requests.  It
never judges a caption and never writes a Pass@K value.

Public packets contain the official system/user prompts and an HMAC packet id,
but no condition or model identity.  The private mapping is required later to
convert A/B responses into the official ``winning_caption_count`` records.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import hmac
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "data/external/homer_official/d1334f295cc1a8f8f6dc67ba7e846c5939dddcec"
OFFICIAL_EVALUATOR = OFFICIAL_ROOT / "evaluation/humorousAI/evaluator_humorAI.py"
REFERENCE_FILE = OFFICIAL_ROOT / "evaluation/humorousAI/sample5gt.json"
CONTEXT_INDEX = ROOT / "data/cache/homer_pretrained_7b_homer_context/index.jsonl"
POPULATION_MANIFEST = ROOT / "manifests/homer_population_public_release_362.json"

GROUPS = ("1-9", "200-209", "1000-1009")
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
HOMER_PROMPT_SHA256 = "3b3626c78f6fcbd37d318387b02a2f698ffdf63b483124fdc470d62931c12487"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count


def extract_official_system_prompt(path: Path = OFFICIAL_EVALUATOR) -> str:
    """Extract the literal system prompt from the pinned public evaluator."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "compare_humor":
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "system_prompt" for target in statement.targets):
                continue
            if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                return statement.value.value
    raise ValueError("could not extract compare_humor system_prompt from pinned evaluator")


def user_prompt(context: str, candidate: str, reference: str) -> str:
    # Keep spacing and labels byte-for-byte equivalent to the official f-string.
    return f"Cartoon description: {context}\nCaption A : {candidate}\nCaption B : {reference}"


def packet_id(secret: bytes, *parts: Any) -> str:
    payload = canonical_json(parts).encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()[:32]


def load_contexts(path: Path = CONTEXT_INDEX) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        image_id = row.get("cluster_id") or row.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("context row has no cluster_id/image_id")
        if image_id in contexts:
            raise ValueError(f"duplicate context for {image_id}")
        contexts[image_id] = row
    return contexts


def load_references(path: Path = REFERENCE_FILE) -> dict[int, dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("sample5gt.json must be a list")
    references: dict[int, dict[str, list[str]]] = {}
    for row in payload:
        contest = row.get("contest_number")
        if not isinstance(contest, int):
            raise ValueError("reference row has non-integer contest_number")
        if contest in references:
            raise ValueError(f"duplicate reference contest {contest}")
        groups: dict[str, list[str]] = {}
        for group in GROUPS:
            captions = row.get(group)
            if not isinstance(captions, list) or len(captions) != 5 or not all(isinstance(x, str) and x for x in captions):
                raise ValueError(f"contest {contest} reference group {group} must contain five non-empty strings")
            groups[group] = captions
        references[contest] = groups
    return references


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.generations).resolve()
    output_dir = Path(args.output_dir).resolve()
    secret = Path(args.secret_file).read_bytes().strip()
    if len(secret) < 16:
        raise ValueError("secret must contain at least 16 bytes")
    if not source.is_file():
        raise FileNotFoundError(source)

    rows = read_jsonl(source)
    contexts = load_contexts()
    references = load_references()
    systems = sorted({row.get("system_id") for row in rows})
    expected_systems = ["latent_bridge", "text_homer_context_replay"]
    if systems != expected_systems:
        raise ValueError(f"expected systems {expected_systems}, got {systems}")
    if len(rows) != 2350:
        raise ValueError(f"expected 2350 normalized generation rows, got {len(rows)}")

    by_system_image: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        system = row.get("system_id")
        image_id = row.get("image_id")
        if not isinstance(system, str) or not isinstance(image_id, str):
            raise ValueError("generation row missing system_id/image_id")
        by_system_image.setdefault((system, image_id), []).append(row)
        if image_id not in contexts:
            raise ValueError(f"no HOMER context for {image_id}")
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise ValueError(f"empty caption for {system}/{image_id}")

    expected_images = {image for system, image in by_system_image if system == expected_systems[0]}
    if len(expected_images) != 47:
        raise ValueError(f"expected 47 test images, got {len(expected_images)}")
    for system in expected_systems:
        if {image for s, image in by_system_image if s == system} != expected_images:
            raise ValueError(f"system image population mismatch for {system}")
        for image_id in expected_images:
            candidates = by_system_image[(system, image_id)]
            if len(candidates) != 25:
                raise ValueError(f"expected 25 candidates for {system}/{image_id}, got {len(candidates)}")
            if any(row.get("candidate_count") != 5 for row in candidates):
                raise ValueError(f"candidate_count contract failed for {system}/{image_id}")

    system_prompt = extract_official_system_prompt()
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / "homer_primary_prompts.jsonl"
    private_path = output_dir / "homer_primary_private_mapping.jsonl"

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for system in expected_systems:
        for image_id in sorted(expected_images):
            context = contexts[image_id]
            contest = context.get("contest_number")
            if not isinstance(contest, int) or contest not in references:
                raise ValueError(f"missing references for contest {contest!r} ({image_id})")
            description = context.get("standard_description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"missing standard_description for {image_id}")
            candidates = sorted(by_system_image[(system, image_id)], key=lambda row: (int(row["trial"]), int(row["candidate_index"])))
            for candidate_row in candidates:
                trial = int(candidate_row["trial"])
                candidate_index = int(candidate_row["candidate_index"])
                candidate_seed = int(candidate_row["generation_seed"])
                for group in GROUPS:
                    for reference_index, reference in enumerate(references[contest][group]):
                        blind_id = packet_id(
                            secret, "homer-primary-v1", system, image_id, trial,
                            candidate_index, group, reference_index,
                        )
                        public_rows.append({
                            "schema_version": 1,
                            "packet_id": blind_id,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt(description, candidate_row["caption"], reference),
                            "response_contract": "exactly one token: A or B",
                            "protocol": "homer_official_humorAI_pairwise",
                        })
                        private_rows.append({
                            "schema_version": 1,
                            "packet_id": blind_id,
                            "condition": system,
                            "image_id": image_id,
                            "contest_number": contest,
                            "image_path": candidate_row.get("image_path"),
                            "image_sha256": candidate_row.get("image_sha256"),
                            "split": candidate_row.get("split"),
                            "trial": trial,
                            "candidate_index": candidate_index,
                            "generation_seed": candidate_seed,
                            "reference_group": group,
                            "reference_index": reference_index,
                            "candidate_caption": candidate_row["caption"],
                            "reference_caption": reference,
                            "description": description,
                        })

    public_rows.sort(key=lambda row: row["packet_id"])
    private_rows.sort(key=lambda row: row["packet_id"])
    write_jsonl(public_path, public_rows)
    write_jsonl(private_path, private_rows)

    provenance = {
        "schema_version": 1,
        "evaluation_id": "v35-pretrained-qwen7b-homer-primary-20260911",
        "track": "homer_comparable",
        "protocol": "homer_official_humorAI_pairwise_reference",
        "dataset": "homer_pretrained_7b_public_release_362",
        "code_commit": current_commit(),
        "source_generations_sha256": sha256_file(source),
        "dataset_manifest_sha256": sha256_file(POPULATION_MANIFEST),
        "models": [
            {"role": "planner", "model_id": MODEL_ID, "revision_or_snapshot": MODEL_REVISION, "adapter": None, "prompt_sha256": HOMER_PROMPT_SHA256},
            {"role": "generator", "model_id": MODEL_ID, "revision_or_snapshot": MODEL_REVISION, "adapter": None, "prompt_sha256": HOMER_PROMPT_SHA256},
            {"role": "bridge", "model_id": "v35_latent_bridge", "revision_or_snapshot": "f2c6eb623f89d7d129213ba86c29df99658f2dea1962ce18a9f1ae5d89d7d508", "adapter": None, "prompt_sha256": HOMER_PROMPT_SHA256},
        ],
        "generation": {
            "temperature": 1.0,
            "candidates_per_image": 5,
            "seeds": sorted({int(row["generation_seed"]) for row in rows}),
            "repeated_trials": 5,
            "max_new_tokens": 96,
        },
        "evaluator": {
            "model_id": "gpt-5-chat-latest",
            "canonical_model_id": "gpt-5-chat-latest",
            "version_or_date": "2026-09-11",
            "temperature": 0,
            "prompt_sha256": prompt_hash,
            "substitution": False,
        },
        "protocol_details": {
            "official_evaluator_source_sha256": sha256_file(OFFICIAL_EVALUATOR),
            "official_reference_file_sha256": sha256_file(REFERENCE_FILE),
            "homer_prompt_sha256": HOMER_PROMPT_SHA256,
            "evaluator_system_prompt_sha256": prompt_hash,
            "reference_groups": list(GROUPS),
            "references_per_group": 5,
            "pairwise_requests": len(public_rows),
            "conditions": expected_systems,
            "images": len(expected_images),
            "notes": "Generated caption is Caption A and each official human reference is Caption B; no model identity is public.",
        },
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "complete_packets_external_judgment_pending",
        "track": "homer_comparable",
        "protocol": "homer_official_humorAI_pairwise_reference",
        "source_generations": str(source),
        "source_generations_sha256": sha256_file(source),
        "dataset_manifest": str(POPULATION_MANIFEST),
        "dataset_manifest_sha256": sha256_file(POPULATION_MANIFEST),
        "public_prompts": public_path.name,
        "public_prompts_sha256": sha256_file(public_path),
        "private_mapping": private_path.name,
        "private_mapping_sha256": sha256_file(private_path),
        "provenance": "provenance.json",
        "official_evaluator": str(OFFICIAL_EVALUATOR),
        "official_evaluator_sha256": sha256_file(OFFICIAL_EVALUATOR),
        "evaluator_system_prompt_sha256": prompt_hash,
        "images": len(expected_images),
        "conditions": expected_systems,
        "trials": 5,
        "candidates_per_image": 5,
        "reference_groups": list(GROUPS),
        "references_per_group": 5,
        "pairwise_requests": len(public_rows),
        "judgments_required": len(public_rows),
        "no_judgments_written": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--secret-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
