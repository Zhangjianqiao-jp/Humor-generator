#!/usr/bin/env python3
"""Convert exact-HOMER pairwise A/B judgments into Pass@K input rows.

The pinned HOMER evaluator compares every generated caption (A) with each of
five human references (B).  A generated caption is counted as successful for a
specific reference when its response is ``A``.  The official implementation
then applies the unbiased Pass@K estimator independently for each reference
and averages the five references in each group.  This adapter preserves that
granularity by naming groups ``<official-group>/gt_caption<1..5>``.

No default, tie, missing response, or guessed score is inserted.  The command
fails closed unless the judgment packet IDs are an exact one-to-one match with
the private mapping.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


GROUPS = ("1-9", "200-209", "1000-1009")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def load_judgments(path: Path) -> dict[str, str]:
    """Load JSONL decisions or a single {decisions:{packet_id: decision}} object."""
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict) and isinstance(value.get("decisions"), dict):
        raw = value["decisions"]
        decisions = {str(key): item for key, item in raw.items()}
    else:
        decisions = {}
        for line_number, row in enumerate(read_jsonl(path), 1):
            packet_id = row.get("packet_id")
            decision = row.get("decision")
            if not isinstance(packet_id, str) or not packet_id:
                raise ValueError(f"judgment row {line_number}: packet_id is required")
            if packet_id in decisions:
                raise ValueError(f"duplicate judgment packet_id: {packet_id}")
            decisions[packet_id] = decision
    invalid = {key: decision for key, decision in decisions.items() if decision not in ("A", "B")}
    if invalid:
        sample = next(iter(invalid.items()))
        raise ValueError(f"judgments must be exactly A or B; invalid sample={sample!r}")
    return decisions


def aggregate(mapping_path: Path, judgment_path: Path, output_path: Path) -> dict[str, Any]:
    mapping = read_jsonl(mapping_path)
    if not mapping:
        raise ValueError("private mapping is empty")
    judgments = load_judgments(judgment_path)
    expected_ids = {row.get("packet_id") for row in mapping}
    if None in expected_ids:
        raise ValueError("private mapping contains a row without packet_id")
    if len(expected_ids) != len(mapping):
        raise ValueError("private mapping contains duplicate packet_id")
    judgment_ids = set(judgments)
    missing = expected_ids - judgment_ids
    extra = judgment_ids - expected_ids
    if missing or extra:
        raise ValueError(f"judgment packet set mismatch: missing={len(missing)} extra={len(extra)}")

    # A separate key is retained for each human reference caption, matching
    # the released calculate_pass_metrics_humorAI.py implementation.
    by_reference: dict[tuple[str, str, int, str, int], list[str]] = defaultdict(list)
    metadata: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    for row in mapping:
        condition = row.get("condition")
        image_id = row.get("image_id")
        trial = row.get("trial")
        group = row.get("reference_group")
        reference_index = row.get("reference_index")
        candidate_index = row.get("candidate_index")
        key = (condition, image_id, int(trial), group, int(reference_index))
        if key not in metadata:
            metadata[key] = {
                "contest_number": row.get("contest_number"),
                "image_path": row.get("image_path"),
                "image_sha256": row.get("image_sha256"),
                "split": row.get("split"),
                "generation_seeds": {},
            }
        if int(candidate_index) in metadata[key]["generation_seeds"]:
            raise ValueError(f"duplicate candidate index for {key}: {candidate_index}")
        metadata[key]["generation_seeds"][int(candidate_index)] = int(row["generation_seed"])
        packet_id = row.get("packet_id")
        by_reference[key].append(judgments[packet_id])

    # Every official group has five references.  Do not silently aggregate a
    # partial reference set, because that would change the released evaluator's
    # denominator while still producing apparently valid Pass@K rows.
    reference_sets: dict[tuple[str, str, int, str], set[int]] = defaultdict(set)
    for condition, image_id, trial, group, reference_index in by_reference:
        reference_sets[(condition, image_id, trial, group)].add(reference_index)
    for base_key, indices in reference_sets.items():
        if indices != set(range(5)):
            raise ValueError(f"{base_key} has reference indices {sorted(indices)}; expected [0, 1, 2, 3, 4]")

    records: list[dict[str, Any]] = []
    for key in sorted(by_reference, key=lambda item: (item[0], item[1], item[2], item[3], item[4])):
        condition, image_id, trial, group, reference_index = key
        responses = by_reference[key]
        if len(responses) != 5:
            raise ValueError(f"{key} has {len(responses)} candidates; expected five")
        if len(metadata[key]["generation_seeds"]) != 5:
            raise ValueError(f"{key} has incomplete candidate index set")
        records.append({
            "system_id": condition,
            "image_id": image_id,
            "contest_number": metadata[key]["contest_number"],
            "trial": trial,
            "reference_group": f"{group}/gt_caption{reference_index + 1}",
            "official_reference_group": group,
            "reference_index": reference_index,
            "candidate_count": 5,
            "winning_caption_count": sum(response == "A" for response in responses),
            "image_path": metadata[key]["image_path"],
            "image_sha256": metadata[key]["image_sha256"],
            "split": metadata[key]["split"],
            "generation_seeds": metadata[key]["generation_seeds"],
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": 1,
        "status": "complete_judgments_aggregated",
        "mapping_rows": len(mapping),
        "judgment_rows": len(judgments),
        "homer_records": len(records),
        "conditions": sorted({row["system_id"] for row in records}),
        "images": len({row["image_id"] for row in records}),
        "trials": len({row["trial"] for row in records}),
        "reference_groups": list(GROUPS),
        "references_per_group": 5,
        "candidate_count": 5,
        "decision_contract": "A=generated caption wins against that human reference; B=reference wins",
        "output": str(output_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-mapping", required=True, type=Path)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.private_mapping, args.judgments, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
