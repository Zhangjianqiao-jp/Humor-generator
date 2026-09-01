#!/usr/bin/env python3
"""Exhaustive, non-semantic integrity audit for every v3.5 dataset row.

The audit intentionally does not print captions or descriptions from held-out
splits.  It verifies bytes, schema and split membership only, so it cannot be
used for model/threshold selection.
"""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/latent_bridge_v35"
SPLITS = (
    "train",
    "validation",
    "internal_test",
    "official_hia_unseen_test",
    "official_hia_seen_diagnostic",
)
REQUIRED_FIELDS = {
    "row_id", "cluster_id", "contest_number", "dataset", "source_split",
    "image", "image_sha256", "standard_description", "description_fields",
    "caption", "caption_rank", "caption_score",
}
INVALID_TEXT = {"", "u", "n", "k", "unk", "unknown", "nan", "none", "null"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"non-object JSON at {path}:{line_number}")
        rows.append(value)
    return rows


def audit(*, output_dir: Path | None = None) -> dict[str, Any]:
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    errors: list[dict[str, str]] = []
    row_reports: list[dict[str, Any]] = []
    image_reports: dict[str, dict[str, Any]] = {}
    global_row_ids: set[str] = set()
    cluster_splits: dict[str, str] = {}
    cluster_datasets: dict[str, set[str]] = {}
    split_counts: dict[str, dict[str, int]] = {}

    def fail(scope: str, identifier: str, message: str) -> None:
        errors.append({"scope": scope, "id": identifier, "error": message})

    for split in SPLITS:
        path = DATA / f"{split}.jsonl"
        try:
            rows = read_jsonl(path)
        except Exception as exc:
            fail("split", split, str(exc))
            continue
        clusters: set[str] = set()
        for row_index, row in enumerate(rows, 1):
            row_id = str(row.get("row_id", f"{split}:line:{row_index}"))
            row_errors: list[str] = []
            fields = set(row)
            if fields != REQUIRED_FIELDS:
                row_errors.append(
                    f"field set mismatch missing={sorted(REQUIRED_FIELDS-fields)} "
                    f"extra={sorted(fields-REQUIRED_FIELDS)}"
                )
            if row_id in global_row_ids:
                row_errors.append("duplicate row_id")
            global_row_ids.add(row_id)

            cluster = row.get("cluster_id")
            match = re.fullmatch(r"nycc_(\d+)", cluster if isinstance(cluster, str) else "")
            if match is None:
                row_errors.append("invalid cluster_id")
            elif not isinstance(row.get("contest_number"), int) or int(match.group(1)) != row["contest_number"]:
                row_errors.append("contest_number does not match cluster_id")
            if isinstance(cluster, str):
                clusters.add(cluster)
                previous = cluster_splits.setdefault(cluster, split)
                if previous != split:
                    row_errors.append(f"cluster leakage from {previous}")
                cluster_datasets.setdefault(cluster, set()).add(str(row.get("dataset")))

            for key in ("caption", "standard_description"):
                value = row.get(key)
                if not isinstance(value, str) or value.strip().casefold() in INVALID_TEXT:
                    row_errors.append(f"invalid {key}")
                elif "\x00" in value:
                    row_errors.append(f"NUL in {key}")
            if not isinstance(row.get("description_fields"), dict):
                row_errors.append("description_fields is not an object")
            if row.get("dataset") not in {"electronic_sheep", "humor_in_ai"}:
                row_errors.append("unknown dataset")
            if not isinstance(row.get("source_split"), str) or not row["source_split"].strip():
                row_errors.append("invalid source_split")
            rank = row.get("caption_rank")
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
                row_errors.append("invalid caption_rank")
            score = row.get("caption_score")
            if score is not None and (
                not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score))
            ):
                row_errors.append("invalid caption_score")

            image_value = row.get("image")
            image_path = Path(image_value) if isinstance(image_value, str) else Path()
            declared_hash = row.get("image_sha256")
            image_key = str(image_path)
            image_report = image_reports.get(image_key)
            if image_report is None:
                image_report = {
                    "image": image_key,
                    "status": "fail",
                    "declared_sha256": declared_hash,
                }
                try:
                    resolved = image_path.resolve(strict=True)
                    if ROOT not in resolved.parents:
                        raise RuntimeError("image escapes the v3.5 project root")
                    if not resolved.is_file():
                        raise RuntimeError("image is not a regular file")
                    actual_hash = sha256(resolved)
                    if not isinstance(declared_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
                        raise RuntimeError("invalid declared image SHA-256")
                    if actual_hash != declared_hash:
                        raise RuntimeError("image SHA-256 mismatch")
                    with Image.open(resolved) as image:
                        image.verify()
                    with Image.open(resolved) as image:
                        image.load()
                        width, height = image.size
                        image_format = image.format
                    if width < 1 or height < 1:
                        raise RuntimeError("zero-sized image")
                    image_report.update({
                        "status": "pass",
                        "actual_sha256": actual_hash,
                        "width": width,
                        "height": height,
                        "format": image_format,
                    })
                except Exception as exc:
                    image_report["error"] = str(exc)
                image_reports[image_key] = image_report
            elif image_report.get("declared_sha256") != declared_hash:
                row_errors.append("same image path has inconsistent declared hashes")
            if image_report["status"] != "pass":
                row_errors.append(f"image integrity failed: {image_report.get('error', 'unknown')}")

            status = "pass" if not row_errors else "fail"
            row_reports.append({
                "split": split,
                "row_id": row_id,
                "cluster_id": cluster,
                "image_sha256": declared_hash,
                "status": status,
                "errors": row_errors,
            })
            for message in row_errors:
                fail("row", row_id, message)

        split_counts[split] = {"rows": len(rows), "clusters": len(clusters)}
        expected_rows = int(manifest["split_rows"][split])
        expected_clusters = int(manifest["split_clusters"][split])
        if len(rows) != expected_rows:
            fail("split", split, f"row count {len(rows)} != manifest {expected_rows}")
        if len(clusters) != expected_clusters:
            fail("split", split, f"cluster count {len(clusters)} != manifest {expected_clusters}")
        expected_hash = manifest["output_sha256"][path.name]
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            fail("split", split, "JSONL SHA-256 mismatch")

    input_reports = []
    for raw_path, expected_hash in sorted(manifest.get("input_sha256", {}).items()):
        path = Path(raw_path)
        report = {"path": raw_path, "status": "fail", "expected_sha256": expected_hash}
        try:
            if not path.is_file():
                raise FileNotFoundError("source input is missing")
            actual_hash = sha256(path)
            if actual_hash != expected_hash:
                raise RuntimeError("source input SHA-256 mismatch")
            report.update(status="pass", actual_sha256=actual_hash)
        except Exception as exc:
            report["error"] = str(exc)
            fail("source_input", raw_path, str(exc))
        input_reports.append(report)

    # The manifest value is computed before official-HIA splits intentionally
    # discard their Electronic-Sheep duplicate rows.  Therefore the serialized
    # post-filter value is reported separately and must not be equated with the
    # pre-filter manifest statistic.
    cross_source = sum(len(values) > 1 for values in cluster_datasets.values())

    passed_rows = sum(row["status"] == "pass" for row in row_reports)
    passed_images = sum(row["status"] == "pass" for row in image_reports.values())
    passed_inputs = sum(row["status"] == "pass" for row in input_reports)
    summary = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "dataset": str(DATA),
        "splits": split_counts,
        "rows_checked": len(row_reports),
        "rows_passed": passed_rows,
        "unique_images_checked": len(image_reports),
        "unique_images_passed": passed_images,
        "source_inputs_checked": len(input_reports),
        "source_inputs_passed": passed_inputs,
        "clusters_checked": len(cluster_splits),
        "cross_source_duplicate_clusters": cross_source,
        "manifest_prefilter_cross_source_duplicate_clusters": int(
            manifest["cross_source_duplicate_clusters"]
        ),
        "errors": errors,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dataset_audit_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        with (output_dir / "row_checks.jsonl").open("w") as handle:
            for row in row_reports:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output_dir / "image_checks.jsonl").open("w") as handle:
            for row in sorted(image_reports.values(), key=lambda item: item["image"]):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output_dir / "source_input_checks.jsonl").open("w") as handle:
            for row in input_reports:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if errors:
        raise RuntimeError(f"dataset audit failed with {len(errors)} errors; first={errors[:5]}")
    return summary


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/preflight/dataset_audit")
    args = parser.parse_args()
    print(json.dumps(audit(output_dir=args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
