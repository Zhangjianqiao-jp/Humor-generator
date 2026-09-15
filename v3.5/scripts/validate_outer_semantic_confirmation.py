#!/usr/bin/env python3
"""Fail closed on malformed outer semantic-confirmation artifacts."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path


CHANNELS = ("conflict", "local", "global")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-clusters", type=int, default=40)
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args()
    root = args.output.resolve()
    summary_path = root / "summary.json"
    details_path = root / "cluster_seed_channel_details.jsonl"
    manifest_path = root / "manifest.json"
    required = [summary_path, details_path, manifest_path, root / "outer_cluster_ids.json"]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(json.dumps({"status": "fail", "reason": "missing_artifacts", "missing": missing}))
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    details = [json.loads(line) for line in details_path.read_text().splitlines() if line.strip()]
    cluster_file = json.loads((root / "outer_cluster_ids.json").read_text())
    clusters = {str(row["cluster_id"]) for row in details}
    seeds = {int(row["seed"]) for row in details}
    keys = [(int(row["seed"]), str(row["cluster_id"]), str(row["channel"])) for row in details]
    errors: list[str] = []
    if len(clusters) != args.expected_clusters:
        errors.append(f"expected {args.expected_clusters} clusters, found {len(clusters)}")
    if len(seeds) != args.expected_seeds:
        errors.append(f"expected {args.expected_seeds} seeds, found {len(seeds)}")
    if len(keys) != len(set(keys)):
        errors.append("duplicate seed/cluster/channel detail")
    if set(str(channel) for channel in summary.get("channel_summary", {})) != set(CHANNELS):
        errors.append("summary does not contain exactly conflict/local/global")
    expected_rows = len(clusters) * len(seeds) * len(CHANNELS)
    if len(details) != expected_rows:
        errors.append(f"expected {expected_rows} detail rows, found {len(details)}")
    if set(cluster_file.get("clusters", [])) != clusters or int(cluster_file.get("count", -1)) != len(clusters):
        errors.append("outer_cluster_ids.json disagrees with details")
    if summary.get("engineering_gate_pass") is not True:
        errors.append("engineering_gate_pass is not true")
    if summary.get("status") not in {"outer_semantic_go", "outer_semantic_inconclusive"}:
        errors.append(f"invalid semantic status {summary.get('status')!r}")
    for row in details:
        if row.get("channel") not in CHANNELS:
            errors.append(f"unknown channel {row.get('channel')!r}")
        for name in ("matched_logp", "counterfactual_logp", "gap", "matched_nll"):
            try:
                if not math.isfinite(float(row[name])):
                    errors.append(f"non-finite {name}")
            except (KeyError, TypeError, ValueError):
                errors.append(f"missing/non-numeric {name}")
    if manifest.get("details_sha256") != sha256(details_path):
        errors.append("manifest details_sha256 mismatch")
    if manifest.get("summary_sha256") != sha256(summary_path):
        errors.append("manifest summary_sha256 mismatch")
    if manifest.get("outer_cluster_ids_sha256") != summary.get("outer_cluster_ids_sha256"):
        errors.append("manifest outer cluster hash mismatch")
    if errors:
        raise SystemExit(json.dumps({"status": "fail", "errors": errors}, ensure_ascii=False))
    report = {
        "status": "pass",
        "semantic_status": summary["status"],
        "clusters": len(clusters),
        "seeds": sorted(seeds),
        "details": len(details),
        "statistical_unit": summary.get("statistical_unit"),
        "caption_generation": "excluded",
        "preference_learning": "excluded",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
