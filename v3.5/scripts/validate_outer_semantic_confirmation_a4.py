#!/usr/bin/env python3
"""Fail closed on malformed A4 outer semantic-confirmation artifacts."""
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
    details_path = root / "cluster_seed_channel_details.jsonl"
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    clusters_path = root / "outer_cluster_ids.json"
    required = (details_path, summary_path, manifest_path, clusters_path)
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(json.dumps({"status": "fail", "reason": "missing_artifacts", "missing": missing}))

    details = [json.loads(line) for line in details_path.read_text().splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    cluster_file = json.loads(clusters_path.read_text())
    errors: list[str] = []
    clusters = {str(row.get("cluster_id")) for row in details}
    seeds = {int(row["seed"]) for row in details if "seed" in row}
    keys = [(int(row.get("seed", -1)), str(row.get("cluster_id")), str(row.get("channel"))) for row in details]
    if len(clusters) != args.expected_clusters:
        errors.append(f"expected {args.expected_clusters} clusters, found {len(clusters)}")
    if len(seeds) != args.expected_seeds:
        errors.append(f"expected {args.expected_seeds} seeds, found {len(seeds)}")
    if len(keys) != len(set(keys)):
        errors.append("duplicate seed/cluster/channel detail")
    expected_rows = len(clusters) * len(seeds) * len(CHANNELS)
    if len(details) != expected_rows:
        errors.append(f"expected {expected_rows} detail rows, found {len(details)}")
    if set(cluster_file.get("clusters", [])) != clusters or int(cluster_file.get("count", -1)) != len(clusters):
        errors.append("outer_cluster_ids.json disagrees with details")
    if manifest.get("protocol") != "A4_channel_isolated_v4":
        errors.append("manifest protocol is not A4_channel_isolated_v4")
    if manifest.get("target_only") is not True:
        errors.append("manifest target_only is not true")
    if manifest.get("semantic_prompt_include_image") is not False:
        errors.append("manifest semantic_prompt_include_image is not false")
    if manifest.get("zero_bridge_included") is not True:
        errors.append("manifest zero_bridge_included is not true")
    if summary.get("protocol") != "A4_channel_isolated_v4":
        errors.append("summary protocol is not A4_channel_isolated_v4")
    if summary.get("engineering_gate_pass") is not True:
        errors.append("engineering_gate_pass is not true")
    if summary.get("status") not in {"outer_semantic_go", "outer_semantic_inconclusive"}:
        errors.append(f"invalid semantic status {summary.get('status')!r}")
    for row in details:
        if row.get("channel") not in CHANNELS:
            errors.append(f"unknown channel {row.get('channel')!r}")
        if row.get("active_channels") != [row.get("channel")]:
            errors.append(f"non-isolated active channel at {row.get('cluster_id')} / {row.get('channel')}")
        if row.get("semantic_prompt_include_image") is not False:
            errors.append(f"image prompt enabled at {row.get('cluster_id')} / {row.get('channel')}")
        for name in (
            "matched_logp", "counterfactual_logp", "gap", "matched_nll",
            "donor_logp", "donor_nll", "zero_bridge_logp", "zero_bridge_nll",
        ):
            try:
                if not math.isfinite(float(row[name])):
                    errors.append(f"non-finite {name}")
            except (KeyError, TypeError, ValueError):
                errors.append(f"missing/non-numeric {name}")
        try:
            if float(row["zero_bridge_gap"]) != 0.0:
                errors.append("zero_bridge_gap is not exactly zero")
        except (KeyError, TypeError, ValueError):
            errors.append("missing/non-numeric zero_bridge_gap")

    if manifest.get("details_sha256") != sha256(details_path):
        errors.append("manifest details_sha256 mismatch")
    if manifest.get("summary_sha256") != sha256(summary_path):
        errors.append("manifest summary_sha256 mismatch")
    if errors:
        raise SystemExit(json.dumps({"status": "fail", "errors": errors}, ensure_ascii=False))
    print(json.dumps({
        "status": "pass",
        "semantic_status": summary["status"],
        "protocol": "A4_channel_isolated_v4",
        "clusters": len(clusters),
        "seeds": sorted(seeds),
        "details": len(details),
        "statistical_unit": "image_cluster",
        "caption_generation": "excluded",
        "preference_learning": "excluded",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
