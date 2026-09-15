#!/usr/bin/env python3
"""Join current HOMER population captions with current Planner/context caches.

The public-release population files intentionally contain one row per image
and only three source caption references.  This builder creates the *training*
view used by the latent bridge without mutating the immutable population
manifest.  Final HOMER evaluation must still read each row's complete ranking
CSV; this top-three source-caption view is never a benchmark reference group.
"""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from humor_generator_v35.data.traces import read_jsonl
from humor_generator_v35.training.public_bridge import load_context_index


DATA_VERSION = "homer_pretrained_7b_public_release_362"
DEFAULT_DATASET = ROOT / "data/processed/homer_pretrained_7b_public_release_362"
DEFAULT_TRACE_INDEX = ROOT / "data/cache/homer_pretrained_7b_planner_traces/index.jsonl"
DEFAULT_CONTEXT_INDEX = ROOT / "data/cache/homer_pretrained_7b_homer_context/index.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/homer_pretrained_7b_bridge_362"
POPULATION_MANIFEST = ROOT / "manifests/homer_population_public_release_362.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_unique(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value or value in result:
            raise ValueError(f"duplicate/empty {key} in {path}: {value!r}")
        result[value] = row
    return result


def build(
    *,
    dataset: Path,
    trace_index: Path,
    context_index: Path,
    output: Path,
    force: bool = False,
) -> dict[str, Any]:
    if dataset.resolve() == (ROOT / "data/processed/latent_bridge_v35").resolve():
        raise RuntimeError("historical latent_bridge_v35 is forbidden for the current bridge route")
    manifest = dataset / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    population_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    if population_manifest.get("data_version") != DATA_VERSION:
        raise ValueError("population dataset has the wrong data_version")
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite non-empty output; use --force: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_rows = _load_unique(dataset / "source_rows.jsonl", "row_id")
    traces = _load_unique(trace_index, "cluster_id")
    contexts = load_context_index(context_index)
    if len(traces) != 362 or len(contexts) != 362:
        raise ValueError("current bridge dataset requires exactly 362 traces and contexts")
    split_counts: dict[str, int] = {}
    source_count = 0
    output_files: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        population_rows = read_jsonl(dataset / f"{split}.jsonl")
        rows: list[dict[str, Any]] = []
        for population in population_rows:
            cluster = str(population["cluster_id"])
            if cluster not in traces or cluster not in contexts:
                raise ValueError(f"missing trace/context for {cluster}")
            context = contexts[cluster]
            if context.data_version != DATA_VERSION:
                raise ValueError(f"context {cluster} has the wrong data_version")
            ids = population.get("source_caption_row_ids")
            if not isinstance(ids, list) or len(ids) != 3:
                raise ValueError(f"{cluster} must have exactly three source caption references")
            for source_id in ids:
                source_id = str(source_id)
                if source_id not in source_rows:
                    raise ValueError(f"{cluster} references missing source row {source_id}")
                source = source_rows[source_id]
                for field in ("image", "image_sha256", "standard_description"):
                    if str(source.get(field, "")) != str(population.get(field, "")):
                        raise ValueError(f"{source_id}/{cluster} mismatch for {field}")
                caption = str(source.get("caption", "")).strip()
                if not caption:
                    raise ValueError(f"{source_id} has an empty caption")
                row = {
                    "schema_version": 1,
                    "data_version": DATA_VERSION,
                    "row_id": source_id,
                    "cluster_id": cluster,
                    "contest_number": int(population["contest_number"]),
                    "dataset": str(population.get("dataset", "humor_in_ai")),
                    "split": split,
                    "image": population["image"],
                    "image_sha256": population["image_sha256"],
                    "standard_description": population["standard_description"],
                    "caption": caption,
                    "caption_rank": source.get("caption_rank"),
                    "caption_score": source.get("caption_score"),
                    "population_key": population["population_key"],
                    "population_version": DATA_VERSION,
                    "ranking_path": population["ranking_path"],
                    "ranking_sha256": population["ranking_sha256"],
                    "trace_cluster_id": cluster,
                    "trace_path": traces[cluster]["trace_path"],
                    "trace_sha256": traces[cluster]["trace_sha256"],
                    "context_cluster_id": cluster,
                    "context_prompt_track": context.context_prompt_track,
                    "caption_source_policy": "public_release_source_rows_top3_for_bridge_training_only",
                }
                rows.append(row)
        rows.sort(key=lambda item: str(item["row_id"]))
        path = output / f"{split}.jsonl"
        write_jsonl(path, rows)
        split_counts[split] = len(rows)
        source_count += len(rows)
        output_files.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "rows": len(rows),
            "bytes": path.stat().st_size,
        })

    out_manifest = {
        "schema_version": 1,
        "data_version": DATA_VERSION,
        "status": "verified_current_pretrained_bridge_training_view",
        "population_manifest": str(POPULATION_MANIFEST.relative_to(ROOT)),
        "population_manifest_sha256": sha256(POPULATION_MANIFEST),
        "population_dataset_manifest": str(manifest.relative_to(ROOT)),
        "population_dataset_manifest_sha256": sha256(manifest),
        "trace_index": str(trace_index.relative_to(ROOT)),
        "trace_index_sha256": sha256(trace_index),
        "context_index": str(context_index.relative_to(ROOT)),
        "context_index_sha256": sha256(context_index),
        "split_counts": split_counts,
        "rows_total": source_count,
        "caption_source_policy": "public_release_source_rows_top3_for_bridge_training_only",
        "final_evaluation_policy": "use_complete_ranking_csv_and_HOMER_reference_groups; never this view",
        "historical_data_forbidden": True,
        "files": output_files,
    }
    manifest_path = output / "manifest.json"
    out_manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(out_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(out_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trace-index", type=Path, default=DEFAULT_TRACE_INDEX)
    parser.add_argument("--context-index", type=Path, default=DEFAULT_CONTEXT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build(
        dataset=args.dataset.resolve(),
        trace_index=args.trace_index.resolve(),
        context_index=args.context_index.resolve(),
        output=args.output.resolve(),
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
