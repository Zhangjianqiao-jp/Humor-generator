"""Hard evidence gate separating a runnable approximation from reproduction."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateResult:
    ready: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def require(self) -> None:
        if not self.ready:
            raise RuntimeError("HOMER reproduction gate failed:\n- " + "\n- ".join(self.failures))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_record_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def audit_reproduction(config: dict[str, Any], root: Path) -> GateResult:
    failures: list[str] = []
    warnings: list[str] = []
    evidence = config.get("paper_evidence", {})
    if evidence.get("paper") != "arXiv:2602.06423v2":
        failures.append("paper version must be pinned to arXiv:2602.06423v2")
    if not evidence.get("model_revision"):
        failures.append("the exact Qwen-VL checkpoint revision used by HOMER is not documented")
    model_manifest_value = evidence.get("model_manifest")
    if not model_manifest_value or not (root / model_manifest_value).is_file():
        failures.append("pinned local model manifest is missing")
    else:
        model_manifest = json.loads((root / model_manifest_value).read_text())
        if model_manifest.get("model_id") != evidence.get("model_name"):
            failures.append("local model ID does not match its manifest")
        if model_manifest.get("revision") != evidence.get("model_revision"):
            failures.append("local model revision does not match its manifest")
        snapshot = Path(model_manifest.get("local_snapshot_path", ""))
        if not snapshot.is_dir():
            failures.append("pinned local Qwen2.5-VL snapshot is unavailable")
        else:
            for filename, expected in model_manifest.get("metadata_sha256", {}).items():
                path = snapshot / filename
                if not path.is_file() or sha256(path) != expected:
                    failures.append(f"local Qwen2.5-VL metadata mismatch: {filename}")
            for filename, expected in model_manifest.get("weight_lfs_sha256", {}).items():
                path = snapshot / filename
                if not path.is_file():
                    failures.append(f"local Qwen2.5-VL weight shard is missing: {filename}")
                elif path.is_symlink() and Path(path.readlink()).name != expected:
                    failures.append(f"local Qwen2.5-VL LFS object mismatch: {filename}")
    if evidence.get("homer_exact_model_match") is not True:
        warnings.append(
            "using pinned Qwen2.5-VL-7B-Instruct as a project substitution; HOMER's exact Qwen-VL revision remains undisclosed"
        )
    description_manifest_value = evidence.get("description_provenance")
    if not description_manifest_value:
        failures.append("benchmark standard-description provenance is missing")
    elif not (root / description_manifest_value).is_file():
        failures.append("benchmark standard-description manifest is missing")
    corpus = config.get("joke_corpus", {})
    manifest_path = root / corpus.get("manifest", "")
    if not manifest_path.is_file():
        failures.append("official HOMER asset manifest is missing")
    else:
        manifest = json.loads(manifest_path.read_text())
        joke = manifest.get("joke_corpus", {})
        if joke.get("paper_declared_rows") != 335_570:
            failures.append("asset manifest does not preserve HOMER's declared 335,570 count")
        if joke.get("csv_records") != 335_569:
            failures.append(f"released HOMER CSV should contain 335,569 records; manifest has {joke.get('csv_records')}")
        if joke.get("physical_lines") != 335_570:
            failures.append("released HOMER CSV should contain 335,570 physical lines including header")
        warnings.append(
            "HOMER declares 335,570 jokes, but its released CSV contains 335,569 records plus one header"
        )
        data_path_value = joke.get("local_path")
        data_path = root / data_path_value if isinstance(data_path_value, str) else None
        if data_path is None or not data_path.is_file():
            failures.append("locally fetched HOMER joke corpus is missing or unreadable")
        elif joke.get("sha256") != sha256(data_path):
            failures.append("joke corpus hash does not match its manifest")
        elif csv_record_count(data_path) != joke.get("csv_records"):
            failures.append("joke corpus record count does not match its manifest")
        for name, asset in manifest.get("standard_descriptions", {}).items():
            path = root / asset.get("local_path", "")
            if not path.is_file():
                failures.append(f"standard-description asset is missing: {name}")
            elif asset.get("sha256") != sha256(path):
                failures.append(f"standard-description hash mismatch: {name}")
    if corpus.get("top_k") != 5 or corpus.get("delta") != 5:
        failures.append("paper settings require top_k=5 and delta=5")
    evaluation = config.get("evaluation", {})
    if evaluation.get("candidates_per_image") != 5 or evaluation.get("repetitions") != 5:
        failures.append("paper evaluation requires five candidates and five repeated trials")
    if evaluation.get("temperature") != 1.0:
        failures.append("paper caption generation temperature is 1")
    return GateResult(not failures, tuple(failures), tuple(warnings))
