"""Hard evidence gate separating a runnable approximation from reproduction."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateResult:
    ready: bool
    failures: tuple[str, ...]

    def require(self) -> None:
        if not self.ready:
            raise RuntimeError("HOMER reproduction gate failed:\n- " + "\n- ".join(self.failures))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_reproduction(config: dict[str, Any], root: Path) -> GateResult:
    failures: list[str] = []
    evidence = config.get("paper_evidence", {})
    if evidence.get("paper") != "arXiv:2602.06423v2":
        failures.append("paper version must be pinned to arXiv:2602.06423v2")
    if not evidence.get("model_revision"):
        failures.append("the exact Qwen-VL checkpoint revision used by HOMER is not documented")
    if not evidence.get("description_provenance"):
        failures.append("benchmark standard-description provenance is missing")
    corpus = config.get("joke_corpus", {})
    manifest_path = root / corpus.get("manifest", "")
    if not manifest_path.is_file():
        failures.append("curated 335,570-joke manifest is missing")
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("rows") != 335_570:
            failures.append(f"HOMER reports 335,570 jokes; manifest has {manifest.get('rows')}")
        data_path_value = manifest.get("data_path")
        data_path = root / data_path_value if isinstance(data_path_value, str) else None
        if data_path is None or not data_path.is_file():
            failures.append("joke corpus data_path in manifest is missing or unreadable")
        elif manifest.get("sha256") != sha256(data_path):
            failures.append("joke corpus hash does not match its manifest")
    if corpus.get("top_k") != 5 or corpus.get("delta") != 5:
        failures.append("paper settings require top_k=5 and delta=5")
    evaluation = config.get("evaluation", {})
    if evaluation.get("candidates_per_image") != 5 or evaluation.get("repetitions") != 5:
        failures.append("paper evaluation requires five candidates and five repeated trials")
    if evaluation.get("temperature") != 1.0:
        failures.append("paper caption generation temperature is 1")
    return GateResult(not failures, tuple(failures))
