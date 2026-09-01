#!/usr/bin/env python3
"""Fail when v3.0 is not running in its own pinned environment."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.0",
    "nltk": "3.10.3",
    "numpy": "2.5.1",
    "peft": "0.20.0",
    "pillow": "12.3.0",
    "pyyaml": "6.0.3",
    "qwen-vl-utils": "0.0.14",
    "safetensors": "0.8.0",
    "scikit-learn": "1.7.2",
    "scipy": "1.16.3",
    "torch": "2.12.0+cu126",
    "torchvision": "0.27.0+cu126",
    "transformers": "5.14.1",
}


def audit() -> dict[str, object]:
    executable = Path(sys.executable).absolute()
    expected_environment = (ROOT / ".venv").resolve()
    prefix = Path(sys.prefix).resolve()
    if prefix != expected_environment:
        raise RuntimeError(
            f"v3.0 requires sys.prefix={expected_environment}; got prefix={prefix}, executable={executable}"
        )
    leaked = [entry for entry in sys.path if "v2.5" in entry]
    if leaked:
        raise RuntimeError(f"v2.5 path leakage detected: {leaked}")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    mismatches = {
        name: {"expected": EXPECTED[name], "actual": actual}
        for name, actual in versions.items()
        if actual != EXPECTED[name]
    }
    if mismatches:
        raise RuntimeError(f"dependency lock mismatch: {mismatches}")
    return {
        "status": "pass",
        "python": sys.version.split()[0],
        "executable": str(executable),
        "prefix": str(prefix),
        "versions": versions,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
