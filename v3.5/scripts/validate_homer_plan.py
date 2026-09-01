#!/usr/bin/env python3
"""Validate one HOMER plan JSON without loading a model."""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.homer.contracts import validate_plan


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    row = json.loads(args.plan.read_text())
    plan = validate_plan(row["description"], row["conflicts"], row["local"], row["global"])
    print(json.dumps({
        "valid": True,
        "conflict_pairs": len(plan.conflicts),
        "local_chains": len(plan.local_chains),
        "global_chains": len(plan.global_chains),
    }))


if __name__ == "__main__":
    main()
