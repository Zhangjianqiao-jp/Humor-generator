#!/usr/bin/env python3
"""Append one structured failure/plan-change record to the v3.5 ledger."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs" / "EXPERIMENT_FAILURES.jsonl"
ALLOWED_LAYERS = {"environment", "data", "engineering", "method", "evaluation"}
ALLOWED_STATUS = {"open", "fixed", "no_go", "superseded"}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--layer", choices=sorted(ALLOWED_LAYERS), required=True)
    parser.add_argument("--status", choices=sorted(ALLOWED_STATUS), required=True)
    parser.add_argument("--symptom", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--root-cause", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--plan-change", required=True)
    parser.add_argument("--artifacts", nargs="*", default=[])
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    record = {
        "id": args.id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "layer": args.layer,
        "status": args.status,
        "symptom": args.symptom,
        "evidence": args.evidence,
        "root_cause": args.root_cause,
        "action": args.action,
        "plan_change": args.plan_change,
        "artifacts": args.artifacts,
    }
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
