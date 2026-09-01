#!/usr/bin/env python3
"""Map canonical multi-rater consensus labels to anonymous system comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_group_winrate_eval import wilson


def summarize_consensus(
    key_doc: dict[str, Any], agreement_doc: dict[str, Any]
) -> dict[str, Any]:
    consensus = agreement_doc["consensus"]
    keys = {row["pair_id"]: row for row in key_doc["key"]}
    if set(consensus) != set(keys):
        raise ValueError("consensus and key pair IDs differ")
    buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair_id, decision in consensus.items():
        buckets[keys[pair_id]["comparison"]].append((keys[pair_id], decision))

    output = {}
    for comparison in key_doc["comparisons"]:
        rows = buckets[comparison]
        system_a, system_b = rows[0][0]["system_a"], rows[0][0]["system_b"]

        def relative(field: str) -> dict[str, Any]:
            counts = Counter()
            for key, decision in rows:
                label = decision[field]
                if label == "unresolved":
                    counts["unresolved"] += 1
                elif label == "Tie":
                    counts["tie"] += 1
                else:
                    winner = key[f"group_{label}_system"]
                    counts["win" if winner == system_a else "loss"] += 1
            n = len(rows)
            resolved = n - counts["unresolved"]
            decisive = counts["win"] + counts["loss"]
            return {
                "wins": counts["win"],
                "losses": counts["loss"],
                "ties": counts["tie"],
                "unresolved": counts["unresolved"],
                "neutral_imputed_score": (
                    counts["win"] + 0.5 * (counts["tie"] + counts["unresolved"])
                )
                / n,
                "resolved_only_score": (
                    (counts["win"] + 0.5 * counts["tie"]) / resolved
                    if resolved
                    else None
                ),
                "decisive_win_rate": counts["win"] / decisive if decisive else None,
                "decisive_win_rate_95ci_wilson": (
                    wilson(counts["win"], decisive) if decisive else [0.0, 1.0]
                ),
            }

        absolute_counts = {
            system: Counter({"good": 0, "weak": 0, "bad": 0, "unresolved": 0})
            for system in (system_a, system_b)
        }
        for key, decision in rows:
            for side in "AB":
                system = key[f"group_{side}_system"]
                absolute_counts[system][decision[f"absolute_{side}"]] += 1
        output[comparison] = {
            "system_a": system_a,
            "system_b": system_b,
            "trials": len(rows),
            "overall": relative("overall"),
            "best_pick": relative("best_pick"),
            "absolute_quality_counts": {
                system: dict(counts) for system, counts in absolute_counts.items()
            },
        }
    return {
        "raters": agreement_doc["raters"],
        "trials": agreement_doc["trials"],
        "comparisons": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--agreement-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key_doc = json.loads(args.key.read_text(encoding="utf-8"))
    agreement_doc = json.loads(args.agreement_report.read_text(encoding="utf-8"))
    report = summarize_consensus(key_doc, agreement_doc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
