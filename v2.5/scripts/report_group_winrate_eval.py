#!/usr/bin/env python3
"""Unblind fixed group-comparison decisions and report win rates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl


def wilson(wins: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = wins / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [centre - margin, centre + margin]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public = {row["pair_id"]: row for row in read_jsonl(args.public)}
    key_doc = json.load(args.key.open())
    keys = {row["pair_id"]: row for row in key_doc["key"]}
    decision_doc = json.load(args.decisions.open())
    decisions = decision_doc["decisions"]
    if set(public) != set(keys) or set(public) != set(decisions):
        raise ValueError("public, key, and decision pair IDs differ")

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    position = {
        "overall_A": 0,
        "overall_B": 0,
        "overall_Tie": 0,
        "best_pick_A": 0,
        "best_pick_B": 0,
        "best_pick_Tie": 0,
    }
    for pair_id, decision in decisions.items():
        if decision.get("overall") not in {"A", "B", "Tie"} or decision.get(
            "best_pick"
        ) not in {"A", "B", "Tie"}:
            raise ValueError(f"{pair_id}: decisions must be A, B, or Tie")
        if decision.get("absolute_A") not in {"good", "weak", "bad"} or decision.get(
            "absolute_B"
        ) not in {"good", "weak", "bad"}:
            raise ValueError(f"{pair_id}: absolute labels must be good, weak, or bad")
        key = keys[pair_id]
        position[f"overall_{decision['overall']}"] += 1
        position[f"best_pick_{decision['best_pick']}"] += 1
        row = {**public[pair_id], **key, **decision}
        row["overall_winner"] = (
            None
            if decision["overall"] == "Tie"
            else key[f"group_{decision['overall']}_system"]
        )
        row["best_pick_winner"] = (
            None
            if decision["best_pick"] == "Tie"
            else key[f"group_{decision['best_pick']}_system"]
        )
        row["absolute_quality"] = {
            key["group_A_system"]: decision["absolute_A"],
            key["group_B_system"]: decision["absolute_B"],
        }
        buckets[key["comparison"]].append(row)

    summary = {}
    for comparison in key_doc["comparisons"]:
        rows = buckets[comparison]
        system_a = rows[0]["system_a"]
        system_b = rows[0]["system_b"]
        n = len(rows)
        overall_wins = sum(row["overall_winner"] == system_a for row in rows)
        overall_losses = sum(row["overall_winner"] == system_b for row in rows)
        overall_ties = n - overall_wins - overall_losses
        best_wins = sum(row["best_pick_winner"] == system_a for row in rows)
        best_losses = sum(row["best_pick_winner"] == system_b for row in rows)
        best_ties = n - best_wins - best_losses
        absolute_counts = {
            system: {
                label: sum(row["absolute_quality"][system] == label for row in rows)
                for label in ("good", "weak", "bad")
            }
            for system in (system_a, system_b)
        }
        summary[comparison] = {
            "system_a": system_a,
            "system_b": system_b,
            "images": n,
            "overall_wins": overall_wins,
            "overall_losses": overall_losses,
            "overall_ties": overall_ties,
            "overall_win_rate": overall_wins / n,
            "overall_win_rate_95ci_wilson": wilson(overall_wins, n),
            "overall_tie_adjusted_rate": (overall_wins + 0.5 * overall_ties) / n,
            "overall_decisive_win_rate": (
                overall_wins / (overall_wins + overall_losses)
                if overall_wins + overall_losses
                else None
            ),
            "overall_decisive_win_rate_95ci_wilson": (
                wilson(overall_wins, overall_wins + overall_losses)
                if overall_wins + overall_losses
                else [0.0, 1.0]
            ),
            "best_pick_wins": best_wins,
            "best_pick_losses": best_losses,
            "best_pick_ties": best_ties,
            "best_pick_win_rate": best_wins / n,
            "best_pick_win_rate_95ci_wilson": wilson(best_wins, n),
            "best_pick_tie_adjusted_rate": (best_wins + 0.5 * best_ties) / n,
            "best_pick_decisive_win_rate": (
                best_wins / (best_wins + best_losses)
                if best_wins + best_losses
                else None
            ),
            "absolute_quality_counts": absolute_counts,
        }
    report = {
        "protocol": decision_doc.get("protocol"),
        "judge": decision_doc.get("judge"),
        "trials": len(public),
        "position_choices": position,
        "comparisons": summary,
        "unblinded_trials": sorted(
            [row for rows in buckets.values() for row in rows],
            key=lambda row: (row["comparison"], row["image_id"]),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "unblinded_trials"}, indent=2))


if __name__ == "__main__":
    main()
