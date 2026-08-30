#!/usr/bin/env python3
"""Validate independently re-blinded annotations and report agreement."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any


FIELDS = ("overall", "best_pick", "best_A_index", "best_B_index", "absolute_A", "absolute_B")


def cohen_kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs must be non-empty and aligned")
    labels = set(left) | set(right)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    lc, rc = Counter(left), Counter(right)
    expected = sum(lc[x] * rc[x] for x in labels) / (len(left) ** 2)
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)


def fleiss_kappa(ratings: list[list[str]]) -> float:
    if not ratings or len(ratings[0]) < 2 or any(len(row) != len(ratings[0]) for row in ratings):
        raise ValueError("Fleiss kappa requires aligned items and at least two raters")
    labels = sorted({value for row in ratings for value in row})
    n = len(ratings[0])
    item_agreement = []
    totals = Counter()
    for row in ratings:
        counts = Counter(row); totals.update(counts)
        item_agreement.append((sum(counts[label] ** 2 for label in labels) - n) / (n * (n - 1)))
    observed = sum(item_agreement) / len(item_agreement)
    expected = sum((totals[label] / (len(ratings) * n)) ** 2 for label in labels)
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)


def exact_two_sided_binomial(left: int, right: int) -> float:
    n = left + right
    if n == 0:
        return 1.0
    observed = min(left, right)
    tail = sum(math.comb(n, k) for k in range(observed + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def canonicalize(doc: dict[str, Any], key_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    decisions = doc.get("decisions", {})
    key = {row["blind_id"]: row for row in key_rows}
    if set(decisions) != set(key):
        raise ValueError(f"{doc.get('rater_id')}: missing or extra blind IDs")
    out = {}
    for blind_id, d in decisions.items():
        if d.get("overall") not in {"A", "B", "Tie"} or d.get("best_pick") not in {"A", "B", "Tie"}:
            raise ValueError(f"{blind_id}: incomplete relative judgment")
        if any(d.get(f"absolute_{s}") not in {"good", "weak", "bad"} for s in "AB"):
            raise ValueError(f"{blind_id}: incomplete absolute judgment")
        if any(d.get(f"best_{s}_index") not in {1, 2, 3} for s in "AB"):
            raise ValueError(f"{blind_id}: incomplete best index")
        swapped = key[blind_id]["swapped"]
        def side(value: str) -> str:
            return value if value == "Tie" or not swapped else ("B" if value == "A" else "A")
        out[key[blind_id]["original_pair_id"]] = {
            "overall": side(d["overall"]), "best_pick": side(d["best_pick"]),
            "absolute_A": d["absolute_B"] if swapped else d["absolute_A"],
            "absolute_B": d["absolute_A"] if swapped else d["absolute_B"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--decision", action="append", type=Path, required=True)
    parser.add_argument(
        "--canonical-decision", action="append", default=[], metavar="RATER=JSON",
        help="Previously locked decisions already keyed by original pair_id.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key_doc = json.load(args.private_key.open(encoding="utf-8"))
    raters = {}
    position_choices = {}
    for path in args.decision:
        doc = json.load(path.open(encoding="utf-8")); rid = doc.get("rater_id")
        if rid not in key_doc["raters"] or rid in raters:
            raise ValueError(f"unknown or duplicate rater_id {rid!r}")
        position_choices[rid] = {}
        for field in ("overall", "best_pick"):
            counts = Counter(row.get(field) for row in doc.get("decisions", {}).values())
            position_choices[rid][field] = {
                "A": counts["A"], "B": counts["B"], "Tie": counts["Tie"],
                "two_sided_binomial_p_excluding_ties": exact_two_sided_binomial(counts["A"], counts["B"]),
            }
        raters[rid] = canonicalize(doc, key_doc["raters"][rid])
    original_ids = {
        row["original_pair_id"]
        for rows in key_doc["raters"].values()
        for row in rows
    }
    for spec in args.canonical_decision:
        if "=" not in spec:
            raise ValueError("--canonical-decision must be RATER=JSON")
        rid, raw_path = spec.split("=", 1)
        doc = json.load(Path(raw_path).open(encoding="utf-8"))
        decisions = doc.get("decisions", {})
        if rid in raters or set(decisions) != original_ids:
            raise ValueError(f"invalid duplicate rater or pair IDs for {rid!r}")
        rows = {}
        for pair_id, d in decisions.items():
            if d.get("overall") not in {"A", "B", "Tie"} or d.get("best_pick") not in {"A", "B", "Tie"}:
                raise ValueError(f"{pair_id}: incomplete canonical relative judgment")
            if any(d.get(f"absolute_{s}") not in {"good", "weak", "bad"} for s in "AB"):
                raise ValueError(f"{pair_id}: incomplete canonical absolute judgment")
            rows[pair_id] = {field: d[field] for field in ("overall", "best_pick", "absolute_A", "absolute_B")}
        raters[rid] = rows
        position_choices[rid] = {}
        for field in ("overall", "best_pick"):
            counts = Counter(row[field] for row in rows.values())
            position_choices[rid][field] = {
                "A": counts["A"], "B": counts["B"], "Tie": counts["Tie"],
                "two_sided_binomial_p_excluding_ties": exact_two_sided_binomial(counts["A"], counts["B"]),
            }
    if len(raters) < 2:
        raise ValueError("at least two independent raters are required")
    ids = set(next(iter(raters.values())))
    if any(set(rows) != ids for rows in raters.values()):
        raise ValueError("raters do not cover the same original trials")
    agreement = {}
    for a, b in combinations(sorted(raters), 2):
        agreement[f"{a}_vs_{b}"] = {
            field: {
                "raw": sum(raters[a][i][field] == raters[b][i][field] for i in ids) / len(ids),
                "cohen_kappa": cohen_kappa([raters[a][i][field] for i in ids], [raters[b][i][field] for i in ids]),
            } for field in ("overall", "best_pick", "absolute_A", "absolute_B")
        }
    consensus = {}
    for pair_id in sorted(ids):
        consensus[pair_id] = {}
        for field in ("overall", "best_pick", "absolute_A", "absolute_B"):
            counts = Counter(rows[pair_id][field] for rows in raters.values())
            top = counts.most_common()
            consensus[pair_id][field] = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else "unresolved"
    multi_rater = {
        field: fleiss_kappa([[raters[rid][pair_id][field] for rid in sorted(raters)] for pair_id in sorted(ids)])
        for field in ("overall", "best_pick", "absolute_A", "absolute_B")
    }
    report = {"raters": sorted(raters), "trials": len(ids), "position_choices": position_choices,
              "pairwise_agreement": agreement, "fleiss_kappa": multi_rater,
              "consensus_resolved": {f: sum(v[f] != "unresolved" for v in consensus.values()) for f in ("overall", "best_pick", "absolute_A", "absolute_B")},
              "consensus": consensus}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "consensus"}, indent=2))


if __name__ == "__main__":
    main()
