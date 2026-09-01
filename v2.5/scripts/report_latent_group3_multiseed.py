#!/usr/bin/env python3
"""Aggregate three-system blind Group-of-3 reports with image-clustered CIs."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def cluster_ci(image_scores: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(image_scores)
    draws = [sum(image_scores[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def score(row: dict, field: str, preferred: str) -> float:
    winner = row[f"{field}_winner"]
    return 0.5 if winner is None else float(winner == preferred)


def draw_report(report: dict, output: Path) -> None:
    canvas = Image.new("RGB", (1800, 920), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    draw.text((50, 25), "Latent communication: blind Group-of-3 validation", fill="#102a43", font=font)
    comparisons = list(report["comparisons"].items())
    for panel, metric in enumerate(("overall", "best_pick")):
        left, top = 80 + panel * 870, 100
        draw.text((left, top - 30), f"{metric} tie-adjusted rate (system A over B)", fill="black", font=font)
        for index, (name, values) in enumerate(comparisons):
            y = top + index * 180
            rate = values[metric]["tie_adjusted_rate"]
            lo, hi = values[metric]["image_clustered_95ci"]
            draw.text((left, y), name.replace("_vs_", " vs "), fill="black", font=font)
            draw.rectangle((left, y + 35, left + 720, y + 75), outline="#cccccc")
            draw.rectangle((left, y + 35, left + int(720 * rate), y + 75), fill="#2f855a")
            draw.line((left + int(720 * lo), y + 28, left + int(720 * lo), y + 82), fill="#c53030", width=3)
            draw.line((left + int(720 * hi), y + 28, left + int(720 * hi), y + 82), fill="#c53030", width=3)
            draw.line((left + 360, y + 25, left + 360, y + 85), fill="#102a43", width=2)
            draw.text((left, y + 90), f"rate={rate:.3f}, CI=[{lo:.3f}, {hi:.3f}], seed SD={values[metric]['seed_std']:.3f}", fill="#334e68", font=font)
    draw.text((80, 740), "Absolute group quality across 24 images x 3 seeds", fill="black", font=font)
    x = 80
    colors = {"good": "#2f855a", "weak": "#d69e2e", "bad": "#c53030"}
    for system, values in report["absolute_quality"].items():
        draw.text((x, 785), system, fill="black", font=font)
        base = x
        for label in ("good", "weak", "bad"):
            width = int(420 * values["rates"][label])
            draw.rectangle((base, 820, base + width, 865), fill=colors[label])
            base += width
        draw.text((x, 875), f"good={values['rates']['good']:.1%}, majority-good images={values['majority_good_images']}/24", fill="#334e68", font=font)
        x += 560
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()

    reports = {
        seed: json.loads((args.input_dir / f"codex_unblinded_report_seed{seed}.json").read_text())
        for seed in args.seeds
    }
    comparison_names = list(next(iter(reports.values()))["comparisons"])
    result = {
        "protocol": "Independent blinded Group-of-3; three generation seeds; image is the statistical unit",
        "judge": "Codex",
        "images": 24,
        "seeds": args.seeds,
        "pair_trials": 24 * len(args.seeds) * len(comparison_names),
        "comparisons": {},
    }
    absolute: dict[tuple[int, str, str], str] = {}
    for name in comparison_names:
        all_rows = []
        per_seed_rows = {}
        for seed, report in reports.items():
            rows = [row for row in report["unblinded_trials"] if row["comparison"] == name]
            per_seed_rows[seed] = rows
            all_rows.extend(rows)
            for row in rows:
                for system, label in row["absolute_quality"].items():
                    key = (seed, row["image_id"], system)
                    if key in absolute and absolute[key] != label:
                        raise ValueError(f"inconsistent absolute label for {key}")
                    absolute[key] = label
        system_a = all_rows[0]["system_a"]
        system_b = all_rows[0]["system_b"]
        by_image = defaultdict(list)
        for row in all_rows:
            by_image[row["image_id"]].append(row)
        metrics = {}
        for field in ("overall", "best_pick"):
            seed_rates = [
                statistics.mean(score(row, field, system_a) for row in per_seed_rows[seed])
                for seed in args.seeds
            ]
            image_scores = [
                statistics.mean(score(row, field, system_a) for row in rows)
                for rows in by_image.values()
            ]
            wins = sum(row[f"{field}_winner"] == system_a for row in all_rows)
            losses = sum(row[f"{field}_winner"] == system_b for row in all_rows)
            ties = len(all_rows) - wins - losses
            metrics[field] = {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "tie_adjusted_rate": statistics.mean(image_scores),
                "image_clustered_95ci": cluster_ci(
                    image_scores, args.bootstrap_samples, 20260829 + len(result["comparisons"]) * 10 + len(metrics)
                ),
                "per_seed_rates": dict(zip(map(str, args.seeds), seed_rates)),
                "seed_mean": statistics.mean(seed_rates),
                "seed_std": statistics.stdev(seed_rates),
            }
        result["comparisons"][name] = {"system_a": system_a, "system_b": system_b, **metrics}

    systems = sorted({key[2] for key in absolute})
    absolute_report = {}
    for system in systems:
        values = {(seed, image): label for (seed, image, name), label in absolute.items() if name == system}
        counts = Counter(values.values())
        images = sorted({image for _, image in values})
        majority_good = sum(sum(values[(seed, image)] == "good" for seed in args.seeds) >= 2 for image in images)
        absolute_report[system] = {
            "counts": {label: counts[label] for label in ("good", "weak", "bad")},
            "rates": {label: counts[label] / len(values) for label in ("good", "weak", "bad")},
            "majority_good_images": majority_good,
            "majority_good_rate": majority_good / len(images),
        }
    result["absolute_quality"] = absolute_report
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Text / Latent / Hybrid 三种通信方式盲评汇总",
        "",
        f"- 图片：24；generation seeds：{', '.join(map(str, args.seeds))}；pair trials：{result['pair_trials']}。",
        "- 统计单位是图片；Tie 计 0.5；95% CI 使用 image-clustered bootstrap。",
        "- 所有 group-level judgments 在读取 private key 前冻结。",
        "",
        "| Comparison (A vs B) | Overall W-L-T | Overall rate [95% CI] | Seed SD | Best-pick rate [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in result["comparisons"].items():
        overall, best = values["overall"], values["best_pick"]
        lines.append(
            f"| {name} | {overall['wins']}-{overall['losses']}-{overall['ties']} | "
            f"{overall['tie_adjusted_rate']:.1%} [{overall['image_clustered_95ci'][0]:.1%}, {overall['image_clustered_95ci'][1]:.1%}] | "
            f"{overall['seed_std']:.1%} | {best['tie_adjusted_rate']:.1%} "
            f"[{best['image_clustered_95ci'][0]:.1%}, {best['image_clustered_95ci'][1]:.1%}] |"
        )
    lines.extend(["", "## 绝对质量", "", "| System | Good | Weak | Bad | Majority-good images |", "|---|---:|---:|---:|---:|"])
    for system, values in absolute_report.items():
        lines.append(
            f"| {system} | {values['counts']['good']}/72 ({values['rates']['good']:.1%}) | "
            f"{values['counts']['weak']}/72 ({values['rates']['weak']:.1%}) | "
            f"{values['counts']['bad']}/72 ({values['rates']['bad']:.1%}) | "
            f"{values['majority_good_images']}/24 ({values['majority_good_rate']:.1%}) |"
        )
    args.output_md.write_text("\n".join(lines) + "\n")
    draw_report(result, args.output_png)


if __name__ == "__main__":
    main()
