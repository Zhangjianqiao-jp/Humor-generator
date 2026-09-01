#!/usr/bin/env python3
"""Validate latent-communication generations and plot non-judgmental diagnostics.

This script intentionally does not infer humor quality.  It checks generation
integrity and possible surface-form regressions before independent blind rating.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\w\s]", re.UNICODE)
GENERIC_RE = re.compile(
    r"(?:\bPOV\b|\bBro\b|\bMeanwhile\b|\bNobody:\b|\bWhen you\b|💀|😂)",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def distinct_ngrams(texts: list[str], n: int) -> float:
    grams: list[tuple[str, ...]] = []
    for text in texts:
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        grams.extend(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return len(set(grams)) / len(grams) if grams else 0.0


def save_diagnostics_png(systems: dict[str, dict], output: Path) -> None:
    """Draw a dependency-light four-panel bar chart with Pillow."""
    width, height = 1800, 1000
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    names = list(systems)
    metrics = [
        "empty_rate",
        "within_group_duplicate_rate",
        "generic_template_rate",
        "distinct_2",
    ]
    colors = {"text": "#1f4e79", "latent": "#2e8b57", "hybrid": "#d47f00"}
    draw.text((40, 18), "Text / Latent / Hybrid surface diagnostics (not humor judgments)", fill="black", font=font)
    for panel_index, metric in enumerate(metrics):
        column, row = panel_index % 2, panel_index // 2
        left, top = 55 + column * 885, 75 + row * 450
        plot_width, plot_height = 800, 330
        bottom = top + plot_height
        draw.text((left, top - 25), metric, fill="black", font=font)
        draw.line((left, top, left, bottom), fill="#333333", width=2)
        draw.line((left, bottom, left + plot_width, bottom), fill="#333333", width=2)
        for tick in range(6):
            value = tick / 5
            y = bottom - int(value * plot_height)
            draw.line((left, y, left + plot_width, y), fill="#dddddd", width=1)
            draw.text((left - 42, y - 6), f"{value:.1f}", fill="#555555", font=font)
        slot = plot_width / len(names)
        for index, name in enumerate(names):
            value = max(0.0, min(1.0, float(systems[name][metric])))
            x0 = left + int(index * slot + slot * 0.16)
            x1 = left + int((index + 1) * slot - slot * 0.16)
            y0 = bottom - int(value * plot_height)
            mode = name.split("_seed", 1)[0]
            draw.rectangle((x0, y0, x1, bottom), fill=colors.get(mode, "#777777"))
            draw.text((x0, y0 - 16), f"{value:.2f}", fill="black", font=font)
            short_seed = name.rsplit("seed", 1)[-1][-2:]
            draw.text((x0, bottom + 8), f"{mode[:1].upper()}-{short_seed}", fill="black", font=font)
    draw.text((55, 970), "T=text, L=latent, H=hybrid; suffix is generation-seed ending", fill="#444444", font=font)
    canvas.save(output)


def summarize(rows: list[dict], expected_images: int, expected_candidates: int) -> dict:
    if len(rows) != expected_images:
        raise ValueError(f"expected {expected_images} image rows, found {len(rows)}")
    image_ids = [str(row.get("image_id", "")).strip() for row in rows]
    if any(not image_id for image_id in image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("missing or duplicate image_id")

    groups: list[list[str]] = []
    for row in rows:
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != expected_candidates:
            raise ValueError(
                f"{row.get('image_id')}: expected {expected_candidates} candidates"
            )
        groups.append([str(candidate).strip() for candidate in candidates])
    texts = [text for group in groups for text in group]
    normalized = [" ".join(text.lower().split()) for text in texts]
    empty = sum(not text for text in texts)
    duplicate_groups = sum(len(set(" ".join(x.lower().split()) for x in group)) < len(group) for group in groups)
    return {
        "images": len(rows),
        "captions": len(texts),
        "empty_rate": empty / len(texts),
        "mean_characters": mean(len(text) for text in texts),
        "mean_tokens": mean(len(TOKEN_RE.findall(text)) for text in texts),
        "unique_caption_rate": len(set(normalized)) / len(normalized),
        "within_group_duplicate_rate": duplicate_groups / len(groups),
        "generic_template_rate": sum(bool(GENERIC_RE.search(text)) for text in texts) / len(texts),
        "distinct_1": distinct_ngrams(texts, 1),
        "distinct_2": distinct_ngrams(texts, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", default=["text", "latent", "hybrid"])
    parser.add_argument("--expected-images", type=int, default=24)
    parser.add_argument("--expected-candidates", type=int, default=3)
    args = parser.parse_args()

    manifest_path = args.input_dir / "generation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test47_read") is not False:
        raise ValueError("sealed-test gate failed: test47_read must be false")
    if int(manifest.get("images", -1)) != args.expected_images:
        raise ValueError("generation manifest image count mismatch")

    report: dict[str, object] = {
        "scope": "surface diagnostics only; humor quality requires blinded rating",
        "manifest": manifest,
        "systems": {},
    }
    systems: dict[str, dict] = {}
    for mode in args.modes:
        for seed in args.seeds:
            name = f"{mode}_seed{seed}"
            systems[name] = summarize(
                read_jsonl(args.input_dir / f"{name}.jsonl"),
                args.expected_images,
                args.expected_candidates,
            )
    report["systems"] = systems
    report["integrity_gate"] = {
        "passed": all(
            values["empty_rate"] == 0.0
            and values["within_group_duplicate_rate"] <= 0.25
            for values in systems.values()
        ),
        "note": "Passing this gate is necessary but not evidence that captions are funny.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "surface_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    save_diagnostics_png(systems, args.output_dir / "surface_diagnostics.png")

    lines = [
        "# Latent Communication 五小时自动审计",
        "",
        "本报告只检查生成完整性、重复、模板化和词汇多样性；不把这些代理指标解释为幽默质量。",
        "",
        f"- 完整性门禁：{'PASS' if report['integrity_gate']['passed'] else 'FAIL'}",
        f"- 图片数：{args.expected_images}",
        f"- Seeds：{', '.join(map(str, args.seeds))}",
        "",
        "| System | Empty | Duplicate-group | Generic-template | Distinct-2 | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in systems.items():
        lines.append(
            f"| {name} | {values['empty_rate']:.3f} | "
            f"{values['within_group_duplicate_rate']:.3f} | "
            f"{values['generic_template_rate']:.3f} | {values['distinct_2']:.3f} | "
            f"{values['mean_tokens']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 下一门禁",
            "",
            "必须完成匿名 Group-of-3 独立盲评，才能决定是否进入通信条件 DPO。",
        ]
    )
    (args.output_dir / "FIVE_HOUR_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["integrity_gate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
