from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

MODULE_PATTERN = re.compile(r"(?:layers|blocks)\.(\d+).*?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
EMOJI_PATTERN = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
MEME_PATTERNS = {
    "pov": re.compile(r"\bpov\b", re.IGNORECASE),
    "bro": re.compile(r"\bbro\b", re.IGNORECASE),
    "meanwhile": re.compile(r"\bmeanwhile\b", re.IGNORECASE),
    "internet_slang": re.compile(r"\b(?:lol|lmao|rofl|sus|based|cringe|bruh)\b", re.IGNORECASE),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def derangement(keys: list[str], seed: int) -> dict[str, str]:
    if len(keys) < 2 or len(keys) != len(set(keys)):
        raise ValueError("derangement requires at least two unique keys")
    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)
    donors = shuffled[1:] + shuffled[:1]
    mapping = dict(zip(shuffled, donors))
    if set(mapping) != set(mapping.values()) or any(key == donor for key, donor in mapping.items()):
        raise AssertionError("failed to construct deterministic derangement")
    return mapping


def select_image_diverse_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Keep the first row per image before applying an optional limit."""
    if limit is None:
        return rows
    selected = []
    seen = set()
    for row in rows:
        image_id = str(row["image_id"])
        if image_id in seen:
            continue
        seen.add(image_id)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def text_features(text: str) -> dict[str, float | int]:
    words = re.findall(r"\b\w+(?:['’]\w+)?\b", text.lower())
    return {
        "chars": len(text.strip()),
        "tokens_whitespace": len(text.split()),
        "words": len(words),
        "emoji_count": len(EMOJI_PATTERN.findall(text)),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "quote_count": text.count('"') + text.count("“") + text.count("”"),
        "pov": int(bool(MEME_PATTERNS["pov"].search(text))),
        "bro": int(bool(MEME_PATTERNS["bro"].search(text))),
        "meanwhile": int(bool(MEME_PATTERNS["meanwhile"].search(text))),
        "internet_slang": int(bool(MEME_PATTERNS["internet_slang"].search(text))),
        "lexical_diversity": len(set(words)) / len(words) if words else 0.0,
    }


def module_identity(name: str) -> tuple[int | None, str]:
    match = MODULE_PATTERN.search(name)
    if match:
        return int(match.group(1)), match.group(2)
    for module in ("vision", "visual", "merger", "projector", "lm_head"):
        if module in name:
            return None, module
    return None, name.rsplit(".", maxsplit=1)[-1]


def line_plot_png(
    path: Path,
    x_values: list[float],
    series: dict[str, list[float]],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 1100, 700
    left, right, top, bottom = 100, 40, 80, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot_w, plot_h = width - left - right, height - top - bottom
    finite = [value for values in series.values() for value in values if math.isfinite(value)]
    y_min, y_max = (min(finite), max(finite)) if finite else (0.0, 1.0)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    x_min, x_max = min(x_values), max(x_values)
    draw.line((left, top, left, top + plot_h), fill="#17365d", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#17365d", width=2)
    draw.text((left, 25), title, fill="#10253f", font=font)
    draw.text((width // 2 - 30, height - 30), x_label, fill="#10253f", font=font)
    draw.text((10, top + plot_h // 2), y_label, fill="#10253f", font=font)
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = top + plot_h - plot_h * tick / 5
        draw.line((left - 5, y, left + plot_w, y), fill="#e5e7eb", width=1)
        draw.text((10, y - 6), f"{value:.3g}", fill="#475569", font=font)
    palette = ["#006d77", "#e76f51", "#3a5a98", "#7b2cbf"]
    for index, (label, values) in enumerate(series.items()):
        color = palette[index % len(palette)]
        points = []
        for x, value in zip(x_values, values):
            px = left + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w
            py = top + plot_h - (value - y_min) / (y_max - y_min) * plot_h
            points.append((px, py))
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
            draw.text((px - 8, top + plot_h + 12), str(int(x)), fill="#475569", font=font)
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        draw.rectangle((left + index * 180, 52, left + index * 180 + 14, 66), fill=color)
        draw.text((left + index * 180 + 20, 52), label, fill="#10253f", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def histogram_png(path: Path, values: list[float], title: str, x_label: str, bins: int = 20) -> None:
    width, height = 1000, 650
    left, right, top, bottom = 90, 35, 75, 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot_w, plot_h = width - left - right, height - top - bottom
    finite = [value for value in values if math.isfinite(value)]
    low, high = (min(finite), max(finite)) if finite else (-1.0, 1.0)
    if low == high:
        low -= 0.5
        high += 0.5
    counts = [0] * bins
    for value in finite:
        index = min(bins - 1, int((value - low) / (high - low) * bins))
        counts[index] += 1
    max_count = max(counts, default=1) or 1
    bar_w = plot_w / bins
    for index, count in enumerate(counts):
        x0 = left + index * bar_w
        y0 = top + plot_h - count / max_count * plot_h
        draw.rectangle((x0 + 1, y0, x0 + bar_w - 1, top + plot_h), fill="#2a9d8f")
    if low <= 0 <= high:
        zero_x = left + (0 - low) / (high - low) * plot_w
        draw.line((zero_x, top, zero_x, top + plot_h), fill="#c1121f", width=3)
    draw.line((left, top, left, top + plot_h), fill="#17365d", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#17365d", width=2)
    draw.text((left, 25), title, fill="#10253f", font=font)
    draw.text((width // 2 - 40, height - 30), x_label, fill="#10253f", font=font)
    draw.text((left, top + plot_h + 12), f"{low:.3g}", fill="#475569", font=font)
    draw.text((left + plot_w - 30, top + plot_h + 12), f"{high:.3g}", fill="#475569", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
