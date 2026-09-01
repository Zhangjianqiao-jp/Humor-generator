#!/usr/bin/env python3
"""Render a dependency-light live dashboard for controlled DPO scaling."""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series: list[tuple[str, list[tuple[float, float]], str]],
    title: str,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=16, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((left + 18, top + 14), title, fill="#0f2747")
    plot = (left + 62, top + 52, right - 22, bottom - 46)
    x0, y0, x1, y1 = plot
    points = [point for _, values, _ in series for point in values]
    if not points:
        draw.text((x0, y0 + 35), "Waiting for metrics...", fill="#64748b")
        return
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    padding = max((ymax - ymin) * 0.12, 1e-6)
    ymin, ymax = ymin - padding, ymax + padding
    draw.line((x0, y1, x1, y1), fill="#94a3b8", width=2)
    draw.line((x0, y0, x0, y1), fill="#94a3b8", width=2)
    draw.text((x0, y1 + 8), f"step {int(xmin)}", fill="#64748b")
    draw.text((x1 - 72, y1 + 8), f"{int(xmax)}", fill="#64748b")
    draw.text((left + 5, y0), f"{ymax:.4f}", fill="#64748b")
    draw.text((left + 5, y1 - 12), f"{ymin:.4f}", fill="#64748b")
    for offset, (label, values, color) in enumerate(series):
        coords = []
        for x, y in values:
            px = x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
            py = y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)
            coords.append((px, py))
        if len(coords) > 1:
            draw.line(coords, fill=color, width=4)
        for point in coords:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
        draw.text((left + 185 + offset * 210, top + 15), label, fill=color)


def render_png(path: Path, status: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    canvas = Image.new("RGB", (1600, 1000), "#f1f5f9")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((48, 30), "QUALITY64 CONTROLLED DPO SCALING", fill="#0b2545", font=font)
    state = str(status.get("state", "waiting"))
    step, total = int(status.get("step", 0)), int(status.get("total_steps", 2163))
    progress = min(1.0, step / max(total, 1))
    draw.rounded_rectangle((48, 70, 1550, 156), radius=14, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((70, 88), f"State: {state}    Step: {step}/{total}    Best: {status.get('best_step', 0)}", fill="#0f2747")
    draw.rounded_rectangle((70, 120, 1525, 140), radius=8, fill="#e2e8f0")
    draw.rounded_rectangle((70, 120, 70 + int(1455 * progress), 140), radius=8, fill="#0f766e")

    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if str(row.get("split", "")).startswith("validation")]
    line_chart(
        draw,
        (48, 184, 780, 555),
        [("train loss", [(float(r["step"]), float(r["loss"])) for r in train], "#d97706")],
        "Training loss (logged batches)",
    )
    line_chart(
        draw,
        (820, 184, 1550, 555),
        [
            ("pair loss", [(float(r["step"]), float(r["eval_loss"])) for r in validation], "#1d4ed8"),
            (
                "image mean",
                [(float(r["step"]), float(r["eval_image_mean_loss"])) for r in validation],
                "#0f766e",
            ),
        ],
        "Validation DPO loss",
    )
    line_chart(
        draw,
        (48, 590, 780, 960),
        [
            (
                "reward accuracy",
                [(float(r["step"]), float(r["eval_image_mean_reward_accuracy"])) for r in validation],
                "#7c3aed",
            ),
            (
                "policy accuracy",
                [(float(r["step"]), float(r["eval_image_mean_policy_accuracy"])) for r in validation],
                "#dc2626",
            ),
        ],
        "Image-clustered preference accuracy",
    )
    line_chart(
        draw,
        (820, 590, 1550, 960),
        [
            (
                "chosen / token",
                [(float(r["step"]), float(r["eval_image_mean_chosen_logp_per_token"])) for r in validation],
                "#0f766e",
            ),
            (
                "rejected / token",
                [(float(r["step"]), float(r["eval_image_mean_rejected_logp_per_token"])) for r in validation],
                "#be123c",
            ),
        ],
        "Absolute policy log-probability",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    canvas.save(temporary)
    temporary.replace(path)


def render_html(path: Path, status: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    validation = [row for row in rows if str(row.get("split", "")).startswith("validation")]
    table_rows = []
    for row in validation:
        table_rows.append(
            "<tr>"
            f"<td>{int(row.get('step', 0))}</td>"
            f"<td>{html.escape(str(row.get('split', '')))}</td>"
            f"<td>{float(row.get('eval_image_mean_loss', float('nan'))):.6f}</td>"
            f"<td>{float(row.get('eval_image_mean_reward_accuracy', float('nan'))):.3%}</td>"
            f"<td>{float(row.get('eval_image_mean_reward_margin', float('nan'))):.6f}</td>"
            f"<td>{float(row.get('eval_image_mean_chosen_logp_per_token', float('nan'))):.5f}</td>"
            "</tr>"
        )
    payload = html.escape(json.dumps(status, ensure_ascii=False, indent=2))
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="15"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quality64 DPO 监控</title><style>
body{{margin:0;background:#eef2f7;color:#152238;font:15px/1.5 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:22px}}
.card{{background:white;border:1px solid #cbd5e1;border-radius:12px;padding:16px;margin-bottom:16px}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #e2e8f0;text-align:right}}th:first-child,td:first-child{{text-align:left}}
pre{{white-space:pre-wrap}}h1{{color:#0b2545}}</style></head><body><main>
<h1>Quality64 受控 DPO Scaling 实时监控</h1><div class="card">页面每15秒自动刷新。Test47 保持封存。</div>
<div class="card"><img src="monitor.png?t={time.time_ns()}" alt="DPO curves"></div>
<div class="card"><h2>阶段验证</h2><table><thead><tr><th>Step</th><th>Split</th><th>Image loss</th><th>Reward acc.</th><th>Margin</th><th>Chosen logp/token</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<div class="card"><h2>运行状态</h2><pre>{payload}</pre></div></main></body></html>"""
    atomic_text(path, document)


def build(run_dir: Path, output_dir: Path) -> str:
    status = read_json(run_dir / "run_status.json")
    rows = read_metrics(run_dir / "train_metrics.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    render_png(output_dir / "monitor.png", status, rows)
    render_html(output_dir / "index.html", status, rows)
    atomic_text(output_dir / "status.json", json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    return str(status.get("state", "waiting"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()
    while True:
        state = build(args.run_dir, args.output_dir)
        if not args.watch or state in {"complete", "early_stopped", "failed"}:
            return
        time.sleep(max(2, args.interval))


if __name__ == "__main__":
    main()
