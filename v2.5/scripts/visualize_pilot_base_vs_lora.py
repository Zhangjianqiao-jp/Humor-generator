#!/usr/bin/env python
from __future__ import annotations

import base64
import html
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs/guided_sft_pipeline/pilot_evaluation"
OUTPUT = EVAL_DIR / "base_vs_lora_gallery.html"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["image_id"]): row for row in read_jsonl(path)}


def parse_judge_response(row: dict[str, Any]) -> dict[str, Any]:
    raw = str(row.get("raw_response") or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def image_data_uri(path: Path) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((720, 520), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=86, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def prompt_context(prompt: str) -> tuple[str, str]:
    description = ""
    cue = ""
    for line in prompt.splitlines():
        if line.startswith("Image description:"):
            description = line.split(":", 1)[1].strip()
        elif line.startswith("Humor cue:"):
            cue = line.split(":", 1)[1].strip()
    return description, cue


def candidate_list(
    candidates: list[str],
    best_index: int | None,
) -> str:
    items = []
    for index, candidate in enumerate(candidates, 1):
        selected = index == best_index
        badge = '<span class="best">盲评最佳</span>' if selected else ""
        css = "candidate selected" if selected else "candidate"
        items.append(
            f'<li class="{css}"><span class="number">{index}</span>'
            f'<span class="caption">{html.escape(str(candidate))}</span>{badge}</li>'
        )
    return "\n".join(items)


def winner_label(winner: str) -> str:
    return {
        "base": "Base 胜",
        "lora": "LoRA 胜",
        "tie": "平局",
    }.get(winner, winner)


def main() -> None:
    base = by_id(EVAL_DIR / "base_candidates.jsonl")
    lora = by_id(EVAL_DIR / "pilot_lora_candidates.jsonl")
    judgments = by_id(EVAL_DIR / "blind_judgments_7b.jsonl")
    shared = sorted(base.keys() & lora.keys() & judgments.keys())
    counts = Counter(str(judgments[key].get("winner_method") or "unknown") for key in shared)

    cards: list[str] = []
    for ordinal, image_id in enumerate(shared, 1):
        base_row = base[image_id]
        lora_row = lora[image_id]
        judgment = judgments[image_id]
        parsed = parse_judge_response(judgment)
        mapping = judgment.get("mapping") or {}

        group_a = str(mapping.get("A") or "")
        group_b = str(mapping.get("B") or "")
        best_a = parsed.get("best_a_index")
        best_b = parsed.get("best_b_index")
        base_best = best_a if group_a == "base" else best_b if group_b == "base" else None
        lora_best = best_a if group_a == "lora" else best_b if group_b == "lora" else None

        winner = str(judgment.get("winner_method") or "unknown")
        description, cue = prompt_context(str(base_row.get("prompt") or ""))
        has_cue = bool(cue)
        image_path = Path(str(base_row["image"]))
        image_uri = image_data_uri(image_path)

        cards.append(
            f"""
<article class="sample" data-winner="{html.escape(winner)}"
         data-cue="{"yes" if has_cue else "no"}">
  <header class="sample-head">
    <div>
      <span class="ordinal">#{ordinal}</span>
      <span class="image-id">{html.escape(image_id)}</span>
    </div>
    <div class="badges">
      <span class="winner {html.escape(winner)}">{winner_label(winner)}</span>
      <span class="cue {"present" if has_cue else "absent"}">
        {"有幽默点" if has_cue else "无幽默点"}
      </span>
    </div>
  </header>

  <div class="context-grid">
    <figure>
      <img src="{image_uri}" alt="{html.escape(image_id)}" loading="lazy">
      <figcaption>{html.escape(str(image_path))}</figcaption>
    </figure>
    <section class="guidance">
      <h3>输入指导</h3>
      <div class="field"><b>Description</b><p>{html.escape(description)}</p></div>
      <div class="field"><b>Humor cue</b><p>{html.escape(cue) if cue else '<em>空</em>'}</p></div>
      <details>
        <summary>完整 prompt</summary>
        <pre>{html.escape(str(base_row.get("prompt") or ""))}</pre>
      </details>
      <div class="judge">
        <b>7B盲评</b>
        <p>{html.escape(str(judgment.get("reason") or ""))}</p>
        <small>置信度 {html.escape(str(judgment.get("confidence")))} / 5；
        Base可用 {html.escape(str(judgment.get("base_usable_count")))}；
        LoRA可用 {html.escape(str(judgment.get("lora_usable_count")))}</small>
      </div>
    </section>
  </div>

  <div class="methods">
    <section class="method base-panel">
      <h3>Base 3B <span>{len(base_row.get("candidates") or [])} candidates</span></h3>
      <ol>{candidate_list(base_row.get("candidates") or [], base_best)}</ol>
    </section>
    <section class="method lora-panel">
      <h3>Pilot LoRA <span>{len(lora_row.get("candidates") or [])} candidates</span></h3>
      <ol>{candidate_list(lora_row.get("candidates") or [], lora_best)}</ol>
    </section>
  </div>
</article>
"""
        )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Base 3B vs Pilot LoRA Caption Gallery</title>
<style>
:root {{
  color-scheme: light;
  --ink:#20242d; --muted:#6b7280; --line:#dfe3ea; --paper:#fff;
  --base:#2563eb; --lora:#7c3aed; --win:#15803d; --tie:#64748b;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       color:var(--ink); background:#f3f5f8; }}
.top {{ position:sticky; top:0; z-index:10; padding:18px 24px; background:rgba(255,255,255,.96);
        border-bottom:1px solid var(--line); backdrop-filter:blur(10px); }}
.top-inner {{ max-width:1440px; margin:auto; }}
h1 {{ margin:0 0 4px; font-size:24px; }}
.summary {{ color:var(--muted); margin-bottom:12px; }}
.filters {{ display:flex; flex-wrap:wrap; gap:8px; }}
button {{ border:1px solid var(--line); border-radius:999px; padding:7px 13px; background:white;
         cursor:pointer; font-weight:650; }}
button.active {{ color:white; background:#111827; border-color:#111827; }}
main {{ max-width:1440px; margin:20px auto 60px; padding:0 20px; }}
.sample {{ background:var(--paper); border:1px solid var(--line); border-radius:16px;
           margin-bottom:22px; overflow:hidden; box-shadow:0 5px 18px rgba(15,23,42,.05); }}
.sample-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px;
                padding:14px 18px; border-bottom:1px solid var(--line); }}
.ordinal {{ color:var(--muted); margin-right:8px; }} .image-id {{ font-weight:750; }}
.badges {{ display:flex; gap:8px; flex-wrap:wrap; }}
.winner,.cue {{ padding:4px 9px; border-radius:999px; font-size:13px; font-weight:750; }}
.winner.base {{ color:#1d4ed8; background:#dbeafe; }}
.winner.lora {{ color:#6d28d9; background:#ede9fe; }}
.winner.tie {{ color:#475569; background:#e2e8f0; }}
.cue.present {{ color:#9a3412; background:#ffedd5; }} .cue.absent {{ color:#64748b; background:#f1f5f9; }}
.context-grid {{ display:grid; grid-template-columns:minmax(300px,.9fr) minmax(360px,1.1fr);
                 gap:20px; padding:18px; }}
figure {{ margin:0; }} figure img {{ width:100%; max-height:520px; object-fit:contain;
                                    border-radius:10px; background:#eef1f5; }}
figcaption {{ margin-top:6px; color:var(--muted); font-size:11px; overflow-wrap:anywhere; }}
h3 {{ margin:0 0 10px; font-size:17px; }} h3 span {{ color:var(--muted); font-size:12px; font-weight:500; }}
.field {{ margin-bottom:14px; }} .field p {{ margin:4px 0; }} em {{ color:var(--muted); }}
details {{ margin:12px 0; }} summary {{ cursor:pointer; color:#374151; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#f8fafc; border:1px solid var(--line);
       border-radius:8px; padding:10px; font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }}
.judge {{ margin-top:16px; padding:12px; border-radius:10px; background:#f8fafc; border:1px solid var(--line); }}
.judge p {{ margin:5px 0; }} .judge small {{ color:var(--muted); }}
.methods {{ display:grid; grid-template-columns:1fr 1fr; border-top:1px solid var(--line); }}
.method {{ padding:18px; min-width:0; }} .method + .method {{ border-left:1px solid var(--line); }}
.base-panel h3 {{ color:var(--base); }} .lora-panel h3 {{ color:var(--lora); }}
ol {{ padding:0; margin:0; list-style:none; display:grid; gap:9px; }}
.candidate {{ display:flex; gap:10px; align-items:flex-start; padding:10px; border:1px solid var(--line);
              border-radius:9px; background:#fff; }}
.candidate.selected {{ border-color:#f59e0b; background:#fffbeb; }}
.number {{ flex:0 0 24px; height:24px; display:grid; place-items:center; border-radius:50%;
           color:#475569; background:#e9eef5; font-size:12px; font-weight:800; }}
.caption {{ flex:1; overflow-wrap:anywhere; }} .best {{ color:#92400e; background:#fde68a;
  border-radius:999px; padding:2px 7px; font-size:11px; font-weight:750; white-space:nowrap; }}
.hidden {{ display:none; }}
@media (max-width:850px) {{
  .context-grid,.methods {{ grid-template-columns:1fr; }}
  .method + .method {{ border-left:0; border-top:1px solid var(--line); }}
  .sample-head {{ align-items:flex-start; }}
}}
</style>
</head>
<body>
<div class="top">
  <div class="top-inner">
    <h1>Base 3B vs Pilot LoRA：Caption 对照</h1>
    <div class="summary">
      共 {len(shared)} 张 · Base胜 {counts["base"]} · LoRA胜 {counts["lora"]} ·
      平局 {counts["tie"]} · 每组4个候选 · 黄色标记为盲评选中的组内最佳
    </div>
    <div class="filters">
      <button class="active" data-filter="all">全部</button>
      <button data-filter="base">Base胜</button>
      <button data-filter="lora">LoRA胜</button>
      <button data-filter="tie">平局</button>
      <button data-filter="cue-yes">有幽默点</button>
      <button data-filter="cue-no">无幽默点</button>
    </div>
  </div>
</div>
<main>
{''.join(cards)}
</main>
<script>
const buttons = [...document.querySelectorAll('button[data-filter]')];
const samples = [...document.querySelectorAll('.sample')];
for (const button of buttons) {{
  button.addEventListener('click', () => {{
    buttons.forEach(b => b.classList.remove('active'));
    button.classList.add('active');
    const f = button.dataset.filter;
    for (const sample of samples) {{
      const show = f === 'all' || sample.dataset.winner === f ||
        (f === 'cue-yes' && sample.dataset.cue === 'yes') ||
        (f === 'cue-no' && sample.dataset.cue === 'no');
      sample.classList.toggle('hidden', !show);
    }}
  }});
}}
</script>
</body>
</html>
"""
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Samples: {len(shared)}; counts: {dict(counts)}")


if __name__ == "__main__":
    main()
