#!/usr/bin/env python3
"""Build independently re-blinded, self-contained human evaluation packets."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_rater(rows: list[dict[str, Any]], rater_id: str, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    swap_ids = {row["pair_id"] for row in ordered[: len(ordered) // 2]}
    public, key = [], []
    image_cache: dict[str, str] = {}
    for row in ordered:
        swapped = row["pair_id"] in swap_ids
        blind_id = hashlib.sha256(f"{rater_id}:{seed}:{row['pair_id']}".encode()).hexdigest()[:16]
        image_path = str(row["image"])
        image_cache.setdefault(image_path, image_uri(Path(image_path)))
        public.append({
            "blind_id": blind_id,
            "image_id": row["image_id"],
            "image_uri": image_cache[image_path],
            "group_A": row["group_B"] if swapped else row["group_A"],
            "group_B": row["group_A"] if swapped else row["group_B"],
        })
        key.append({"blind_id": blind_id, "original_pair_id": row["pair_id"], "swapped": swapped})
    return public, key


def render_html(rater_id: str, rows: list[dict[str, Any]]) -> str:
    images = {row["image_id"]: row["image_uri"] for row in rows}
    compact_rows = [{key: value for key, value in row.items() if key != "image_uri"} for row in rows]
    payload = json.dumps(compact_rows, ensure_ascii=False).replace("</", "<\\/")
    image_payload = json.dumps(images).replace("</", "<\\/")
    return f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">
<title>独立幽默 Caption 盲评 — {html.escape(rater_id)}</title>
<style>body{{font:16px system-ui;max-width:1180px;margin:auto;padding:20px;background:#f5f7fa}}.card{{background:white;padding:20px;margin:18px 0;border-radius:12px;box-shadow:0 2px 9px #ccd}}img{{max-width:620px;max-height:460px;display:block;margin:auto}}.groups{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.group{{border:1px solid #ccd;padding:12px;border-radius:9px}}label{{margin-right:14px}}button{{padding:10px 16px}}.sticky{{position:sticky;top:0;background:#eef;padding:10px;z-index:2}}small{{color:#566}}@media(max-width:760px){{.groups{{grid-template-columns:1fr}}}}</style>
<body><h1>独立幽默 Caption 盲评</h1><div class=\"sticky\"><b id=\"progress\"></b> <button onclick=\"exportData()\">导出 JSON</button></div>
<p>不知道模型身份的前提下判断。Overall：三条整体哪组更好笑；Best：两组最优单条相比。真正无法区分时选 Tie。绝对质量：good=至少一条图像相关且有明确笑点；weak=相关但一般/牵强；bad=全部不可用。不要查阅答案映射。</p><main id=\"root\"></main>
<script>const RID={json.dumps(rater_id)}, rows={payload}, images={image_payload}; const key='humor-blind-'+RID; let state=JSON.parse(localStorage.getItem(key)||'{{}}');
function opts(name,vals){{return vals.map(v=>`<label><input type=radio name="${{name}}" value="${{v}}">${{v}}</label>`).join('')}}
function render(){{root.innerHTML=rows.map((r,i)=>`<section class=card id="c-${{r.blind_id}}"><h2>${{i+1}} / ${{rows.length}}</h2><img src="${{images[r.image_id]}}"><div class=groups><div class=group><h3>A</h3>${{r.group_A.map((x,j)=>`<p>${{j+1}}. ${{esc(x)}}</p>`).join('')}}<b>绝对质量</b> ${{opts(r.blind_id+'-absolute_A',['good','weak','bad'])}}<p>最优序号 ${{opts(r.blind_id+'-best_A_index',['1','2','3'])}}</p></div><div class=group><h3>B</h3>${{r.group_B.map((x,j)=>`<p>${{j+1}}. ${{esc(x)}}</p>`).join('')}}<b>绝对质量</b> ${{opts(r.blind_id+'-absolute_B',['good','weak','bad'])}}<p>最优序号 ${{opts(r.blind_id+'-best_B_index',['1','2','3'])}}</p></div></div><p><b>Overall</b> ${{opts(r.blind_id+'-overall',['A','B','Tie'])}}</p><p><b>Best pick</b> ${{opts(r.blind_id+'-best_pick',['A','B','Tie'])}}</p></section>`).join(''); document.querySelectorAll('input').forEach(x=>{{if(state[x.name]===x.value)x.checked=true;x.onchange=()=>{{state[x.name]=x.value;localStorage.setItem(key,JSON.stringify(state));progress()}}}});progress()}}
function esc(s){{const d=document.createElement('div');d.textContent=s;return d.innerHTML}} function progress(){{let done=rows.filter(r=>['overall','best_pick','best_A_index','best_B_index','absolute_A','absolute_B'].every(f=>state[r.blind_id+'-'+f])).length;document.getElementById('progress').textContent=`已完成 ${{done}} / ${{rows.length}}`}}
function exportData(){{let decisions={{}},missing=[];for(const r of rows){{let d={{}};for(const f of ['overall','best_pick','best_A_index','best_B_index','absolute_A','absolute_B']){{let v=state[r.blind_id+'-'+f];if(!v)missing.push(r.blind_id+':'+f);d[f]=f.includes('index')&&v?Number(v):(v||'')}}decisions[r.blind_id]=d}}if(missing.length&&!confirm(`尚缺 ${{missing.length}} 项，仍然导出？`))return;let blob=new Blob([JSON.stringify({{rater_id:RID,decisions}},null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=RID+'_decisions.json';a.click()}}render();</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raters", nargs="+", default=["human_1", "human_2"])
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    rows = read_jsonl(args.trials)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    master = {"source": str(args.trials), "raters": {}}
    for index, rater in enumerate(args.raters):
        public, key = build_rater(rows, rater, args.seed + index * 1009)
        (args.output_dir / f"{rater}.html").write_text(render_html(rater, public), encoding="utf-8")
        master["raters"][rater] = key
    private = args.output_dir.parent / f"{args.output_dir.name}_private_key.json"
    private.write_text(json.dumps(master, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"trials": len(rows), "raters": args.raters, "public_dir": str(args.output_dir), "private_key": str(private)}, indent=2))


if __name__ == "__main__":
    main()
