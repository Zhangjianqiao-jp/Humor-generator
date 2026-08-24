#!/usr/bin/env python3
"""Build blind, browser-based human validation packets for preference diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preference.diagnostics import read_jsonl, sha256, write_json


def stable_id(*values: str) -> str:
    payload = "\x1f".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def plan_from_prompt(prompt: str) -> str:
    marker = "Humor plan:"
    return prompt.split(marker, 1)[1].strip() if marker in prompt else prompt.strip()


def blind_pair_records(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    records, keys = [], []
    for row in rows:
        pair_id = stable_id(str(row["image_id"]), str(row["chosen"]), str(row["rejected"]))
        swapped = bool(rng.getrandbits(1))
        caption_a = str(row["rejected"] if swapped else row["chosen"])
        caption_b = str(row["chosen"] if swapped else row["rejected"])
        records.append(
            {
                "item_id": pair_id,
                "image_id": str(row["image_id"]),
                "thumb": f"thumbs/{row['image_id']}.jpg",
                "plan": plan_from_prompt(str(row.get("prompt") or "")),
                "caption_a": caption_a,
                "caption_b": caption_b,
            }
        )
        keys.append(
            {
                "item_id": pair_id,
                "image_id": str(row["image_id"]),
                "a_source": "rejected" if swapped else "chosen",
                "b_source": "chosen" if swapped else "rejected",
                "chosen_score": row.get("chosen_score"),
                "rejected_score": row.get("rejected_score"),
                "score_margin": row.get("score_margin"),
                "source_pair_type": row.get("pair_type"),
            }
        )
    order = sorted(range(len(records)), key=lambda i: (records[i]["image_id"], records[i]["item_id"]))
    return [records[i] for i in order], [keys[i] for i in order]


def best_candidate(row: dict[str, Any]) -> dict[str, Any]:
    items = row.get("judged_candidates") or []
    if not items:
        raise ValueError(f"Best-of-N row {row.get('image_id')} has no judged candidates")
    return max(items, key=lambda item: (int(item["humor"]), int(item["grounding"]), -int(item["index"])))


def blind_best_of_n_records(
    joint: list[dict[str, Any]], direct: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed + 1)
    left = {str(row["image_id"]): row for row in joint}
    right = {str(row["image_id"]): row for row in direct}
    if set(left) != set(right):
        raise ValueError("joint and direct Best-of-N rows do not have identical image IDs")
    records, keys = [], []
    for image_id in sorted(left):
        joint_best, direct_best = best_candidate(left[image_id]), best_candidate(right[image_id])
        item_id = stable_id("best-of-n", image_id, str(joint_best["candidate"]), str(direct_best["candidate"]))
        swapped = bool(rng.getrandbits(1))
        values = {"joint": joint_best, "direct": direct_best}
        a_source, b_source = (("direct", "joint") if swapped else ("joint", "direct"))
        records.append(
            {
                "item_id": item_id,
                "image_id": image_id,
                "thumb": f"thumbs/{image_id}.jpg",
                "caption_a": str(values[a_source]["candidate"]),
                "caption_b": str(values[b_source]["candidate"]),
                "plan": "Blind comparison of each system's auxiliary-judge best-of-32 candidate.",
            }
        )
        keys.append(
            {
                "item_id": item_id,
                "image_id": image_id,
                "a_source": a_source,
                "b_source": b_source,
                "joint_candidate_index": joint_best["index"],
                "joint_aux_humor": joint_best["humor"],
                "joint_aux_grounding": joint_best["grounding"],
                "direct_candidate_index": direct_best["index"],
                "direct_aux_humor": direct_best["humor"],
                "direct_aux_grounding": direct_best["grounding"],
            }
        )
    return records, keys


def dashboard_html(title: str, description: str, records: list[dict[str, Any]], storage_key: str, mode: str) -> str:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    pair_type = """<label>Pair type<select id="pair_type"><option value="">unreviewed</option><option>H1</option><option>H2</option><option>H3</option><option>H4</option><option>invalid</option></select></label>""" if mode == "pairs" else ""
    use_train = """<label>Use for training<select id="use_for_training"><option value="">unreviewed</option><option value="yes">yes</option><option value="no">no</option></select></label>""" if mode == "pairs" else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--navy:#0b2545;--blue:#174f82;--green:#087f5b;--red:#b42318;--paper:#f5f7fb;--line:#d0d7e2}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);font:15px/1.45 system-ui,sans-serif;color:#172033}}
header{{position:sticky;top:0;z-index:4;background:var(--navy);color:white;padding:14px 22px;display:flex;gap:18px;align-items:center}}
header h1{{font-size:20px;margin:0}} header .progress{{margin-left:auto}} button{{border:0;border-radius:7px;padding:9px 14px;cursor:pointer}}
.primary{{background:#dbeafe;color:#0b2545}} main{{max-width:1280px;margin:20px auto;padding:0 20px}}
.guide{{background:white;border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:16px}}
.workspace{{display:grid;grid-template-columns:minmax(300px,42%) 1fr;gap:18px}}
.image-panel,.review{{background:white;border:1px solid var(--line);border-radius:12px;padding:16px}}
.image-panel img{{display:block;max-width:100%;max-height:560px;margin:auto;object-fit:contain}}
.meta{{color:#657085;font-size:13px;margin-bottom:10px}} details{{margin-top:12px}} pre{{white-space:pre-wrap;background:#f3f5f8;padding:10px;border-radius:7px}}
.captions{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .caption{{border:2px solid var(--line);border-radius:10px;padding:14px;min-height:130px}}
.caption h2{{margin:0 0 10px;color:var(--blue)}} .caption p{{font-size:18px}}
.fields{{display:grid;grid-template-columns:repeat(2,minmax(180px,1fr));gap:10px;margin-top:16px}} label{{display:flex;flex-direction:column;gap:5px;font-weight:600}}
select,textarea{{font:inherit;padding:8px;border:1px solid #aeb8c6;border-radius:6px;background:white}} textarea{{width:100%;min-height:70px}}
.nav{{display:flex;gap:10px;margin-top:16px}} .nav button{{background:var(--blue);color:white}} .nav button:first-child{{background:#667085}}
.checks{{margin:0;padding-left:22px;columns:2}} .status{{font-weight:700;color:#b54708}} @media(max-width:800px){{.workspace{{grid-template-columns:1fr}}.checks{{columns:1}}}}
</style></head><body>
<header><h1>{title}</h1><span id="status" class="status"></span><span class="progress" id="progress"></span><button class="primary" onclick="exportJSONL()">Export JSONL</button></header>
<main><section class="guide"><p>{description}</p><ol class="checks"><li>Judge the image before reading the humor plan.</li><li>Select which caption is funnier for this image.</li><li>Score visual grounding independently.</li><li>Flag hallucination and generic/template humor.</li><li>Use tie/invalid when preference is not clear.</li><li>Do not consult the blind key while annotating.</li></ol></section>
<section class="workspace"><div class="image-panel"><div class="meta" id="meta"></div><img id="image"><details><summary>Show conditioning plan/context</summary><pre id="plan"></pre></details></div>
<div class="review"><div class="captions"><article class="caption"><h2>A</h2><p id="caption_a"></p></article><article class="caption"><h2>B</h2><p id="caption_b"></p></article></div>
<div class="fields"><label>Preference<select id="preference"><option value="">unreviewed</option><option value="A">A is funnier</option><option value="B">B is funnier</option><option value="tie">tie / unclear</option><option value="invalid">invalid pair</option></select></label>
<label>Confidence<select id="confidence"><option value="">unreviewed</option><option>high</option><option>medium</option><option>low</option></select></label>
<label>A grounding<select id="a_grounding"><option value=""></option>{''.join(f'<option>{i}</option>' for i in range(1,6))}</select></label>
<label>B grounding<select id="b_grounding"><option value=""></option>{''.join(f'<option>{i}</option>' for i in range(1,6))}</select></label>
<label>A hallucinated?<select id="a_hallucination"><option value=""></option><option>no</option><option>yes</option></select></label>
<label>B hallucinated?<select id="b_hallucination"><option value=""></option><option>no</option><option>yes</option></select></label>
<label>A generic/template?<select id="a_generic"><option value=""></option><option>no</option><option>yes</option></select></label>
<label>B generic/template?<select id="b_generic"><option value=""></option><option>no</option><option>yes</option></select></label>{pair_type}{use_train}</div>
<label style="margin-top:12px">Notes<textarea id="notes" placeholder="Why does the winner work? Record visual ambiguity or failure mode."></textarea></label>
<div class="nav"><button onclick="move(-1)">Previous</button><button onclick="save();move(1)">Save & Next</button><button onclick="jumpUnreviewed()">Next unreviewed</button></div></div></section></main>
<script>const DATA={payload}; const KEY={json.dumps(storage_key)}; let idx=0; const fieldIds=['preference','confidence','a_grounding','b_grounding','a_hallucination','b_hallucination','a_generic','b_generic','notes'{",'pair_type','use_for_training'" if mode == "pairs" else ""}];
let answers=JSON.parse(localStorage.getItem(KEY)||'{{}}'); function el(x){{return document.getElementById(x)}}
function save(){{let r={{item_id:DATA[idx].item_id,image_id:DATA[idx].image_id}};fieldIds.forEach(k=>r[k]=el(k).value);answers[r.item_id]=r;localStorage.setItem(KEY,JSON.stringify(answers));renderProgress();el('status').textContent='saved';setTimeout(()=>el('status').textContent='',800)}}
function load(){{let d=DATA[idx],a=answers[d.item_id]||{{}};el('meta').textContent=`${{idx+1}}/${{DATA.length}} · ${{d.image_id}} · ${{d.item_id}}`;el('image').src=d.thumb;el('caption_a').textContent=d.caption_a;el('caption_b').textContent=d.caption_b;el('plan').textContent=d.plan;fieldIds.forEach(k=>el(k).value=a[k]||'');renderProgress()}}
function move(n){{save();idx=Math.max(0,Math.min(DATA.length-1,idx+n));load();window.scrollTo(0,0)}}
function jumpUnreviewed(){{save();for(let n=1;n<=DATA.length;n++){{let j=(idx+n)%DATA.length,a=answers[DATA[j].item_id];if(!a||!a.preference){{idx=j;load();return}}}}alert('All items have a preference decision.')}}
function renderProgress(){{let done=DATA.filter(d=>answers[d.item_id]&&answers[d.item_id].preference).length;el('progress').textContent=`Reviewed ${{done}} / ${{DATA.length}}`}}
function exportJSONL(){{save();let text=DATA.map(d=>JSON.stringify(answers[d.item_id]||{{item_id:d.item_id,image_id:d.image_id}})).join('\\n')+'\\n';let b=new Blob([text],{{type:'application/jsonl'}}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download=KEY+'.jsonl';a.click();URL.revokeObjectURL(u)}} load();</script></body></html>"""


def save_thumbnails(image_paths: dict[str, Path], output_dir: Path) -> None:
    thumb_dir = output_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    for image_id, source in image_paths.items():
        target = thumb_dir / f"{image_id}.jpg"
        if target.exists():
            continue
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((900, 700))
            image.save(target, quality=90, optimize=True)


def blank_csv(path: Path, records: list[dict[str, Any]], mode: str) -> None:
    fields = ["item_id", "image_id", "preference", "confidence", "a_grounding", "b_grounding", "a_hallucination", "b_hallucination", "a_generic", "b_generic"]
    if mode == "pairs":
        fields.extend(["pair_type", "use_for_training"])
    fields.append("notes")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({"item_id": record["item_id"], "image_id": record["image_id"]})


def overview_png(path: Path, pair_count: int, image_count: int, eval_count: int) -> None:
    image = Image.new("RGB", (1500, 820), "#f7f9fc")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default()
    draw.text((55, 35), "HUMAN VALIDATION BEFORE PREFERENCE TRAINING", fill="#0b2545", font=font)
    boxes = [
        ("SOURCE", "13,190 captions\n79 training cartoons"),
        ("AUTO MATCH", f"{pair_count} H2 candidates\n{image_count} pair-producing images"),
        ("BLIND HUMAN CHECK", "Humor preference\nGrounding / hallucination\nGeneric shortcut / pair type"),
        ("FREEZE DATASET", "Keep clear, grounded pairs\nSplit by image\nVersion + hash"),
    ]
    x_positions = [45, 400, 755, 1110]
    for x, (heading, body) in zip(x_positions, boxes, strict=True):
        draw.rounded_rectangle((x, 130, x + 300, 360), radius=20, fill="white", outline="#174f82", width=4)
        draw.text((x + 25, 165), heading, fill="#174f82", font=font)
        draw.multiline_text((x + 25, 220), body, fill="#172033", font=font, spacing=12)
    for x in (350, 705, 1060):
        draw.line((x, 245, x + 40, 245), fill="#087f5b", width=8)
        draw.polygon([(x + 40, 245), (x + 24, 234), (x + 24, 256)], fill="#087f5b")
    draw.rounded_rectangle((120, 475, 1380, 745), radius=22, fill="#fff7e6", outline="#d97706", width=3)
    draw.text((160, 515), "SECOND VALIDATION TRACK: DO NOT TURN AUXILIARY JUDGE SCORES INTO LABELS", fill="#9a3412", font=font)
    draw.multiline_text((160, 575), f"Blindly compare joint vs direct best-of-32 on {eval_count} held-out images.\nConfirm humor and visual grounding. Record judge disagreement.\nOnly after validation: DPO vs SimPO baseline, then image-conditional preference.", fill="#172033", font=font, spacing=16)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("results/preference_diagnostics/current_rank_pairs/pairs.jsonl"))
    parser.add_argument("--joint", type=Path, default=Path("results/preference_diagnostics/best_of_n_joint_vs_direct_v1/joint/humor_judgments.jsonl"))
    parser.add_argument("--direct", type=Path, default=Path("results/preference_diagnostics/best_of_n_joint_vs_direct_v1/direct/humor_judgments.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/preference_diagnostics/human_validation_v1"))
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    pairs, joint, direct = read_jsonl(args.pairs), read_jsonl(args.joint), read_jsonl(args.direct)
    pair_records, pair_keys = blind_pair_records(pairs, args.seed)
    eval_records, eval_keys = blind_best_of_n_records(joint, direct, args.seed)
    paired_records = list(zip(pair_records, pair_keys, strict=True))
    quick_records = []
    for image_id in sorted({record["image_id"] for record in pair_records}):
        candidates = [(record, key) for record, key in paired_records if record["image_id"] == image_id]
        quick_records.append(min(candidates, key=lambda item: (float(item[1]["score_margin"]), item[0]["item_id"]))[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, Path] = {}
    for row in pairs:
        image_paths[str(row["image_id"])] = Path(str(row.get("image_path") or row["image"]))
    for row in joint:
        image_paths[str(row["image_id"])] = Path(str(row["image"]))
    missing = [str(path) for path in image_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing validation images: {missing[:5]}")
    save_thumbnails(image_paths, args.output_dir)

    (args.output_dir / "preference_pairs_blind.html").write_text(
        dashboard_html(
            "Blind Preference Pair Validation",
            "Validate all candidate training pairs. A/B positions are deterministically blinded. H1=humor vs literal; H2=strong vs weak/cliche; H3=grounded vs hallucinated; H4=image-specific vs generic meme.",
            pair_records, "preference_pairs_human_v1", "pairs",
        ), encoding="utf-8",
    )
    (args.output_dir / "preference_pairs_quick_gate_blind.html").write_text(
        dashboard_html(
            "Blind Preference Pair Quick Gate",
            "First-pass quality gate: one lowest-margin candidate pair from each of the 61 pair-producing images. Complete this page before deciding whether all 485 pairs deserve full review.",
            quick_records, "preference_pairs_quick_gate_human_v1", "pairs",
        ), encoding="utf-8",
    )
    (args.output_dir / "best_of_n_blind.html").write_text(
        dashboard_html(
            "Blind Best-of-N Result Audit",
            "Compare each system's auxiliary-judge best-of-32 candidate without seeing system identity or automatic scores. This validates the diagnostic conclusion, not training labels.",
            eval_records, "best_of_n_human_v1", "eval",
        ), encoding="utf-8",
    )
    blank_csv(args.output_dir / "preference_pair_annotations.csv", pair_records, "pairs")
    blank_csv(args.output_dir / "preference_pair_quick_gate_annotations.csv", quick_records, "pairs")
    blank_csv(args.output_dir / "best_of_n_annotations.csv", eval_records, "eval")
    write_json(args.output_dir / "blind_key.json", {"preference_pairs": pair_keys, "best_of_n": eval_keys})
    overview_png(args.output_dir / "validation_overview.png", len(pair_records), len({r["image_id"] for r in pair_records}), len(eval_records))
    manifest = {
        "version": "human-validation-v1",
        "seed": args.seed,
        "preference_pair_items": len(pair_records),
        "preference_pair_images": len({row["image_id"] for row in pair_records}),
        "quick_gate_items": len(quick_records),
        "best_of_n_items": len(eval_records),
        "pairs_sha256": sha256(args.pairs),
        "joint_scores_sha256": sha256(args.joint),
        "direct_scores_sha256": sha256(args.direct),
        "annotation_rule": "Annotate blinded A/B pages first; do not open blind_key.json until exports are frozen.",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "README.md").write_text(
        """# Human validation packet / 人工校验包\n\n1. 先打开 `preference_pairs_quick_gate_blind.html`：每张训练图抽一个最低 margin pair，共 61 项。\n2. 若 quick gate 可接受，再打开 `preference_pairs_blind.html` 完成全部 485 项。\n3. 打开 `best_of_n_blind.html`，复核 24 张 held-out 图片上的 joint/direct 结论。\n4. 网页自动把进度保存在浏览器 localStorage；请定期点击 Export JSONL。\n5. 冻结导出的标注文件以后，才能打开 `blind_key.json` 解盲。\n6. 不要把辅助 judge 分数或原始排名直接当作 preference 标签。\n\n不使用网页时，可以填写对应的空白 CSV。详细标准见 `VALIDATION_GUIDE.md`。\n""",
        encoding="utf-8",
    )
    (args.output_dir / "VALIDATION_GUIDE.md").write_text(
        """# Preference validation guide\n\n## 推荐顺序\n\n1. Quick gate：61 项，每张 pair-producing image 一项，选取该图最低 score-margin 的困难 pair。\n2. Best-of-N audit：24 项，盲评两个系统各自的 best-of-32。\n3. 只有 quick gate 通过后才做完整 485 项。\n\n## 每个 pair 必须判断\n\n- Preference：A/B 哪个对当前图片更好笑；没有明确差异就选 tie。\n- Grounding：1=无关或与图片矛盾，3=大致相关，5=准确利用具体视觉细节。\n- Hallucination：caption 是否依赖图片中不存在的对象、动作或关系。\n- Generic/template：caption 是否可以不改动地套用到很多图片。\n- Pair type：H1 humor-vs-literal；H2 strong-vs-weak/cliche；H3 grounded-vs-hallucinated；H4 image-specific-vs-generic。\n- Use for training：只有偏好明确、winner grounded、negative 不属于低级语法错误时才选 yes。\n\n## Quick gate 停止条件\n\n解盲前不要看来源分数。解盲后若出现以下任一情况，不应直接训练：\n\n- 原 chosen 的人工胜率低于 70%；\n- tie/invalid 合计超过 20%；\n- 超过 15% 的原 chosen 被判 hallucinated 或 grounding <= 2；\n- 大部分 pair 仍只能归为 H2，无法测量 H1/H3/H4。\n\n这些阈值是数据工程 gate，不是论文中的通用常数；最终应报告分子、分母和 bootstrap 95% CI。\n""",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
