#!/usr/bin/env python3
"""Build copy-ready prompts and image-bearing PDF packets for two LLM judges."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image


PAPERS = """Authoritative evaluation basis:
- Zhang et al., Humor in AI, NeurIPS 2024.
- Hessel et al., Do Androids Laugh at Electric Sheep?, ACL 2023.
- Zheng et al., Judging LLM-as-a-Judge, NeurIPS 2023.
- Artstein and Poesio, Inter-Coder Agreement for Computational Linguistics, 2008.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt(rater_id: str) -> str:
    return f"""You are independent blinded multimodal humor judge {rater_id}.

You will receive PDF pages. Every page contains one New Yorker-style cartoon and two anonymous groups, A and B, with exactly three caption candidates per group. Judge every page independently. You do not know which model produced either side. Do not infer or guess model identity.

Mandatory procedure for each page:
1. Inspect the image before reading the captions. Identify the visible situation and unusual visual element.
2. Read all six captions. Do not reward fluent generic jokes that are weakly related to the image.
3. Select OVERALL: A, B, or Tie. This asks which group of three is funnier overall for this specific image.
4. Select the best caption index within A and within B (1, 2, or 3), then select BEST_PICK: A, B, or Tie by comparing those two best captions.
5. Assign ABSOLUTE_A and ABSOLUTE_B:
   - good: at least one caption is genuinely usable, image-grounded, and has a clear humorous turn;
   - weak: the best caption is relevant or mildly amusing, but generic, strained, or not clearly funny;
   - bad: all three captions are off-image, incoherent, merely literal, or unusable.
6. Use Tie when groups or best captions are identical or genuinely indistinguishable. A relative winner is not automatically good.
7. Penalize hallucinated objects/actions, generic meme language, unexplained non sequiturs, and accidental word salad.

Output requirements:
- Complete every blind_id in the supplied response template. The number of IDs is defined by the accompanying packet; do not omit any item.
- Return one JSON object only, with no Markdown fences and no explanation.
- Follow the supplied response template exactly.
- overall and best_pick must be A, B, or Tie.
- best_A_index and best_B_index must be integers 1, 2, or 3.
- absolute_A and absolute_B must be good, weak, or bad.
- Do not reveal chain-of-thought. Make the categorical decisions only.

Important independence rule: do not consult another judge, previous evaluation, answer key, SFT/DPO names, repository results, or web search. Evaluate only the provided image and anonymous captions.

{PAPERS}"""


def pdf_text(value: str) -> bytes:
    data = value.encode("cp1252", errors="replace")
    return data.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def page_stream(row: dict[str, Any], image_name: str, width: int, height: int) -> bytes:
    max_w, max_h = 540.0, 300.0
    scale = min(max_w / width, max_h / height)
    draw_w, draw_h = width * scale, height * scale
    x, y = (612 - draw_w) / 2, 760 - draw_h
    commands = [f"q {draw_w:.2f} 0 0 {draw_h:.2f} {x:.2f} {y:.2f} cm /{image_name} Do Q"]
    lines = [f"blind_id: {row['blind_id']}", "", "GROUP A"]
    for index, caption in enumerate(row["group_A"], 1):
        lines.extend(textwrap.wrap(f"A{index}. {caption}", width=84, subsequent_indent="    ") or [""])
    lines.extend(["", "GROUP B"])
    for index, caption in enumerate(row["group_B"], 1):
        lines.extend(textwrap.wrap(f"B{index}. {caption}", width=84, subsequent_indent="    ") or [""])
    lines.extend(["", "Return: overall, best_pick, best_A_index, best_B_index, absolute_A, absolute_B"])
    text_y = min(y - 18, 430)
    commands.append("BT /F1 9 Tf 11 TL 36 %.2f Td" % text_y)
    for line_index, line in enumerate(lines):
        if line_index:
            commands.append("T*")
        commands.append(b"(" + pdf_text(line) + b") Tj")
    commands.append("ET")
    return b"\n".join(x if isinstance(x, bytes) else x.encode("ascii") for x in commands)


def write_pdf(path: Path, rows: list[dict[str, Any]]) -> None:
    image_paths = list(dict.fromkeys(str(row["image"]) for row in rows))
    # 1 catalog, 2 pages root, 3 font, then unique images, page streams, pages.
    image_obj = {name: 4 + i for i, name in enumerate(image_paths)}
    stream_start = 4 + len(image_paths)
    page_start = stream_start + len(rows)
    page_numbers = [page_start + i for i in range(len(rows))]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: ("<< /Type /Pages /Count %d /Kids [%s] >>" % (len(rows), " ".join(f"{n} 0 R" for n in page_numbers))).encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    image_meta = {}
    for image_index, image_path in enumerate(image_paths):
        raw = Path(image_path).read_bytes()
        with Image.open(image_path) as image:
            width, height = image.size
            colorspace = "/DeviceGray" if image.mode == "L" else "/DeviceRGB"
        image_meta[image_path] = (f"Im{image_index}", width, height)
        header = f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace {colorspace} /BitsPerComponent 8 /Filter /DCTDecode /Length {len(raw)} >>\nstream\n".encode()
        objects[image_obj[image_path]] = header + raw + b"\nendstream"
    for index, row in enumerate(rows):
        name, width, height = image_meta[str(row["image"])]
        stream = page_stream(row, name, width, height)
        objects[stream_start + index] = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        objects[page_start + index] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> /XObject << /{name} {image_obj[str(row['image'])]} 0 R >> >> "
            f"/Contents {stream_start + index} 0 R >>"
        ).encode()
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {0: 0}
    for number in range(1, max(objects) + 1):
        offsets[number] = len(payload)
        payload.extend(f"{number} 0 obj\n".encode()); payload.extend(objects[number]); payload.extend(b"\nendobj\n")
    xref = len(payload); count = max(objects) + 1
    payload.extend(f"xref\n0 {count}\n0000000000 65535 f \n".encode())
    for number in range(1, count):
        payload.extend(f"{offsets[number]:010d} 00000 n \n".encode())
    payload.extend(f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part-size", type=int, default=47)
    args = parser.parse_args()
    trials = {row["pair_id"]: row for row in read_jsonl(args.trials)}
    keys = json.load(args.private_key.open(encoding="utf-8"))["raters"]
    manifest = {"source_trials": str(args.trials), "packages": {}}
    for rater_id, key_rows in keys.items():
        rows = []
        for key in key_rows:
            source = trials[key["original_pair_id"]]
            swapped = bool(key["swapped"])
            rows.append({
                "blind_id": key["blind_id"], "image_id": source["image_id"], "image": source["image"],
                "group_A": source["group_B"] if swapped else source["group_A"],
                "group_B": source["group_A"] if swapped else source["group_B"],
            })
        target = args.output_dir / rater_id; target.mkdir(parents=True, exist_ok=True)
        (target / "PROMPT.txt").write_text(prompt(rater_id), encoding="utf-8")
        with (target / "packet.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        def response_template(items: list[dict[str, Any]]) -> dict[str, Any]:
            return {"rater_id": rater_id, "decisions": {row["blind_id"]: {
                "overall": "", "best_pick": "", "best_A_index": None, "best_B_index": None,
                "absolute_A": "", "absolute_B": ""} for row in items}}
        template = response_template(rows)
        (target / "RESPONSE_TEMPLATE.json").write_text(json.dumps(template, indent=2) + "\n")
        write_pdf(target / "BLIND_PACKET.pdf", rows)
        for part_index, start in enumerate(range(0, len(rows), args.part_size), 1):
            part = rows[start:start + args.part_size]
            write_pdf(target / f"BLIND_PACKET_PART_{part_index:02d}.pdf", part)
            (target / f"RESPONSE_TEMPLATE_PART_{part_index:02d}.json").write_text(
                json.dumps(response_template(part), indent=2) + "\n"
            )
        files = sorted(path for path in target.iterdir() if path.is_file())
        manifest["packages"][rater_id] = {"trials": len(rows), "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
