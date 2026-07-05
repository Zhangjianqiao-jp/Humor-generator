from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _escape(value: Any) -> str:
    return html.escape(str(value or ""))


def _list_html(items: Any) -> str:
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        items = []
    if not items:
        return "<em>None</em>"
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in items if str(item).strip()) + "</ul>"


def _visual_facts_html(visual_facts: dict[str, Any] | None) -> str:
    visual_facts = visual_facts or {}
    # Backward compatibility for older rows that still contain humor_points.
    if "literal_description" not in visual_facts and any(
        key in visual_facts for key in ("abnormal_points", "conflict_points", "humor_angle", "avoid")
    ):
        return f"""
      <div class="kv"><strong>Legacy salient</strong>{_list_html(visual_facts.get("salient_points"))}</div>
      <div class="kv"><strong>Legacy avoid</strong>{_list_html(visual_facts.get("avoid"))}</div>
    """
    return f"""
      <div class="kv"><strong>Literal description</strong><p>{_escape(visual_facts.get("literal_description"))}</p></div>
      <div class="kv"><strong>Visible objects</strong>{_list_html(visual_facts.get("visible_objects"))}</div>
      <div class="kv"><strong>Visible actions</strong>{_list_html(visual_facts.get("visible_actions"))}</div>
      <div class="kv"><strong>Salient points</strong>{_list_html(visual_facts.get("salient_points"))}</div>
      <div class="kv"><strong>Visible text</strong>{_list_html(visual_facts.get("visible_text"))}</div>
      <div class="kv"><strong>Uncertain / unreadable</strong>{_list_html(visual_facts.get("uncertain_or_unreadable"))}</div>
    """


def _entity_summary(entity: dict[str, Any]) -> str:
    attrs = entity.get("attributes") if isinstance(entity.get("attributes"), list) else []
    suffix = f" ({'; '.join(str(attr) for attr in attrs if str(attr).strip())})" if attrs else ""
    return f"{entity.get('id')}: {entity.get('label')}{suffix}"


def _relation_summary(relation: dict[str, Any]) -> str:
    return " ".join(
        str(relation.get(key) or "").strip()
        for key in ("subject", "predicate", "object")
        if str(relation.get(key) or "").strip()
    )


def _humor_viewpoint_html(humor_viewpoint: dict[str, Any] | None) -> str:
    if not isinstance(humor_viewpoint, dict) or not humor_viewpoint:
        return "<em>None</em>"
    anchors = []
    for anchor in humor_viewpoint.get("visual_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        parts = [str(anchor.get("label") or "").strip()]
        if anchor.get("evidence"):
            parts.append(f"evidence: {anchor.get('evidence')}")
        if anchor.get("role"):
            parts.append(f"role: {anchor.get('role')}")
        text = "; ".join(part for part in parts if part)
        if text:
            anchors.append(text)
    rows = [
        f"type: {humor_viewpoint.get('humor_type') or 'none'}",
        f"primary viewpoint: {humor_viewpoint.get('primary_viewpoint') or 'none'}",
        f"required viewpoints: {' | '.join(str(v) for v in humor_viewpoint.get('required_viewpoints') or []) or 'none'}",
        f"confidence: {humor_viewpoint.get('confidence') or 'none'}",
        f"external knowledge: {bool(humor_viewpoint.get('needs_external_knowledge'))}",
    ]
    return f"""
      <div class="kv"><strong>Viewpoint summary</strong>{_list_html(rows)}</div>
      <div class="kv"><strong>Literal scene</strong><p>{_escape(humor_viewpoint.get("literal_image_description"))}</p></div>
      <div class="kv"><strong>Humor point</strong><p>{_escape(humor_viewpoint.get("humor_point"))}</p></div>
      <div class="kv"><strong>Visual anchors</strong>{_list_html(anchors)}</div>
      <div class="kv"><strong>Uncertainty</strong><p>{_escape(humor_viewpoint.get("uncertainty"))}</p></div>
    """


def _structured_humor_html(structured_humor: dict[str, Any] | None, parse_error: Any = None) -> str:
    if not isinstance(structured_humor, dict):
        if parse_error:
            return f"<div class=\"kv\"><strong>Parse error</strong><p>{_escape(parse_error)}</p></div>"
        return "<em>None</em>"
    visible = structured_humor.get("visible_facts") if isinstance(structured_humor.get("visible_facts"), dict) else {}
    inferred = structured_humor.get("inferred_context") if isinstance(structured_humor.get("inferred_context"), dict) else {}
    mechanism = structured_humor.get("humor_mechanism") if isinstance(structured_humor.get("humor_mechanism"), dict) else {}
    guidance = structured_humor.get("generator_guidance") if isinstance(structured_humor.get("generator_guidance"), dict) else {}
    entities = [_entity_summary(entity) for entity in visible.get("entities", []) if isinstance(entity, dict)]
    relations = [_relation_summary(relation) for relation in visible.get("relations", []) if isinstance(relation, dict)]
    inferred_items = []
    for item in inferred.get("items", []):
        if not isinstance(item, dict):
            continue
        claim = item.get("claim")
        basis = item.get("basis")
        confidence = item.get("confidence")
        parts = [str(claim or "")]
        if confidence:
            parts.append(f"confidence: {confidence}")
        if basis:
            parts.append(f"basis: {basis}")
        inferred_items.append("; ".join(part for part in parts if part))
    mechanism_rows = [
        f"type: {mechanism.get('type') or 'none'}",
        f"expected: {mechanism.get('expected_frame') or 'None'}",
        f"violation: {mechanism.get('observed_violation') or 'None'}",
        f"resolution: {mechanism.get('resolution') or 'None'}",
        f"strategy: {mechanism.get('caption_strategy') or 'none'}",
        f"cue: {guidance.get('one_line_cue') or 'None'}",
    ]
    parse_html = ""
    if parse_error:
        parse_html = f"<div class=\"kv warn\"><strong>Parse error</strong><p>{_escape(parse_error)}</p></div>"
    return f"""
      {parse_html}
      <div class="kv"><strong>Entities</strong>{_list_html(entities)}</div>
      <div class="kv"><strong>Relations</strong>{_list_html(relations)}</div>
      <div class="kv"><strong>Inferred context</strong>{_list_html(inferred_items)}</div>
      <div class="kv"><strong>Humor mechanism</strong>{_list_html(mechanism_rows)}</div>
      <div class="kv"><strong>Warnings</strong>{_list_html(structured_humor.get("warnings"))}</div>
    """


def write_guided_review_html(rows: list[dict[str, Any]], output_html: Path, max_rows: int | None = None) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    if max_rows is not None:
        rows = rows[:max_rows]
    cards = []
    for idx, row in enumerate(rows, start=1):
        candidates = row.get("candidates") or []
        candidate_html = "<ol>" + "".join(f"<li>{_escape(candidate)}</li>" for candidate in candidates) + "</ol>"
        meta = {
            "image_id": row.get("image_id"),
            "method": row.get("method"),
            "num_candidates": len(candidates),
            "generator_mode": (row.get("meta") or {}).get("generator_mode"),
        }
        cards.append(
            f"""
            <section class="card">
              <div class="media">
                <img src="{_escape(row.get("image"))}" alt="image {idx}">
              </div>
              <div class="content">
                <h2>{idx}. {_escape(row.get("image_id") or Path(str(row.get("image") or "")).stem)}</h2>
                <pre class="meta">{_escape(json.dumps(meta, ensure_ascii=False, indent=2))}</pre>
                <h3>Image Description</h3>
                <p>{_escape(row.get("image_description"))}</p>
                <h3>Visual Facts</h3>
                {_visual_facts_html(row.get("visual_facts") or row.get("humor_points"))}
                <h3>Structured Humor</h3>
                {_structured_humor_html(row.get("structured_humor"), row.get("structured_humor_parse_error"))}
                <h3>Humor Viewpoint</h3>
                {_humor_viewpoint_html(row.get("humor_viewpoint"))}
                <h3>Candidates</h3>
                {candidate_html}
                <details>
                  <summary>Prompt</summary>
                  <pre>{_escape(row.get("prompt"))}</pre>
                </details>
              </div>
            </section>
            """
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VLM-guided caption review</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f5f5f2;
      color: #222;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 16px 24px;
      background: #1f2933;
      color: white;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 20px;
    }}
    .card {{
      display: grid;
      grid-template-columns: minmax(240px, 420px) 1fr;
      gap: 20px;
      margin: 0 0 20px;
      padding: 16px;
      background: white;
      border: 1px solid #ddd;
      border-radius: 8px;
    }}
    img {{
      width: 100%;
      height: auto;
      max-height: 520px;
      object-fit: contain;
      background: #eee;
      border-radius: 6px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    h3 {{
      margin: 14px 0 6px;
      font-size: 15px;
      color: #374151;
    }}
    p, li {{
      line-height: 1.45;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f3f4f6;
      padding: 10px;
      border-radius: 6px;
      font-size: 12px;
    }}
    .meta {{
      color: #555;
    }}
    .kv ul {{
      margin: 4px 0 8px 20px;
      padding: 0;
    }}
    .warn {{
      color: #991b1b;
    }}
    @media (max-width: 800px) {{
      .card {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <strong>VLM-guided caption review</strong>
    <span>{len(rows)} rows</span>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    output_html.write_text(page, encoding="utf-8")
