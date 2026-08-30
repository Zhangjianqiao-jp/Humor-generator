import json

from scripts.build_counterfactual_plan_inputs import build


def _row(image_id: str, target: str, kind: str) -> dict:
    payload = {
        "scene": f"scene {image_id}",
        "type": kind,
        "target": target,
        "primary_view": "full_image",
        "views": ["full_image"],
        "anchors": [{"label": image_id, "evidence": "visible", "role": "grounds it"}],
        "external_knowledge": False,
    }
    return {
        "image": f"{image_id}.jpg",
        "image_id": image_id,
        "compact_json": json.dumps(payload),
    }


def test_counterfactuals_are_deranged_and_preserve_corrupted_grounding() -> None:
    rows = [_row("a", "target a", "scale_contrast"), _row("b", "target b", "object_misuse"), _row("c", "target c", "role_mismatch")]
    swapped, corrupted = build(rows, seed=7)
    source = {row["image_id"]: json.loads(row["compact_json"]) for row in rows}
    for swap, corrupt in zip(swapped, corrupted, strict=True):
        image_id = swap["image_id"]
        donor_id = swap["donor_image_id"]
        assert image_id != donor_id
        assert json.loads(swap["compact_json"]) == source[donor_id]
        corrupt_payload = json.loads(corrupt["compact_json"])
        assert corrupt_payload["scene"] == source[image_id]["scene"]
        assert corrupt_payload["anchors"] == source[image_id]["anchors"]
        assert corrupt_payload["target"] == source[donor_id]["target"]
