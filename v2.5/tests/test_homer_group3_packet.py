from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_group3_uses_nine_candidates_without_reuse(tmp_path: Path) -> None:
    source = tmp_path / "pools.jsonl"
    rows = [
        {"cluster_id": "nycc_1", "seed": 7, "image": "1.jpg", "system": name, "candidates": [f"{name}-{i}" for i in range(10)]}
        for name in ("base", "sft")
    ]
    source.write_text("".join(json.dumps(row)+"\n" for row in rows))
    output = tmp_path / "packet"
    subprocess.run([sys.executable, "scripts/build_homer_group3_packet.py", "--input", str(source), "--output-dir", str(output)], check=True)
    packet = [json.loads(line) for line in (output / "blind_packet.jsonl").read_text().splitlines()]
    assert len(packet) == 3
    flattened = [caption for row in packet for label in ("group_A", "group_B") for caption in row[label]]
    assert len(flattened) == len(set(flattened)) == 18
    assert all(not value.endswith("-9") for value in flattened)
