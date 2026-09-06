#!/usr/bin/env python3
"""CPU-only contract smoke for the HOMER public-code stage graph.

No model weights, API key, CUDA or benchmark files are touched.  This gate
exists so a real Qwen job is never submitted with a missing summary/selection
stage or an accidental SFT adapter.
"""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.homer.public_code_pipeline import HomerPublicCodePipeline


class SmokeBackend:
    def __init__(self) -> None:
        self.stages: list[str] = []

    def generate(self, messages, *, temperature, max_new_tokens, seed):
        del temperature, max_new_tokens, seed
        text = " ".join(
            part
            for message in messages
            for part in (
                [message.get("content")]
                if isinstance(message.get("content"), str)
                else [block.get("text", "") for block in (message.get("content") or []) if isinstance(block, dict)]
            )
            if isinstance(part, str)
        )
        if "canny and accurate description" in text:
            self.stages.append("extractor.description")
            return "##Vivid Description:\nA cat in an office files a tax return."
        if "Script Opposition" in text:
            self.stages.append("extractor.conflict")
            return "ordinary office work vs. absurd animal bureaucracy"
        if "two JSON lists" in text:
            self.stages.append("imaginator.summary")
            return '{"cat":["keyboard","office","deadline"],"keyboard":["office","deadline","audit"]}'
        if "cartoon description" in text and "Output JSON" in text:
            self.stages.append("imaginator.local")
            return '{"cat":["keyboard","office","deadline"]}'
        if "a cartoon" in text and "Output JSON" in text:
            self.stages.append("imaginator.global")
            return '{"cat":["keyboard","office","deadline"]}'
        if "select two most funny conflict" in text:
            self.stages.append("generator.select_conflict")
            return "##Conflict Scripts:\nordinary office work vs. absurd animal bureaucracy"
        if "Two key entities" in text:
            self.stages.append("generator.select_entities")
            return "[cat, keyboard]"
        self.stages.append("generator.caption")
        return "##Caption:\nThe cat has an audit trail.\n##Explanation:\nThe office joke is literal."


def run(*, online_description: bool) -> dict[str, object]:
    backend = SmokeBackend()
    pipeline = HomerPublicCodePipeline(
        backend,
        retriever=lambda summary, description, conflicts: summary,
        require_retrieval=True,
        successors=3,
    )
    kwargs = {} if online_description else {
        "standard_description": "A cat in an office files a tax return."
    }
    result = pipeline.run(image="/tmp/homer-smoke.jpg", seed=11, **kwargs)
    expected = (
        [
            "extractor.description", "extractor.conflict", "imaginator.global",
            "imaginator.local", "imaginator.summary", "generator.select_conflict",
            "generator.select_entities", "generator.caption",
        ]
        if online_description
        else [
            "extractor.conflict", "imaginator.global", "imaginator.local",
            "imaginator.summary", "generator.select_conflict",
            "generator.select_entities", "generator.caption",
        ]
    )
    if backend.stages != expected:
        raise RuntimeError(f"stage order mismatch: {backend.stages!r} != {expected!r}")
    return {
        "status": "pass",
        "description_mode": "online" if online_description else "standard_replay",
        "request_count": result.request_count,
        "stage_order": backend.stages,
        "caption_schema": ["##Caption", "##Explanation"],
        "caption": result.caption,
        "explanation": result.explanation,
        "seed": result.seed,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "online": run(online_description=True),
        "standard_replay": run(online_description=False),
        "scientific_evaluation": False,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
