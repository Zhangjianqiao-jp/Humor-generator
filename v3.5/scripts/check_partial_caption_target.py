#!/usr/bin/env python3
"""Offline check through the real processor; no model weights, GPU or API."""
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from humor_generator_v35.homer.official_prompts import caption_messages
from humor_generator_v35.qwen_backend import QwenBackend
from humor_generator_v35.training.formal_bridge import _prompt_and_full


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifests/local_qwen2_5_vl_7b.json").read_text())
    processor = AutoProcessor.from_pretrained(manifest["local_snapshot_path"], local_files_only=True)
    # encode() needs a device, but this check never executes receiver.forward().
    backend = QwenBackend(torch.nn.Linear(1, 1), processor, lambda messages: (None, None))
    answer = "##Caption: I work for treats."
    results = []
    targets_by_condition = []
    for conflict in ("", "office work vs animal instincts"):
        messages = caption_messages("A cat sits at an office desk.", conflict, "")
        prompt, full, targets = _prompt_and_full(backend, messages, answer, partial_answer=True)
        text = processor.tokenizer.decode(targets[0], skip_special_tokens=False)
        assert text == answer, repr(text)
        assert not targets.eq(processor.tokenizer.eos_token_id).any()
        assert full["input_ids"].shape[1] == prompt["input_ids"].shape[1] + targets.shape[1]
        targets_by_condition.append(targets)
        results.append({"target": text, "target_tokens": targets.shape[1], "contains_eos": False})
    assert torch.equal(*targets_by_condition)
    print(json.dumps({"status": "pass", "processor": str(manifest["local_snapshot_path"]),
                      "revision": manifest["revision"], "conditions": results,
                      "scope": "production encode and _prompt_and_full, local processor only; no receiver forward"}, indent=2))


if __name__ == "__main__":
    main()
