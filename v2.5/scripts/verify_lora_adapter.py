#!/usr/bin/env python
"""Fail closed unless a saved PEFT LoRA adapter is structurally and numerically valid."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def verify_adapter(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing adapter weights: {weights_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Expected PEFT LoRA adapter, got peft_type={config.get('peft_type')!r}")

    tensor_count = 0
    element_count = 0
    lora_a_count = 0
    lora_b_count = 0
    dtypes: set[str] = set()
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"Adapter has no tensors: {weights_path}")
        for key in keys:
            tensor = handle.get_tensor(key)
            if tensor.numel() == 0:
                raise ValueError(f"Adapter tensor is empty: {key}")
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"Adapter tensor contains NaN/Inf: {key}")
            tensor_count += 1
            element_count += tensor.numel()
            dtypes.add(str(tensor.dtype))
            lora_a_count += ".lora_A." in key
            lora_b_count += ".lora_B." in key

    if lora_a_count == 0 or lora_a_count != lora_b_count:
        raise ValueError(
            f"Unbalanced LoRA tensors: lora_A={lora_a_count}, lora_B={lora_b_count}"
        )

    return {
        "adapter_dir": str(adapter_dir),
        "base_model": config.get("base_model_name_or_path"),
        "rank": config.get("r"),
        "tensor_count": tensor_count,
        "lora_a_count": lora_a_count,
        "lora_b_count": lora_b_count,
        "element_count": element_count,
        "file_bytes": weights_path.stat().st_size,
        "dtypes": sorted(dtypes),
        "all_finite": True,
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for adapter_dir in args.adapter_dirs:
        print(json.dumps(verify_adapter(adapter_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
