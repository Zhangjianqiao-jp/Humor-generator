from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from scripts.verify_lora_adapter import verify_adapter


def write_adapter(tmp_path, tensors):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "base_model_name_or_path": "base"}),
        encoding="utf-8",
    )
    save_file(tensors, adapter_dir / "adapter_model.safetensors")
    return adapter_dir


def test_verify_adapter_accepts_balanced_finite_lora(tmp_path) -> None:
    adapter_dir = write_adapter(
        tmp_path,
        {
            "layer.lora_A.weight": torch.ones(2, 3),
            "layer.lora_B.weight": torch.ones(3, 2),
        },
    )

    report = verify_adapter(adapter_dir)

    assert report["tensor_count"] == 2
    assert report["element_count"] == 12
    assert report["all_finite"] is True


def test_verify_adapter_rejects_nonfinite_tensor(tmp_path) -> None:
    adapter_dir = write_adapter(
        tmp_path,
        {
            "layer.lora_A.weight": torch.tensor([[float("nan")]]),
            "layer.lora_B.weight": torch.ones(1, 1),
        },
    )

    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        verify_adapter(adapter_dir)
