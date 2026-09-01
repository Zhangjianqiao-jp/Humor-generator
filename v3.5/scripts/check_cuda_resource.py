#!/usr/bin/env python3
"""Fail closed unless a formal bridge job has one stable full CUDA device."""
from __future__ import annotations

import json
import os

import torch


def main() -> None:
    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(f"formal bridge job requires exactly one visible GPU; got {count}")
    torch.cuda.set_device(0)
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info(0)
    gib = 1024**3
    report = {
        "status": "pass",
        "device_count": count,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "allocator_backend": torch.cuda.get_allocator_backend(),
        "free_gib": free / gib,
        "total_gib": total / gib,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    if report["allocator_backend"] != "native":
        raise RuntimeError(f"formal bridge job requires native allocator: {report}")
    # A MIG slice reports the slice capacity, not the physical GPU capacity.
    # v3.5 formal training requires at least a 40-GiB full-device allocation.
    if total < 40 * gib:
        raise RuntimeError(f"formal bridge job received a MIG-sized device: {report}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
