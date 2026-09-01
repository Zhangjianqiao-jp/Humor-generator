"""Memory-efficient capture of generated-token states from a frozen planner."""
from __future__ import annotations

from typing import Any

import torch


def _unwrap_candidates(model: Any) -> list[Any]:
    candidates = [model]
    seen = {id(model)}
    for _ in range(8):
        added = []
        for candidate in candidates:
            for name in ("base_model", "model", "language_model"):
                child = getattr(candidate, name, None)
                if child is not None and id(child) not in seen:
                    seen.add(id(child))
                    added.append(child)
        if not added:
            break
        candidates.extend(added)
    return candidates


def find_last_decoder_layer(model: Any) -> Any:
    """Locate the final language decoder block through PEFT/Qwen wrappers."""
    for candidate in reversed(_unwrap_candidates(model)):
        layers = getattr(candidate, "layers", None)
        if layers is not None and len(layers):
            return layers[-1]
        transformer = getattr(candidate, "transformer", None)
        blocks = getattr(transformer, "h", None) if transformer is not None else None
        if blocks is not None and len(blocks):
            return blocks[-1]
    raise ValueError(f"Cannot locate decoder layers under {type(model).__name__}")


class GeneratedTokenStateCapture:
    """Forward hook that retains only the last position of each decode call."""

    def __init__(self, *, detach: bool = True, output_device: str | torch.device = "cpu") -> None:
        self.detach = detach
        self.output_device = torch.device(output_device)
        self.states: list[torch.Tensor] = []

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim not in (2, 3):
            return
        state = hidden[:, -1:, :] if hidden.ndim == 3 else hidden[:, None, :]
        if self.detach:
            state = state.detach()
        self.states.append(state.to(self.output_device))

    def clear(self) -> None:
        self.states.clear()

    def stacked(self, *, keep_last: int | None = None) -> torch.Tensor:
        if not self.states:
            raise RuntimeError("No planner hidden states were captured")
        result = torch.cat(self.states, dim=1)
        if keep_last is not None:
            if keep_last < 1:
                raise ValueError("keep_last must be positive")
            result = result[:, -keep_last:, :]
        return result
