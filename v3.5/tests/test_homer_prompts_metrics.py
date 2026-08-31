from __future__ import annotations

from humor_generator_v35.homer.metrics import unbiased_pass_at_k
from humor_generator_v35.homer.prompts import CAPTION_SYSTEM, caption_messages, global_imagination_messages


def test_caption_instruction_is_a_system_message() -> None:
    messages = caption_messages("scene", "normal vs. absurd", ["cup", "milk", "cow"])
    assert messages[0]["role"] == "system"
    assert messages[0]["content"][0]["text"] == CAPTION_SYSTEM
    assert messages[1]["role"] == "user"


def test_global_view_contains_image_and_conflict() -> None:
    messages = global_imagination_messages("cartoon.png", "power vs. danger")
    assert messages[1]["content"][0] == {"type": "image", "image": "cartoon.png"}


def test_unbiased_pass_at_k() -> None:
    assert unbiased_pass_at_k(5, 0, 3) == 0.0
    assert unbiased_pass_at_k(5, 5, 3) == 1.0
    assert abs(unbiased_pass_at_k(5, 1, 3) - 0.6) < 1e-9
