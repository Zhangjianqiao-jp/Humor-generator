#!/usr/bin/env python
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_inference import load_qwen_vl_lora_for_inference

DESCRIBE_PROMPT = "Describe the image in one specific sentence. Mention the main visible objects and action."
CAPTION_PROMPT = (
    "Generate one short, natural, image-specific humorous caption for this image. "
    "Do not explain."
)


def _generate(model, processor, messages, max_new_tokens: int, temperature: float) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = 0.9
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated_ids[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def _messages(image: Path, prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main() -> None:
    parser = ArgumentParser(description="Diagnose whether Qwen2.5-VL and the LoRA adapter are using image content.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-name", default="/home/zhang.jianqiao/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/lora_sft_v1_5/final_lora"))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    print("== Base model ==")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    base_processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    base_model.eval()
    print("[describe]", _generate(base_model, base_processor, _messages(args.image, DESCRIBE_PROMPT), args.max_new_tokens, 0.0))
    print("[caption ]", _generate(base_model, base_processor, _messages(args.image, CAPTION_PROMPT), args.max_new_tokens, 0.8))

    print("\n== LoRA adapter ==")
    lora_model, lora_processor, _ = load_qwen_vl_lora_for_inference(
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
    )
    print("[describe]", _generate(lora_model, lora_processor, _messages(args.image, DESCRIBE_PROMPT), args.max_new_tokens, 0.0))
    print("[caption ]", _generate(lora_model, lora_processor, _messages(args.image, CAPTION_PROMPT), args.max_new_tokens, 0.8))


if __name__ == "__main__":
    main()
