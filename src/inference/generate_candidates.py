from __future__ import annotations

from pathlib import Path

from src.inference.mock_generator import generate_mock_candidates
from src.models.qwen_vl_loader import load_qwen_model
from src.utils.io import read_jsonl, write_jsonl

PROMPT = "Generate a short humorous caption for this image. Make it image-specific, concise, and avoid generic meme templates."


def generate_candidates(input_jsonl: Path, output_jsonl: Path, num_candidates: int = 10, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", dry_run: bool = True) -> None:
    rows = read_jsonl(input_jsonl)
    out = []

    model_bundle = None
    if not dry_run:
        model_bundle = load_qwen_model(model_name=model_name)
    for row in rows:
        image = row["image"]
        image_id = row["image_id"]
        if dry_run:
            candidates = generate_mock_candidates(image, num_candidates)
            generator_name = "mock"
        else:
            model, processor, process_vision_info = model_bundle
            # Minimal inference stub for future extension.
            # TODO(V2): add batched generation and decoding optimizations.
            candidates = []
            for _ in range(num_candidates):
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
                generated_ids = model.generate(**inputs, max_new_tokens=48)
                decoded = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                candidates.append(decoded.strip())
            generator_name = model_name
        out.append({"image": image, "image_id": image_id, "candidates": candidates, "meta": {"generator": generator_name, "num_candidates": num_candidates}})
    write_jsonl(output_jsonl, out)
