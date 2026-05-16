from pathlib import Path

TEMPLATES = [
    "When the image decided regular captions were too serious.",
    "A perfectly normal scene, according to absolutely no one.",
    "Everyone in this photo agreed to chaos and signed nothing.",
    "Plot twist: this was the plan all along.",
    "Somehow this makes sense before coffee.",
]


def generate_mock_candidates(image_path: str | Path, num_candidates: int = 10) -> list[str]:
    del image_path
    out = []
    for i in range(num_candidates):
        out.append(TEMPLATES[i % len(TEMPLATES)])
    return out
