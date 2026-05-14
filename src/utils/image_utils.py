from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS
