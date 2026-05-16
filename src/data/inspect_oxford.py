from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.image_utils import is_image_file
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def inspect_hic_root(hic_root: Path, sample_n: int = 3) -> None:
    hic_data = hic_root / "hic_data"
    if not hic_data.exists():
        LOGGER.warning("hic_data directory not found under %s", hic_root)
        return

    csv_files = sorted(hic_data.rglob("*.csv"))
    if not csv_files:
        LOGGER.warning("No CSV files found under %s", hic_data)
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            LOGGER.info("CSV: %s | shape=%s | columns=%s", csv_path, df.shape, list(df.columns))
            LOGGER.info("Sample rows:\n%s", df.head(sample_n).to_string(index=False))
        except Exception as exc:
            LOGGER.warning("Failed reading %s: %s", csv_path, exc)

    image_dir = hic_root / "images"
    if not image_dir.exists():
        LOGGER.warning("images directory not found under %s", hic_root)
        return
    img_files = [p for p in image_dir.rglob("*") if p.is_file() and is_image_file(p)]
    total_bytes = sum(p.stat().st_size for p in img_files)
    LOGGER.info("Image directory: %s | num_images=%d | size_mb=%.2f", image_dir, len(img_files), total_bytes / (1024 * 1024))
