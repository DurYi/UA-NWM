from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(output_dir: str | Path | None = None, rank: int = 0) -> logging.Logger:
    logger = logging.getLogger("dino_feature_ar_wm")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if output_dir is not None and rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(output_dir) / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
