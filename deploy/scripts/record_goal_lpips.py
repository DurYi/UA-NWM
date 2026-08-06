#!/usr/bin/env python3
"""Record LPIPS distances for image-goal threshold calibration without stopping navigation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
from PIL import Image

from uav_openpi.goal_termination import GoalTerminationConfig, ImageGoalScorer
from uav_openpi.image_tools import resize_long_edge


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def collect_front_images(files: list[str], directories: list[str], recursive: bool) -> list[Path]:
    paths = [Path(value).expanduser() for value in files]
    pattern = "**/*" if recursive else "*"
    for directory_value in directories:
        directory = Path(directory_value).expanduser()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        paths.extend(
            path
            for path in directory.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    unique = sorted({path.resolve() for path in paths})
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"front image does not exist: {missing[0]}")
    if not unique:
        raise ValueError("no front images found; use --front-rgb or --front-dir")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Record current-to-goal LPIPS distances; never sends stop commands.")
    parser.add_argument("--goal-rgb", required=True, help="Fixed RGB goal image.")
    parser.add_argument("--front-rgb", action="append", default=[], help="Front image; may be passed more than once.")
    parser.add_argument("--front-dir", action="append", default=[], help="Directory of recorded front images.")
    parser.add_argument("--recursive", action="store_true", help="Search front directories recursively.")
    parser.add_argument("--output-csv", default="logs/goal_lpips_calibration.csv")
    parser.add_argument("--device", default="auto", help="LPIPS device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--max-image-edge",
        type=int,
        default=640,
        help="Match the online client's transport downsampling; use 0 to disable.",
    )
    args = parser.parse_args()

    goal_path = Path(args.goal_rgb).expanduser().resolve()
    if not goal_path.is_file():
        raise FileNotFoundError(goal_path)
    if args.max_image_edge < 0:
        raise ValueError("--max-image-edge must be non-negative")
    front_paths = collect_front_images(args.front_rgb, args.front_dir, args.recursive)
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be positive")
        front_paths = front_paths[: args.max_frames]

    goal_image = load_rgb(goal_path)
    if args.max_image_edge:
        goal_image = resize_long_edge(goal_image, args.max_image_edge)
    scorer = ImageGoalScorer(
        GoalTerminationConfig(method="lpips", lpips_device=args.device),
        goal_image=goal_image,
    )
    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "timestamp_ns",
        "front_rgb",
        "goal_rgb",
        "width",
        "height",
        "lpips_distance",
        "score_time_ms",
        "lpips_net",
        "device",
        "preprocess",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for step, front_path in enumerate(front_paths):
            image = load_rgb(front_path)
            if args.max_image_edge:
                image = resize_long_edge(image, args.max_image_edge)
            started = time.perf_counter()
            distance = scorer.score(image)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            row = {
                "step": step,
                "timestamp_ns": time.time_ns(),
                "front_rgb": str(front_path),
                "goal_rgb": str(goal_path),
                "width": image.shape[1],
                "height": image.shape[0],
                "lpips_distance": f"{distance:.8f}",
                "score_time_ms": f"{elapsed_ms:.3f}",
                "lpips_net": scorer.config.lpips_net,
                "device": scorer.device,
                "preprocess": (
                    f"long_edge_{args.max_image_edge},center_crop_4:3,resize_224x224,range_-1_to_1"
                    if args.max_image_edge
                    else "center_crop_4:3,resize_224x224,range_-1_to_1"
                ),
            }
            writer.writerow(row)
            csv_file.flush()
            print(
                f"step={step:04d} lpips={distance:.6f} score_ms={elapsed_ms:.2f} "
                f"front={front_path}"
            )
    print(f"recorded {len(front_paths)} LPIPS scores to {output_path}")


if __name__ == "__main__":
    main()
