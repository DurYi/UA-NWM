"""Dependency-light hardware factories for local runtime smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from uav_openpi.uav_schema import LocalWaypointAction, make_observation


class ImageSequenceObservationSource:
    """Read a fixed sequence of RGB files and repeat the final frame."""

    def __init__(self, front_rgb_paths: Sequence[str], prompt: str = "navigate to the goal image") -> None:
        if not front_rgb_paths:
            raise ValueError("front_rgb_paths must contain at least one image")
        self._paths = [Path(value).expanduser().resolve() for value in front_rgb_paths]
        missing = [path for path in self._paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        self._images = [self._load_rgb(path) for path in self._paths]
        self._prompt = prompt
        self._step = 0

    def read(self) -> dict:
        index = min(self._step, len(self._images) - 1)
        image = self._images[index]
        observation = make_observation(
            images={"front": image},
            state=np.zeros(12, dtype=np.float32),
            prompt=self._prompt,
            extra={"mock_frame_path": str(self._paths[index]), "mock_step": self._step},
        )
        self._step += 1
        return observation

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)


class LoggingFlightController:
    """Print commands instead of touching hardware."""

    def __init__(self) -> None:
        self.waypoints: list[LocalWaypointAction] = []
        self.hover_count = 0

    def send_local_waypoint(self, waypoint: LocalWaypointAction) -> None:
        self.waypoints.append(waypoint)
        print(f"MOCK send_local_waypoint waypoint={waypoint}")

    def hover(self) -> None:
        self.hover_count += 1
        print("MOCK hover")


def create_image_observation_source(
    front_rgb_paths: Sequence[str], prompt: str = "navigate to the goal image"
) -> ImageSequenceObservationSource:
    return ImageSequenceObservationSource(front_rgb_paths=front_rgb_paths, prompt=prompt)


def create_logging_flight_controller() -> LoggingFlightController:
    return LoggingFlightController()
