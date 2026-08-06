"""Reference observation schema and local action safety boundary."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np

ACTION_FIELDS = ("dx_m", "dy_m", "dz_m", "dyaw_deg")


class SchemaError(ValueError):
    """An observation or action violates the declared UAV wire schema."""


def make_observation(
    *,
    images: Mapping[str, np.ndarray],
    state: np.ndarray,
    prompt: str = "",
    timestamp_ns: int | None = None,
    frame: str = "body_flu",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observation = {
        "timestamp_ns": time.time_ns() if timestamp_ns is None else int(timestamp_ns),
        "images": dict(images),
        "state": np.asarray(state, dtype=np.float32),
        "prompt": prompt,
        "frame": frame,
    }
    if extra:
        observation["extra"] = dict(extra)
    validate_observation(observation)
    return observation


def validate_observation(observation: Mapping[str, Any]) -> None:
    if not isinstance(observation.get("timestamp_ns"), int):
        raise SchemaError("timestamp_ns must be an integer")
    images = observation.get("images")
    if not isinstance(images, Mapping) or not images:
        raise SchemaError("images must be a non-empty mapping")
    for name, image in images.items():
        if not isinstance(name, str) or not name:
            raise SchemaError("image names must be non-empty strings")
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise SchemaError(f"image {name!r} must be a uint8 NumPy array")
        if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
            raise SchemaError(f"image {name!r} must have HWC layout")
    state = observation.get("state")
    if not isinstance(state, np.ndarray) or state.ndim != 1 or not np.issubdtype(state.dtype, np.floating):
        raise SchemaError("state must be a one-dimensional floating NumPy array")
    if not np.all(np.isfinite(state)):
        raise SchemaError("state contains NaN or infinity")
    if not isinstance(observation.get("prompt", ""), str):
        raise SchemaError("prompt must be a string")


@dataclass(frozen=True)
class LocalWaypointAction:
    """One body-FLU pose increment relative to the current aircraft pose."""

    dx_m: float
    dy_m: float
    dz_m: float
    dyaw_deg: float


@dataclass(frozen=True)
class ActionLimits:
    max_dx_m: float = 1.0
    max_dy_m: float = 1.0
    max_dz_m: float = 0.5
    max_dyaw_deg: float = 15.0


class ActionGuard:
    """Validate and clip remote model output before it reaches a flight SDK."""

    def __init__(self, limits: ActionLimits | None = None) -> None:
        self.limits = limits or ActionLimits()

    def first_safe_waypoint(self, result: Mapping[str, Any]) -> LocalWaypointAction:
        actions = result.get("actions")
        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)
        expected_shape = (1, len(ACTION_FIELDS))
        if actions.shape != expected_shape:
            raise SchemaError(f"actions must have shape {expected_shape}, got {actions.shape}")
        action = np.asarray(actions[0], dtype=np.float64)
        if not np.all(np.isfinite(action)):
            raise SchemaError("action contains NaN or infinity")
        limits = self.limits
        return LocalWaypointAction(
            dx_m=float(np.clip(action[0], -limits.max_dx_m, limits.max_dx_m)),
            dy_m=float(np.clip(action[1], -limits.max_dy_m, limits.max_dy_m)),
            dz_m=float(np.clip(action[2], -limits.max_dz_m, limits.max_dz_m)),
            dyaw_deg=float(np.clip(action[3], -limits.max_dyaw_deg, limits.max_dyaw_deg)),
        )

    def first_safe_action(self, result: Mapping[str, Any]) -> LocalWaypointAction:
        """Backward-compatible method name; the returned action is a waypoint."""
        return self.first_safe_waypoint(result)
