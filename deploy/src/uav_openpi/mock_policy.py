"""Small deterministic server-side policy for transport and hardware tests."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from uav_openpi.base_policy import BasePolicy
from uav_openpi.uav_schema import ACTION_FIELDS, validate_observation


class ConstantWaypointPolicy(BasePolicy):
    def __init__(
        self,
        *,
        dx_m: float = 0.1,
        dy_m: float = 0.0,
        dz_m: float = 0.0,
        dyaw_deg: float = 0.0,
    ) -> None:
        self._action = np.array([dx_m, dy_m, dz_m, dyaw_deg], dtype=np.float32)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": "constant_local_waypoint_mock",
            "model_version": "2",
            "observation_schema": {
                "images": "mapping[str, uint8[H,W,C]]",
                "state": "float32[state_dim]",
                "timestamp_ns": "int",
                "prompt": "str",
                "frame": "body_flu",
            },
            "action_schema": {
                "fields": list(ACTION_FIELDS),
                "shape": [1, len(ACTION_FIELDS)],
                "frame": "body_flu",
                "representation": "local_pose_delta",
            },
            "result_schema": {"goal_reached": "bool"},
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        validate_observation(obs)
        started = time.monotonic()
        return {
            "actions": self._action[None, :].copy(),
            "goal_reached": False,
            "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000},
        }


class WaypointPolicy(BasePolicy):
    """Tiny closed-loop mock policy that steers toward a target position.

    The reference mock state layout uses the first four state entries as
    ``x_m, y_m, z_m, yaw_deg``. This is intentionally simple and deterministic;
    it is for transport/runtime tests, not flight control.
    """

    def __init__(
        self,
        *,
        target_xyz: tuple[float, float, float] = (1.0, 0.0, 0.0),
        max_step_m: float = 0.5,
        max_dyaw_deg: float = 15.0,
        goal_tolerance_m: float = 0.05,
    ) -> None:
        self._target_xyz = np.asarray(target_xyz, dtype=np.float32)
        self._max_step_m = float(max_step_m)
        self._max_dyaw_deg = float(max_dyaw_deg)
        self._goal_tolerance_m = float(goal_tolerance_m)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": "waypoint_mock",
            "model_version": "2",
            "target_xyz": self._target_xyz.tolist(),
            "observation_schema": {
                "images": "mapping[str, uint8[H,W,C]]",
                "state": "float32[state_dim], first entries are x_m, y_m, z_m, yaw_deg",
                "timestamp_ns": "int",
                "prompt": "str",
                "frame": "body_flu",
            },
            "action_schema": {
                "fields": list(ACTION_FIELDS),
                "shape": [1, len(ACTION_FIELDS)],
                "frame": "body_flu",
                "representation": "local_pose_delta",
            },
            "result_schema": {"goal_reached": "bool"},
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        validate_observation(obs)
        started = time.monotonic()
        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape[0] < 4:
            raise ValueError("waypoint mock expects state[0:4] = x_m, y_m, z_m, yaw_deg")

        position = state[:3]
        delta_world = self._target_xyz - position
        distance_to_goal = float(np.linalg.norm(delta_world))
        if distance_to_goal <= self._goal_tolerance_m:
            return {
                "goal_reached": True,
                "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000},
                "debug": {
                    "position_xyz": position.astype(float).tolist(),
                    "target_xyz": self._target_xyz.astype(float).tolist(),
                    "distance_m": distance_to_goal,
                },
            }
        yaw_rad = np.radians(float(state[3]))
        cos_yaw = float(np.cos(yaw_rad))
        sin_yaw = float(np.sin(yaw_rad))
        delta_local = np.array(
            [
                delta_world[0] * cos_yaw + delta_world[1] * sin_yaw,
                -delta_world[0] * sin_yaw + delta_world[1] * cos_yaw,
                delta_world[2],
            ],
            dtype=np.float32,
        )
        distance = float(np.linalg.norm(delta_local))
        if distance > self._max_step_m > 0:
            delta_local *= self._max_step_m / distance

        horizontal_norm = float(np.linalg.norm(delta_world[:2]))
        desired_yaw_deg = (
            float(np.degrees(np.arctan2(delta_world[1], delta_world[0])))
            if horizontal_norm > 1e-6
            else float(state[3])
        )
        yaw_error = ((desired_yaw_deg - float(state[3]) + 180.0) % 360.0) - 180.0
        dyaw_deg = float(np.clip(yaw_error, -self._max_dyaw_deg, self._max_dyaw_deg))

        action = np.array([delta_local[0], delta_local[1], delta_local[2], dyaw_deg], dtype=np.float32)
        return {
            "actions": action[None, :],
            "goal_reached": False,
            "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000},
            "debug": {
                "position_xyz": position.astype(float).tolist(),
                "target_xyz": self._target_xyz.astype(float).tolist(),
                "distance_m": distance_to_goal,
                "local_waypoint_flu": action.astype(float).tolist(),
            },
        }
