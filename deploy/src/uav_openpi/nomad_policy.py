"""Server-side NoMaD policy adapter for the UAV OpenPI protocol."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np
from PIL import Image

from uav_openpi.base_policy import BasePolicy
from uav_openpi.uav_schema import ACTION_FIELDS, validate_observation


DEFAULT_VISUALNAV_ROOT = Path("/path/to/visualnav-transformer")
DEFAULT_NOMAD_CONFIG = DEFAULT_VISUALNAV_ROOT / "train" / "config" / "nomad_world_model_combined_v5_2_learnangle.yaml"
DEFAULT_NOMAD_CHECKPOINT = (
    DEFAULT_VISUALNAV_ROOT
    / "train"
    / "logs"
    / "nomad"
    / "nomad_world_model_combined_v5_2_2026_06_30_23_27_09"
    / "ema_latest.pth"
)


@dataclass(frozen=True)
class NoMaDPolicyConfig:
    visualnav_root: Path = DEFAULT_VISUALNAV_ROOT
    nomad_config: Path = DEFAULT_NOMAD_CONFIG
    nomad_checkpoint: Path = DEFAULT_NOMAD_CHECKPOINT
    device: str = "cuda"
    num_samples: int = 8
    select_proposal: Literal["first", "mean"] = "mean"
    image_key: str = "front"
    goal_image_key: str = "goal"
    input_color_space: Literal["rgb", "bgr"] = "rgb"
    waypoint_index: int = 0
    translation_scale: float = 1.0
    yaw_scale: float = 1.0
    model_frame: Literal["body_frd", "body_flu"] = "body_frd"


class NoMaDPolicy(BasePolicy):
    """NoMaD image-goal policy exposed as ``BasePolicy``.

    Observation requirements:
    - ``images[image_key]`` is the current front image, uint8 HWC.
    - ``images[goal_image_key]`` is the goal image, uint8 HWC.

    The policy keeps episode-local visual context internally. Call ``reset`` at
    the start of each flight episode. This matches the current server model of
    one active aircraft client per policy process.
    """

    def __init__(self, config: NoMaDPolicyConfig | None = None) -> None:
        self.config = config or NoMaDPolicyConfig()
        _validate_policy_config(self.config)
        self._helpers = _load_nomad_helpers(self.config.visualnav_root)
        self._torch = _import_torch()
        self._device = self._torch_device(self.config.device)
        self._nomad_config = self._helpers.merged_config(self.config.nomad_config)
        self._model = self._helpers.build_model(self._nomad_config, self.config.nomad_checkpoint, self._device)
        self._scheduler = self._helpers.build_scheduler(self._nomad_config)
        self._pos_dimension = int(self._nomad_config.get("pos_dimension", 2))
        self._learn_angle = bool(self._nomad_config["learn_angle"])
        self._action_stats = self._helpers.load_action_stats(
            self.config.visualnav_root / "train" / "vint_train" / "data" / "data_config.yaml",
            pos_dimension=self._pos_dimension,
            learn_angle=self._learn_angle,
        )
        data_cfg = self._helpers.load_yaml(
            self.config.visualnav_root / "train" / "vint_train" / "data" / "data_config.yaml"
        )
        dataset_name = next(iter(self._nomad_config.get("datasets", {"world_model_combined_v5_2": {}}).keys()))
        self._metric_spacing = float(data_cfg[dataset_name]["metric_waypoint_spacing"])
        self._context_len = int(self._nomad_config["context_size"]) + 1
        self._image_size = tuple(int(v) for v in self._nomad_config["image_size"])
        self._waypoint_index = int(self.config.waypoint_index)
        if self._waypoint_index >= int(self._nomad_config["len_traj_pred"]):
            raise ValueError(
                f"waypoint_index {self._waypoint_index} is outside NoMaD horizon "
                f"{self._nomad_config['len_traj_pred']}"
            )
        self._history: deque[np.ndarray] = deque(maxlen=self._context_len)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": "nomad_policy",
            "model_version": "world_model_combined_v5_2_learnangle_2026_06_30",
            "nomad_config": str(self.config.nomad_config),
            "nomad_checkpoint": str(self.config.nomad_checkpoint),
            "num_samples": int(self.config.num_samples),
            "select_proposal": self.config.select_proposal,
            "context_len": self._context_len,
            "image_keys": {"current": self.config.image_key, "goal": self.config.goal_image_key},
            "input_color_space": self.config.input_color_space,
            "model_coordinate_frame": {
                "name": self.config.model_frame,
                "origin": "current_observation_pose",
                "waypoint_type": "cumulative_local_position",
            },
            "output_coordinate_frame": "body_flu",
            "waypoint_conversion": {
                "metric_spacing": float(self._metric_spacing),
                "waypoint_index": self._waypoint_index,
                "translation_scale": float(self.config.translation_scale),
                "yaw_scale": float(self.config.yaw_scale),
            },
            "action_schema": {
                "fields": list(ACTION_FIELDS),
                "shape": [1, len(ACTION_FIELDS)],
                "frame": "body_flu",
                "representation": "local_pose_delta",
            },
        }

    def reset(self, episode_id: str | None = None) -> None:
        del episode_id
        self._history.clear()

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        validate_observation(obs)
        images = obs["images"]
        if self.config.image_key not in images:
            raise KeyError(f"observation images missing current image key {self.config.image_key!r}")
        if self.config.goal_image_key not in images:
            raise KeyError(f"observation images missing goal image key {self.config.goal_image_key!r}")

        current = _as_rgb(images[self.config.image_key], self.config.input_color_space)
        goal = _as_rgb(images[self.config.goal_image_key], self.config.input_color_space)
        self._history.append(current)
        history = _pad_history(list(self._history), self._context_len)

        waypoints = self._sample_waypoints(history, goal)
        local_waypoints = waypoints_to_local_waypoints(
            waypoints,
            metric_spacing=self._metric_spacing,
            pos_dimension=self._pos_dimension,
            learn_angle=self._learn_angle,
        )
        selected_idx = select_action_sequence(local_waypoints, self.config.select_proposal)
        selected_model_local = local_waypoints[selected_idx]
        selected_output_local = convert_local_waypoints_frame(
            selected_model_local,
            source_frame=self.config.model_frame,
            target_frame="body_flu",
        )
        unscaled_waypoint_actions = local_waypoints_to_wire_actions(selected_output_local)
        scaled_waypoint_actions = scale_waypoint_actions(
            unscaled_waypoint_actions,
            translation_scale=float(self.config.translation_scale),
            yaw_scale=float(self.config.yaw_scale),
        )
        selected_action = scaled_waypoint_actions[self._waypoint_index : self._waypoint_index + 1]
        return {
            "actions": selected_action.astype(np.float32),
            "goal_reached": False,
            "debug": {
                "selected_index": int(selected_idx),
                "waypoint_index": self._waypoint_index,
                "raw_model_output": waypoints[selected_idx].astype(float).tolist(),
                "model_local_waypoints_m_rad": selected_model_local.astype(float).tolist(),
                "output_local_waypoints_flu_m_rad": selected_output_local.astype(float).tolist(),
                "unscaled_waypoint_actions": unscaled_waypoint_actions.astype(float).tolist(),
                "scaled_waypoint_actions": scaled_waypoint_actions.astype(float).tolist(),
                "selected_waypoint_action": selected_action[0].astype(float).tolist(),
                "num_candidates": int(local_waypoints.shape[0]),
                "metric_spacing": float(self._metric_spacing),
                "translation_scale": float(self.config.translation_scale),
                "yaw_scale": float(self.config.yaw_scale),
                "model_coordinate_frame": self.config.model_frame,
                "output_coordinate_frame": "body_flu",
            },
        }

    def _sample_waypoints(self, history_rgb: list[np.ndarray], goal_rgb: np.ndarray) -> np.ndarray:
        obs = self._torch.cat([self._image_tensor(image) for image in history_rgb[-self._context_len :]], dim=0)
        obs = self._helpers.normalize_obs_image(obs).unsqueeze(0)
        goal = self._helpers.normalize_obs_image(self._image_tensor(goal_rgb)).unsqueeze(0)
        return self._helpers.sample_actions(
            model=self._model,
            scheduler=self._scheduler,
            obs_images=obs,
            goal_image=goal,
            config=self._nomad_config,
            action_stats=self._action_stats,
            num_samples=int(self.config.num_samples),
            device=self._device,
        ).astype(np.float32)

    def _image_tensor(self, image_rgb: np.ndarray) -> Any:
        return self._helpers.resize_and_aspect_crop(Image.fromarray(image_rgb), self._image_size)

    def _torch_device(self, requested: str) -> Any:
        if requested == "cuda" and not self._torch.cuda.is_available():
            return self._torch.device("cpu")
        return self._torch.device(requested)


# Kept module-level and dependency-light so tests can cover conversion without loading NoMaD.
def waypoints_to_local_waypoints(
    waypoints: np.ndarray,
    *,
    metric_spacing: float,
    pos_dimension: int,
    learn_angle: bool,
) -> np.ndarray:
    local_waypoints_m = waypoints[:, :, :pos_dimension].astype(np.float32) * float(metric_spacing)
    if pos_dimension == 2:
        zeros_z = np.zeros((*local_waypoints_m.shape[:2], 1), dtype=np.float32)
        local_waypoints_m = np.concatenate([local_waypoints_m, zeros_z], axis=-1)
    if learn_angle and waypoints.shape[-1] >= pos_dimension + 2:
        cos_sin = waypoints[:, :, pos_dimension : pos_dimension + 2]
        relative_yaw = np.arctan2(cos_sin[:, :, 1], cos_sin[:, :, 0]).astype(np.float32)
        relative_yaw = np.arctan2(np.sin(relative_yaw), np.cos(relative_yaw))[..., None].astype(np.float32)
    else:
        relative_yaw = np.zeros((*local_waypoints_m.shape[:2], 1), dtype=np.float32)
    return np.concatenate([local_waypoints_m, relative_yaw], axis=-1).astype(np.float32)


def select_action_sequence(actions: np.ndarray, mode: str) -> int:
    if actions.shape[0] <= 1 or mode == "first":
        return 0
    if mode == "mean":
        center = actions.mean(axis=0, keepdims=True)
        distances = np.linalg.norm((actions - center).reshape(actions.shape[0], -1), axis=1)
        return int(np.argmin(distances))
    raise ValueError(f"unsupported NoMaD proposal selection mode {mode!r}")


def convert_local_waypoints_frame(
    local_waypoints: np.ndarray,
    *,
    source_frame: str,
    target_frame: str,
) -> np.ndarray:
    """Convert xyz and yaw deltas between body FRD and body FLU."""
    waypoints = np.asarray(local_waypoints, dtype=np.float32)
    if waypoints.ndim != 2 or waypoints.shape[1] < 4:
        raise ValueError("local waypoints must have shape (horizon, >=4)")
    supported = {"body_frd", "body_flu"}
    if source_frame not in supported or target_frame not in supported:
        raise ValueError(f"unsupported coordinate frame conversion {source_frame!r} -> {target_frame!r}")
    converted = waypoints.copy()
    if source_frame != target_frame:
        converted[:, 1:4] *= -1.0
    return converted


def local_waypoints_to_wire_actions(local_waypoints: np.ndarray) -> np.ndarray:
    waypoints = np.asarray(local_waypoints, dtype=np.float32)
    if waypoints.ndim != 2 or waypoints.shape[1] != len(ACTION_FIELDS):
        raise ValueError(f"local waypoints must have shape (horizon, {len(ACTION_FIELDS)})")
    actions = waypoints.copy()
    actions[:, 3] = np.degrees(actions[:, 3])
    return actions


def scale_waypoint_actions(
    waypoint_actions: np.ndarray,
    *,
    translation_scale: float,
    yaw_scale: float,
) -> np.ndarray:
    """Scale a body-FLU local pose increment."""
    actions = np.asarray(waypoint_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != len(ACTION_FIELDS):
        raise ValueError(f"waypoint actions must have shape (horizon, {len(ACTION_FIELDS)})")
    translation_scale = float(translation_scale)
    yaw_scale = float(yaw_scale)
    if not np.isfinite(translation_scale) or translation_scale < 0:
        raise ValueError("translation_scale must be finite and non-negative")
    if not np.isfinite(yaw_scale) or yaw_scale < 0:
        raise ValueError("yaw_scale must be finite and non-negative")
    scaled = actions.copy()
    scaled[:, :3] *= translation_scale
    scaled[:, 3] *= yaw_scale
    return scaled


def create_policy_from_env() -> NoMaDPolicy:
    return NoMaDPolicy(
        NoMaDPolicyConfig(
            visualnav_root=Path(os.environ.get("NOMAD_VISUALNAV_ROOT", str(DEFAULT_VISUALNAV_ROOT))),
            nomad_config=Path(os.environ.get("NOMAD_CONFIG", str(DEFAULT_NOMAD_CONFIG))),
            nomad_checkpoint=Path(os.environ.get("NOMAD_CHECKPOINT", str(DEFAULT_NOMAD_CHECKPOINT))),
            device=os.environ.get("NOMAD_DEVICE", "cuda"),
            num_samples=int(os.environ.get("NOMAD_NUM_SAMPLES", "8")),
            select_proposal=os.environ.get("NOMAD_SELECT_PROPOSAL", "mean"),
            image_key=os.environ.get("NOMAD_IMAGE_KEY", "front"),
            goal_image_key=os.environ.get("NOMAD_GOAL_IMAGE_KEY", "goal"),
            input_color_space=os.environ.get("NOMAD_INPUT_COLOR_SPACE", "rgb"),
            waypoint_index=int(os.environ.get("NOMAD_WAYPOINT_INDEX", "0")),
            translation_scale=float(os.environ.get("NOMAD_TRANSLATION_SCALE", "1.0")),
            yaw_scale=float(os.environ.get("NOMAD_YAW_SCALE", "1.0")),
            model_frame=os.environ.get("NOMAD_MODEL_FRAME", "body_frd"),
        )
    )


def create_goal_aware_policy_from_env() -> BasePolicy:
    """Create NoMaD wrapped with server-side final-goal detection."""
    from uav_openpi.goal_aware_policy import GoalAwarePolicy, goal_termination_config_from_env

    return GoalAwarePolicy(create_policy_from_env(), goal_termination_config_from_env())


def _validate_policy_config(config: NoMaDPolicyConfig) -> None:
    if config.model_frame not in {"body_frd", "body_flu"}:
        raise ValueError(f"unsupported NoMaD model frame {config.model_frame!r}")
    if config.select_proposal not in {"first", "mean"}:
        raise ValueError(f"unsupported NoMaD proposal selection mode {config.select_proposal!r}")
    numeric_non_negative = {
        "translation_scale": config.translation_scale,
        "yaw_scale": config.yaw_scale,
    }
    for name, value in numeric_non_negative.items():
        if not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if int(config.waypoint_index) < 0:
        raise ValueError("waypoint_index must be non-negative")
    if int(config.num_samples) < 1:
        raise ValueError("num_samples must be positive")


def _load_nomad_helpers(visualnav_root: Path) -> Any:
    helper_path = visualnav_root / "train" / "generate_nomad_test_trajs.py"
    for path in (visualnav_root, visualnav_root / "diffusion_policy", visualnav_root / "train"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    spec = importlib.util.spec_from_file_location("uav_openpi_nomad_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load NoMaD helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["uav_openpi_nomad_helpers"] = module
    spec.loader.exec_module(module)
    return module


def _import_torch() -> Any:
    import torch

    return torch


def _as_rgb(image: np.ndarray, color_space: str) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("NoMaD images must be uint8 HWC RGB/BGR arrays with 3 or 4 channels")
    image = image[..., :3]
    if color_space == "rgb":
        return image.copy()
    if color_space == "bgr":
        return image[..., ::-1].copy()
    raise ValueError(f"unsupported color_space {color_space!r}")


def _pad_history(history: list[np.ndarray], context_len: int) -> list[np.ndarray]:
    if not history:
        raise ValueError("cannot build NoMaD context from empty image history")
    return [history[0].copy() for _ in range(max(0, context_len - len(history)))] + history
