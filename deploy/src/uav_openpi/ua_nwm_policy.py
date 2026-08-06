"""Server adapter for UA-NWM HEP-based MPC-CEM navigation."""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import logging
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from PIL import Image

from uav_openpi.base_policy import BasePolicy
from uav_openpi.nomad_policy import (
    convert_local_waypoints_frame,
    local_waypoints_to_wire_actions,
    scale_waypoint_actions,
)
from uav_openpi.uav_schema import ACTION_FIELDS, validate_observation


logger = logging.getLogger(__name__)

DEFAULT_UANWM_ROOT = Path("/path/to/UA-NWM")
DEFAULT_UANWM_CONFIG = DEFAULT_UANWM_ROOT / "configs" / "ua_nwm.yaml"
DEFAULT_UANWM_CHECKPOINT = DEFAULT_UANWM_ROOT / "pretrained" / "ua-nwm" / "ua-nwm_best.pt"
DEFAULT_UANWM_ACTION_STATS = DEFAULT_UANWM_ROOT / "pretrained" / "ua-nwm" / "action_stats.json"


@dataclass(frozen=True)
class UANWMRealNavConfig:
    uanwm_root: Path = DEFAULT_UANWM_ROOT
    model_config: Path = DEFAULT_UANWM_CONFIG
    checkpoint: Path = DEFAULT_UANWM_CHECKPOINT
    action_stats: Path = DEFAULT_UANWM_ACTION_STATS
    device: str = "cuda"
    image_key: str = "front"
    goal_image_key: str = "goal"
    input_color_space: Literal["rgb", "bgr"] = "rgb"
    model_frame: Literal["body_frd", "body_flu"] = "body_frd"
    horizon: int = 8
    cem_candidates: int = 32
    cem_iterations: int = 3
    elite_ratio: float = 0.1
    score_batch_size: int = 4
    action_mean: tuple[float, float, float, float] = (4.7, 0.0, -0.8, 0.0)
    action_std: tuple[float, float, float, float] = (1.5, 1.0, 1.0, 0.35)
    action_low: tuple[float, float, float, float] = (0.0, -3.0, -3.0, -0.785398)
    action_high: tuple[float, float, float, float] = (7.5, 3.0, 3.0, 0.785398)
    goal_hint_mode: Literal["none", "extra_flu"] = "none"
    goal_hint_key: str = "goal_delta_body_flu"
    translation_scale: float = 1.0
    yaw_scale: float = 1.0
    random_seed: int | None = None
    encoder_name: str | None = None


class UANWMMpcCemPolicy(BasePolicy):
    """UA-NWM HEP scorer with the validated Gaussian CEM planner.

    The model and CEM candidates use the training-time body-FRD convention.
    Only the selected first action crosses the server boundary, where it is
    converted to a scaled body-FLU local pose increment.
    """

    def __init__(self, config: UANWMRealNavConfig | None = None) -> None:
        self.config = config or UANWMRealNavConfig()
        _validate_config(self.config)
        self._torch = _import_torch()
        self._device = self._resolve_device(self.config.device)
        self._model_cfg = _load_yaml_config(self.config.model_config)
        _apply_model_config_overrides(self._model_cfg, self.config)
        model_type = _load_world_model_type(self.config.uanwm_root)
        self._model = model_type(self._model_cfg).to(self._device)
        checkpoint_state = self._torch.load(
            self.config.checkpoint,
            map_location=self._device,
            weights_only=False,
        )
        _load_model_state(self._model, checkpoint_state)
        if getattr(self._model.fuser, "hep", None) is None:
            raise RuntimeError("UA-NWM model config did not construct the HEP scoring head")
        self._model.eval()
        self._normalizer = _load_action_normalizer(
            self._torch,
            self.config.action_stats,
            checkpoint_state,
            self._device,
        )
        self._context_len = int(self._model_cfg["training"]["context_len"])
        self._image_size = int(self._model_cfg["model"]["image_size"])
        self._spacing = float(self._model_cfg["data"].get("metric_waypoint_spacing", 1.0))
        self._normalize_position_delta = bool(
            self._model_cfg["data"].get("normalize_position_delta", True)
        )
        self._amp_dtype = _amp_dtype(self._torch, self._model_cfg["training"].get("amp", "bf16"))
        self._history: deque[Any] = deque(maxlen=self._context_len)
        self._step = 0

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": "ua_nwm_hep_mpc_cem",
            "model_version": "ua_nwm",
            "model_config": str(self.config.model_config),
            "checkpoint": str(self.config.checkpoint),
            "score_metric": "hep",
            "planner": {
                "name": "mpc_cem",
                "sampler": "per_step_gaussian",
                "horizon": int(self.config.horizon),
                "candidates": int(self.config.cem_candidates),
                "iterations": int(self.config.cem_iterations),
                "elite_ratio": float(self.config.elite_ratio),
                "score_batch_size": int(self.config.score_batch_size),
                "goal_hint_mode": self.config.goal_hint_mode,
                "goal_hint_key": self.config.goal_hint_key,
            },
            "image_keys": {"current": self.config.image_key, "goal": self.config.goal_image_key},
            "input_color_space": self.config.input_color_space,
            "model_coordinate_frame": self.config.model_frame,
            "output_coordinate_frame": "body_flu",
            "waypoint_conversion": {
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
        self._step = 0

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        validate_observation(obs)
        images = obs["images"]
        if self.config.image_key not in images:
            raise KeyError(f"observation images missing current image key {self.config.image_key!r}")
        if self.config.goal_image_key not in images:
            raise KeyError(f"observation images missing goal image key {self.config.goal_image_key!r}")

        current = _as_rgb(images[self.config.image_key], self.config.input_color_space)
        goal = _as_rgb(images[self.config.goal_image_key], self.config.input_color_space)
        self._history.append(self._image_tensor(current))
        history = _pad_history(list(self._history), self._context_len)
        goal_tensor = self._image_tensor(goal)
        goal_hint_model = goal_hint_from_observation(obs, self.config)

        context_states, goal_raw = self._encode_context_goal(history, goal_tensor)
        plan = self._plan(context_states, goal_raw, goal_hint_model)
        selected_model_action = np.asarray(plan["executed_action"], dtype=np.float32).reshape(1, 4)
        selected_output_action = model_actions_to_flu_wire_actions(
            selected_model_action,
            model_frame=self.config.model_frame,
            translation_scale=self.config.translation_scale,
            yaw_scale=self.config.yaw_scale,
        )
        self._step += 1
        return {
            "actions": selected_output_action,
            "goal_reached": False,
            "debug": {
                **plan,
                "goal_hint_mode": self.config.goal_hint_mode,
                "goal_hint_model_frd_m_rad": None if goal_hint_model is None else goal_hint_model.tolist(),
                "selected_model_action_m_rad": selected_model_action[0].astype(float).tolist(),
                "selected_output_action_flu_m_deg": selected_output_action[0].astype(float).tolist(),
                "translation_scale": float(self.config.translation_scale),
                "yaw_scale": float(self.config.yaw_scale),
            },
        }

    def _image_tensor(self, image_rgb: np.ndarray) -> Any:
        resampling = getattr(Image, "Resampling", Image)
        image = Image.fromarray(center_crop_square(image_rgb)).resize(
            (self._image_size, self._image_size),
            resample=resampling.BICUBIC,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = self._torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return (tensor - mean) / std

    def _amp_context(self):
        if self._device.type == "cuda" and self._amp_dtype is not None:
            return self._torch.autocast(device_type="cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

    def _encode_context_goal(self, history_tensors: list[Any], goal_tensor: Any) -> tuple[Any, Any]:
        context_images = self._torch.stack(history_tensors, dim=0).unsqueeze(0).to(self._device)
        goal_images = goal_tensor.unsqueeze(0).unsqueeze(0).to(self._device)
        with self._torch.no_grad(), self._amp_context():
            context_raw = self._model.encode_images(context_images)
            goal_raw = self._model.encode_images(goal_images)[:, 0]
            context_states = self._model.encode_state(context_raw)
        return context_states, goal_raw

    def _physical_to_model_actions(self, physical: Any) -> Any:
        actions = physical.clone()
        if self._normalize_position_delta:
            actions[..., :3] = actions[..., :3] / self._spacing
        prefix = self._torch.zeros(
            physical.shape[0],
            self._context_len - 1,
            physical.shape[-1],
            dtype=physical.dtype,
            device=physical.device,
        )
        actions = self._torch.cat([prefix, actions], dim=1)
        return (actions - self._normalizer[0]) / self._normalizer[1]

    def _score(self, context_states: Any, goal_raw: Any, actions_physical: Any) -> Any:
        if actions_physical.shape[0] > self.config.score_batch_size:
            return self._torch.cat(
                [
                    self._score(
                        context_states,
                        goal_raw,
                        actions_physical[start : start + self.config.score_batch_size],
                    )
                    for start in range(0, actions_physical.shape[0], self.config.score_batch_size)
                ],
                dim=0,
            )
        actions = self._physical_to_model_actions(actions_physical)
        repeated_context = context_states.expand(actions.shape[0], -1, -1, -1).contiguous()
        with self._torch.no_grad(), self._amp_context():
            outputs = self._model.rollout(repeated_context, actions, steps=self.config.horizon)
            final_pred = outputs["pred_features"][:, -1]
            hep_ctx = (
                repeated_context.flatten(1, 2)
                if getattr(self._model.fuser.hep, "use_context", False)
                else None
            )
            return self._model.fuser.hep_cost(
                outputs["pred_latents"][:, -1],
                goal_raw.float().expand_as(final_pred),
                normalize=False,
                context_summary=hep_ctx,
            ).float().mean(dim=1)

    def _initial_distribution(self, goal_hint_model: np.ndarray | None) -> tuple[Any, Any]:
        mean = self._torch.tensor(self.config.action_mean, dtype=self._torch.float32, device=self._device)
        mean = mean.unsqueeze(0).repeat(self.config.horizon, 1)
        if goal_hint_model is not None:
            directed = self._torch.as_tensor(goal_hint_model, dtype=self._torch.float32, device=self._device)
            directed = directed / float(self.config.horizon)
            mean = 0.5 * mean + 0.5 * directed.unsqueeze(0).repeat(self.config.horizon, 1)
        low = self._torch.tensor(self.config.action_low, dtype=self._torch.float32, device=self._device)
        high = self._torch.tensor(self.config.action_high, dtype=self._torch.float32, device=self._device)
        mean = self._torch.max(self._torch.min(mean, high), low)
        std = self._torch.tensor(self.config.action_std, dtype=self._torch.float32, device=self._device)
        return mean, std.unsqueeze(0).repeat(self.config.horizon, 1)

    def _plan(self, context_states: Any, goal_raw: Any, goal_hint_model: np.ndarray | None) -> dict[str, Any]:
        mean, std = self._initial_distribution(goal_hint_model)
        low = self._torch.tensor(self.config.action_low, dtype=self._torch.float32, device=self._device)
        high = self._torch.tensor(self.config.action_high, dtype=self._torch.float32, device=self._device)
        num_elites = max(1, int(round(self.config.cem_candidates * self.config.elite_ratio)))
        best_sequence = None
        best_energy = None
        iteration_stats = []

        generator = None
        if self.config.random_seed is not None:
            generator = self._torch.Generator(device=self._device)
            generator.manual_seed(int(self.config.random_seed) + self._step)

        with self._torch.no_grad():
            for _ in range(self.config.cem_iterations):
                noise = self._torch.randn(
                    self.config.cem_candidates,
                    self.config.horizon,
                    len(ACTION_FIELDS),
                    device=self._device,
                    generator=generator,
                )
                candidates = mean.unsqueeze(0) + noise * std.unsqueeze(0)
                candidates = self._torch.max(self._torch.min(candidates, high), low)
                energy = self._score(context_states, goal_raw, candidates)
                elite_energy, elite_idx = self._torch.topk(energy, k=num_elites, largest=False)
                elites = candidates[elite_idx]
                current_best = elite_idx[0]
                if best_energy is None or energy[current_best] < best_energy:
                    best_energy = energy[current_best].detach()
                    best_sequence = candidates[current_best].detach().clone()
                mean = elites.mean(dim=0)
                std = elites.std(dim=0, unbiased=False).clamp_min(1e-3)
                iteration_stats.append(
                    {
                        "min_energy": float(energy.min().item()),
                        "mean_energy": float(energy.mean().item()),
                        "max_energy": float(energy.max().item()),
                        "elite_mean_energy": float(elite_energy.mean().item()),
                    }
                )

        assert best_sequence is not None and best_energy is not None
        return {
            "planner": "mpc_cem",
            "cem_sampler": "per_step_gaussian",
            "executed_action": best_sequence[0].cpu().float().tolist(),
            "best_action_sequence": best_sequence.cpu().float().tolist(),
            "best_energy": float(best_energy.item()),
            "cem_iterations": iteration_stats,
        }

    def _resolve_device(self, requested: str) -> Any:
        if requested == "cuda" and not self._torch.cuda.is_available():
            logger.warning("CUDA is unavailable; UA-NWM is falling back to CPU")
            return self._torch.device("cpu")
        return self._torch.device(requested)


def center_crop_square(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    height, width = image.shape[:2]
    size = min(height, width)
    y0 = max(0, (height - size) // 2)
    x0 = max(0, (width - size) // 2)
    return np.ascontiguousarray(image[y0 : y0 + size, x0 : x0 + size])


def goal_hint_from_observation(
    observation: Mapping[str, Any],
    config: UANWMRealNavConfig,
) -> np.ndarray | None:
    if config.goal_hint_mode == "none":
        return None
    extra = observation.get("extra")
    if not isinstance(extra, Mapping) or config.goal_hint_key not in extra:
        raise ValueError(
            f"goal_hint_mode='extra_flu' requires observation extra[{config.goal_hint_key!r}] "
            "as [dx_m, dy_m, dz_m, dyaw_deg]"
        )
    goal_flu = np.asarray(extra[config.goal_hint_key], dtype=np.float32)
    if goal_flu.shape != (4,) or not np.all(np.isfinite(goal_flu)):
        raise ValueError(
            f"observation extra[{config.goal_hint_key!r}] must contain four finite FLU values"
        )
    goal_flu_rad = goal_flu.copy()
    goal_flu_rad[3] = np.radians(goal_flu_rad[3])
    return convert_local_waypoints_frame(
        goal_flu_rad.reshape(1, 4),
        source_frame="body_flu",
        target_frame=config.model_frame,
    )[0]


def model_actions_to_flu_wire_actions(
    model_actions: np.ndarray,
    *,
    model_frame: str,
    translation_scale: float,
    yaw_scale: float,
) -> np.ndarray:
    flu_m_rad = convert_local_waypoints_frame(
        np.asarray(model_actions, dtype=np.float32),
        source_frame=model_frame,
        target_frame="body_flu",
    )
    flu_m_deg = local_waypoints_to_wire_actions(flu_m_rad)
    return scale_waypoint_actions(
        flu_m_deg,
        translation_scale=translation_scale,
        yaw_scale=yaw_scale,
    ).astype(np.float32)


def create_policy_from_env() -> UANWMMpcCemPolicy:
    return UANWMMpcCemPolicy(config_from_env())


def create_goal_aware_policy_from_env() -> BasePolicy:
    from uav_openpi.goal_aware_policy import GoalAwarePolicy, goal_termination_config_from_env

    return GoalAwarePolicy(create_policy_from_env(), goal_termination_config_from_env())


def config_from_env() -> UANWMRealNavConfig:
    root = Path(os.environ.get("UANWM_ROOT", str(DEFAULT_UANWM_ROOT)))
    default_config = root / "configs" / "ua_nwm.yaml"
    default_checkpoint = root / "pretrained" / "ua-nwm" / "ua-nwm_best.pt"
    default_action_stats = root / "pretrained" / "ua-nwm" / "action_stats.json"
    return UANWMRealNavConfig(
        uanwm_root=root,
        model_config=Path(os.environ.get("UANWM_CONFIG", str(default_config))),
        checkpoint=Path(os.environ.get("UANWM_CHECKPOINT", str(default_checkpoint))),
        action_stats=Path(os.environ.get("UANWM_ACTION_STATS", str(default_action_stats))),
        device=os.environ.get("UANWM_DEVICE", "cuda"),
        image_key=os.environ.get("UANWM_IMAGE_KEY", "front"),
        goal_image_key=os.environ.get("UANWM_GOAL_IMAGE_KEY", "goal"),
        input_color_space=os.environ.get("UANWM_INPUT_COLOR_SPACE", "rgb"),
        model_frame=os.environ.get("UANWM_MODEL_FRAME", "body_frd"),
        horizon=int(os.environ.get("UANWM_CEM_HORIZON", "8")),
        cem_candidates=int(os.environ.get("UANWM_CEM_CANDIDATES", "32")),
        cem_iterations=int(os.environ.get("UANWM_CEM_ITERATIONS", "3")),
        elite_ratio=float(os.environ.get("UANWM_CEM_ELITE_RATIO", "0.1")),
        score_batch_size=int(os.environ.get("UANWM_CEM_SCORE_BATCH_SIZE", "4")),
        action_mean=_parse_vector(os.environ.get("UANWM_CEM_ACTION_MEAN", "4.7,0.0,-0.8,0.0")),
        action_std=_parse_vector(os.environ.get("UANWM_CEM_ACTION_STD", "1.5,1.0,1.0,0.35")),
        action_low=_parse_vector(os.environ.get("UANWM_CEM_ACTION_LOW", "0.0,-3.0,-3.0,-0.785398")),
        action_high=_parse_vector(os.environ.get("UANWM_CEM_ACTION_HIGH", "7.5,3.0,3.0,0.785398")),
        goal_hint_mode=os.environ.get("UANWM_CEM_GOAL_HINT_MODE", "none"),
        goal_hint_key=os.environ.get("UANWM_CEM_GOAL_HINT_KEY", "goal_delta_body_flu"),
        translation_scale=float(os.environ.get("UANWM_TRANSLATION_SCALE", "1.0")),
        yaw_scale=float(os.environ.get("UANWM_YAW_SCALE", "1.0")),
        random_seed=_optional_int(os.environ.get("UANWM_CEM_RANDOM_SEED")),
        encoder_name=os.environ.get("UANWM_ENCODER_NAME"),
    )


def _validate_config(config: UANWMRealNavConfig) -> None:
    for name, path in {
        "uanwm_root": config.uanwm_root,
        "model_config": config.model_config,
        "checkpoint": config.checkpoint,
        "action_stats": config.action_stats,
    }.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"UA-NWM {name} does not exist: {path}")
    if config.input_color_space not in {"rgb", "bgr"}:
        raise ValueError(f"unsupported input color space {config.input_color_space!r}")
    if config.model_frame not in {"body_frd", "body_flu"}:
        raise ValueError(f"unsupported UA-NWM model frame {config.model_frame!r}")
    if config.goal_hint_mode not in {"none", "extra_flu"}:
        raise ValueError(f"unsupported UA-NWM goal hint mode {config.goal_hint_mode!r}")
    for name, value in {
        "horizon": config.horizon,
        "cem_candidates": config.cem_candidates,
        "cem_iterations": config.cem_iterations,
        "score_batch_size": config.score_batch_size,
    }.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < float(config.elite_ratio) <= 1.0:
        raise ValueError("elite_ratio must be in (0, 1]")
    for name, values in {
        "action_mean": config.action_mean,
        "action_std": config.action_std,
        "action_low": config.action_low,
        "action_high": config.action_high,
    }.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (4,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain four finite values")
    if np.any(np.asarray(config.action_std) < 0):
        raise ValueError("action_std must be non-negative")
    if np.any(np.asarray(config.action_low) > np.asarray(config.action_high)):
        raise ValueError("action_low must not exceed action_high")
    for name, value in {
        "translation_scale": config.translation_scale,
        "yaw_scale": config.yaw_scale,
    }.items():
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if config.encoder_name and _looks_like_filesystem_path(config.encoder_name):
        encoder_path = Path(config.encoder_name).expanduser()
        if not encoder_path.exists():
            raise FileNotFoundError(f"UA-NWM encoder_name does not exist: {encoder_path}")


def _apply_model_config_overrides(model_cfg: dict[str, Any], config: UANWMRealNavConfig) -> None:
    model_section = model_cfg.get("model")
    if not isinstance(model_section, dict):
        raise ValueError("UA-NWM model config must contain a model mapping")
    encoder_name = config.encoder_name or _infer_local_encoder_name(model_section.get("encoder_name"), config)
    if encoder_name:
        old_encoder_name = model_section.get("encoder_name")
        model_section["encoder_name"] = encoder_name
        if old_encoder_name != encoder_name:
            logger.info("Using UA-NWM encoder_name=%s (was %s)", encoder_name, old_encoder_name)


def _infer_local_encoder_name(value: Any, config: UANWMRealNavConfig) -> str | None:
    if not isinstance(value, str) or not _looks_like_filesystem_path(value):
        return None
    configured_path = Path(value).expanduser()
    if configured_path.exists():
        return None
    model_dir_name = configured_path.name
    for candidate in (
        config.uanwm_root / "pretrained" / model_dir_name,
        config.uanwm_root.parent / "UA-NWM-draft" / "pretrained" / model_dir_name,
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _looks_like_filesystem_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~"))


def _load_world_model_type(root: Path) -> Any:
    root = Path(root).resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("uanwm.models")
    module_path = Path(module.__file__).resolve()
    if root not in module_path.parents:
        raise ImportError(f"Python module 'uanwm.models' resolved to {module_path}, not {root}")
    return module.UncertaintyAwareWorldModel


def _load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base_name = config.pop("_base_", None)
    if not base_name:
        return config
    return _deep_update(_load_yaml_config(path.parent / base_name), config)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_model_state(model: Any, checkpoint_state: Mapping[str, Any]) -> None:
    if "model" not in checkpoint_state:
        raise KeyError("UA-NWM checkpoint is missing the model state")
    load_result = model.load_state_dict(checkpoint_state["model"], strict=False)
    missing = [name for name in load_result.missing_keys if not name.startswith("encoder.")]
    unexpected = [name for name in load_result.unexpected_keys if not name.startswith("encoder.")]
    if missing or unexpected:
        raise RuntimeError(
            f"UA-NWM checkpoint does not match the model config: missing={missing} unexpected={unexpected}"
        )


def _load_action_normalizer(
    torch: Any,
    path: Path,
    checkpoint_state: Mapping[str, Any],
    device: Any,
) -> tuple[Any, Any]:
    if Path(path).is_file():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    elif checkpoint_state.get("action_stats"):
        payload = checkpoint_state["action_stats"]
    else:
        raise FileNotFoundError(f"UA-NWM action stats are unavailable: {path}")
    mean_values = np.asarray(payload["mean"], dtype=np.float32)
    std_values = np.asarray(payload["std"], dtype=np.float32)
    if mean_values.shape != (4,) or std_values.shape != (4,):
        raise ValueError("UA-NWM action stats mean/std must each contain four values")
    if not np.all(np.isfinite(mean_values)) or not np.all(np.isfinite(std_values)) or np.any(std_values <= 0):
        raise ValueError("UA-NWM action stats must be finite and have positive std")
    mean = torch.tensor(mean_values, dtype=torch.float32, device=device)
    std = torch.tensor(std_values, dtype=torch.float32, device=device)
    return mean, std


def _amp_dtype(torch: Any, value: str) -> Any | None:
    normalized = str(value).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return None


def _parse_vector(value: str) -> tuple[float, float, float, float]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(parsed) != 4:
        raise ValueError(f"expected four comma-separated values, got {value!r}")
    return parsed  # type: ignore[return-value]


def _optional_int(value: str | None) -> int | None:
    return None if value is None or not value.strip() else int(value)


def _pad_history(history: list[Any], length: int) -> list[Any]:
    if not history:
        raise ValueError("image history must not be empty")
    return [history[0]] * max(0, length - len(history)) + history[-length:]


def _import_torch() -> Any:
    import torch

    return torch


def _as_rgb(image: np.ndarray, color_space: str) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("UA-NWM images must be uint8 HWC arrays with 3 or 4 channels")
    array = array[..., :3]
    if color_space == "bgr":
        array = array[..., ::-1]
    return np.ascontiguousarray(array)
