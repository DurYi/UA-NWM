"""Image-goal termination checks for real UAV runs without a known goal pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from PIL import Image


TerminationMethod = Literal["zncc", "rgb_cosine", "histogram_intersection", "lpips"]


@dataclass(frozen=True)
class GoalTerminationConfig:
    image_key: str = "front"
    goal_image_key: str = "goal"
    method: TerminationMethod = "zncc"
    threshold: float = 0.92
    consecutive_hits: int = 2
    min_steps: int = 1
    resize_hw: tuple[int, int] = (64, 64)
    center_crop_fraction: float = 1.0
    lpips_net: str = "alex"
    lpips_device: str = "auto"
    lpips_resize_hw: tuple[int, int] = (224, 224)
    lpips_center_crop_aspect_ratio: float = 4.0 / 3.0


@dataclass(frozen=True)
class GoalTerminationResult:
    reached: bool
    score: float
    threshold: float
    consecutive_hits: int
    required_consecutive_hits: int
    method: str
    step: int
    reason: str
    lower_is_better: bool


class ImageGoalScorer:
    """Score a current RGB frame against a fixed goal frame."""

    def __init__(self, config: GoalTerminationConfig | None = None, goal_image: np.ndarray | None = None) -> None:
        self.config = config or GoalTerminationConfig()
        self._goal_descriptor: Any | None = None
        self._reset_goal_descriptor: Any | None = None
        self._lpips_model: Any | None = None
        self._lpips_device: Any | None = None
        if goal_image is not None:
            self.set_goal(goal_image)

    @property
    def lower_is_better(self) -> bool:
        return self.config.method == "lpips"

    @property
    def device(self) -> str:
        if self.config.method != "lpips" or self._lpips_device is None:
            return "cpu"
        return str(self._lpips_device)

    @property
    def has_goal(self) -> bool:
        return self._goal_descriptor is not None

    def reset(self, goal_image: np.ndarray | None = None) -> None:
        self._goal_descriptor = _copy_descriptor(self._reset_goal_descriptor)
        if goal_image is not None:
            self.set_goal(goal_image)

    def set_goal(self, goal_image: np.ndarray, *, remember: bool = True) -> None:
        descriptor = self._descriptor(goal_image)
        self._goal_descriptor = descriptor
        if remember:
            self._reset_goal_descriptor = _copy_descriptor(descriptor)

    def score(self, current_image: np.ndarray, goal_image: np.ndarray | None = None) -> float:
        if goal_image is not None:
            self.set_goal(goal_image, remember=False)
        if self._goal_descriptor is None:
            raise ValueError("goal image is not initialized")
        current_descriptor = self._descriptor(current_image)
        if self.config.method == "lpips":
            self._ensure_lpips_model()
            with self._torch().inference_mode():
                return float(self._lpips_model(current_descriptor, self._goal_descriptor).item())
        return float(_similarity(current_descriptor, self._goal_descriptor, self.config.method))

    def _descriptor(self, image: np.ndarray) -> Any:
        rgb = _as_rgb(image)
        if self.config.method == "lpips":
            rgb = _center_crop_aspect(rgb, float(self.config.lpips_center_crop_aspect_ratio))
            height, width = self.config.lpips_resize_hw
            pil = Image.fromarray(rgb).resize((int(width), int(height)), Image.Resampling.BILINEAR)
            arr = np.asarray(pil, dtype=np.float32) / 127.5 - 1.0
            tensor = self._torch().from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0)
            self._ensure_lpips_model()
            return tensor.to(self._lpips_device)

        rgb = _center_crop(rgb, float(self.config.center_crop_fraction))
        height, width = self.config.resize_hw
        pil = Image.fromarray(rgb).resize((int(width), int(height)), Image.Resampling.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        if self.config.method == "histogram_intersection":
            hist_parts = []
            for channel in range(3):
                hist, _ = np.histogram(arr[..., channel], bins=32, range=(0.0, 1.0), density=False)
                hist_parts.append(hist.astype(np.float32))
            desc = np.concatenate(hist_parts)
            total = float(desc.sum())
            return desc / max(total, 1e-6)
        return arr.reshape(-1).astype(np.float32)

    def _ensure_lpips_model(self) -> None:
        if self._lpips_model is not None:
            return
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise ImportError("LPIPS scoring requires the optional dependency: pip install -e '.[lpips]'") from exc
        requested = self.config.lpips_device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"LPIPS device {requested!r} requested but CUDA is unavailable")
        self._lpips_device = torch.device(requested)
        self._lpips_model = lpips.LPIPS(net=self.config.lpips_net).to(self._lpips_device).eval()
        for parameter in self._lpips_model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("LPIPS scoring requires torch; install the optional 'lpips' extra") from exc
        return torch


class ImageGoalTerminator:
    """Compare a current camera image with a goal image and decide completion."""

    def __init__(self, config: GoalTerminationConfig | None = None, goal_image: np.ndarray | None = None) -> None:
        self.config = config or GoalTerminationConfig()
        self._scorer = ImageGoalScorer(self.config, goal_image=goal_image)
        self._hit_count = 0
        self._step_count = 0

    @property
    def scorer(self) -> ImageGoalScorer:
        return self._scorer

    def reset(self, goal_image: np.ndarray | None = None) -> None:
        self._hit_count = 0
        self._step_count = 0
        self._scorer.reset(goal_image)

    def set_goal(self, goal_image: np.ndarray) -> None:
        self._scorer.set_goal(goal_image)

    def evaluate(self, observation: dict[str, Any], *, step: int | None = None) -> GoalTerminationResult:
        images = observation.get("images")
        if not isinstance(images, dict):
            raise ValueError("observation must contain an images dictionary")
        if self.config.image_key not in images:
            raise KeyError(f"observation images missing current image key {self.config.image_key!r}")
        if self.config.goal_image_key in images:
            score = self._scorer.score(images[self.config.image_key], images[self.config.goal_image_key])
        elif not self._scorer.has_goal:
            raise KeyError(f"observation images missing goal image key {self.config.goal_image_key!r}")
        else:
            score = self._scorer.score(images[self.config.image_key])
        current_step = self._step_count if step is None else int(step)
        self._step_count = current_step + 1

        if current_step < int(self.config.min_steps):
            self._hit_count = 0
            return self._result(False, score, current_step, "before_min_steps")

        if _meets_threshold(score, float(self.config.threshold), self._scorer.lower_is_better):
            self._hit_count += 1
        else:
            self._hit_count = 0
        reached = self._hit_count >= max(1, int(self.config.consecutive_hits))
        return self._result(reached, score, current_step, "threshold_met" if reached else "below_required_hits")

    def _result(self, reached: bool, score: float, step: int, reason: str) -> GoalTerminationResult:
        return GoalTerminationResult(
            reached=bool(reached),
            score=float(score),
            threshold=float(self.config.threshold),
            consecutive_hits=int(self._hit_count),
            required_consecutive_hits=max(1, int(self.config.consecutive_hits)),
            method=str(self.config.method),
            step=int(step),
            reason=reason,
            lower_is_better=self._scorer.lower_is_better,
        )


def _as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError("termination images must be uint8 HWC arrays with 1, 3, or 4 channels")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr[..., :3].copy()


def _center_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.05, 1.0))
    if fraction >= 0.999:
        return image
    height, width = image.shape[:2]
    crop_h = max(1, int(round(height * fraction)))
    crop_w = max(1, int(round(width * fraction)))
    y0 = max(0, (height - crop_h) // 2)
    x0 = max(0, (width - crop_w) // 2)
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _center_crop_aspect(image: np.ndarray, aspect_ratio: float) -> np.ndarray:
    if aspect_ratio <= 0:
        raise ValueError("center crop aspect ratio must be positive")
    height, width = image.shape[:2]
    current_ratio = width / height
    if current_ratio > aspect_ratio:
        crop_width = max(1, int(round(height * aspect_ratio)))
        x0 = (width - crop_width) // 2
        return image[:, x0 : x0 + crop_width]
    crop_height = max(1, int(round(width / aspect_ratio)))
    y0 = (height - crop_height) // 2
    return image[y0 : y0 + crop_height, :]


def _copy_descriptor(descriptor: Any | None) -> Any | None:
    if descriptor is None:
        return None
    if hasattr(descriptor, "clone"):
        return descriptor.clone()
    return descriptor.copy()


def _meets_threshold(score: float, threshold: float, lower_is_better: bool) -> bool:
    return score <= threshold if lower_is_better else score >= threshold


def _similarity(current: np.ndarray, goal: np.ndarray | None, method: str) -> float:
    if goal is None:
        raise ValueError("goal descriptor is not initialized")
    if method == "histogram_intersection":
        return float(np.minimum(current, goal).sum())
    cur = current.astype(np.float32).reshape(-1)
    ref = goal.astype(np.float32).reshape(-1)
    if method == "zncc":
        cur = cur - float(cur.mean())
        ref = ref - float(ref.mean())
    elif method != "rgb_cosine":
        raise ValueError(f"unsupported termination similarity method {method!r}")
    denom = float(np.linalg.norm(cur) * np.linalg.norm(ref))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(cur, ref) / denom, -1.0, 1.0))
