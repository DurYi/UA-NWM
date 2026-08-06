"""Server-side final-goal detection wrapper for image-goal policies."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict
from typing import Any

from uav_openpi.base_policy import BasePolicy
from uav_openpi.goal_termination import GoalTerminationConfig, GoalTerminationResult, ImageGoalTerminator

logger = logging.getLogger(__name__)


class GoalAwarePolicy(BasePolicy):
    """Evaluate final-goal completion before asking the wrapped policy for an action."""

    def __init__(self, policy: BasePolicy, config: GoalTerminationConfig) -> None:
        self._policy = policy
        self._terminator = ImageGoalTerminator(config)
        self._goal_reached = False

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._policy.metadata)
        metadata["goal_termination"] = {
            "location": "server",
            "method": self._terminator.config.method,
            "threshold": float(self._terminator.config.threshold),
            "consecutive_hits": int(self._terminator.config.consecutive_hits),
            "min_steps": int(self._terminator.config.min_steps),
            "image_key": self._terminator.config.image_key,
            "goal_image_key": self._terminator.config.goal_image_key,
        }
        metadata["result_schema"] = {"goal_reached": "bool"}
        return metadata

    def reset(self, episode_id: str | None = None) -> None:
        self._policy.reset(episode_id)
        self._terminator.reset()
        self._goal_reached = False

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._goal_reached:
            return {"goal_reached": True}

        termination = self._terminator.evaluate(obs)
        self._log_result(termination)
        termination_debug = asdict(termination)
        if termination.reached:
            self._goal_reached = True
            return {"goal_reached": True, "debug": {"goal_termination": termination_debug}}

        result = self._policy.infer(obs)
        if not isinstance(result, dict):
            raise TypeError("wrapped policy infer() must return a dictionary")
        result = dict(result)
        result["goal_reached"] = False
        debug = result.get("debug")
        if isinstance(debug, dict):
            debug = dict(debug)
        elif debug is None:
            debug = {}
        else:
            debug = {"policy": debug}
        debug["goal_termination"] = termination_debug
        result["debug"] = debug
        return result

    @staticmethod
    def _log_result(result: GoalTerminationResult) -> None:
        logger.info(
            "Server goal check step=%s method=%s score=%.6f threshold=%.6f hits=%s/%s reached=%s",
            result.step,
            result.method,
            result.score,
            result.threshold,
            result.consecutive_hits,
            result.required_consecutive_hits,
            result.reached,
        )


def goal_termination_config_from_env() -> GoalTerminationConfig:
    threshold_value = os.environ.get("GOAL_REACHED_THRESHOLD")
    if threshold_value is None:
        raise ValueError("GOAL_REACHED_THRESHOLD is required for server-side goal detection")
    config = GoalTerminationConfig(
        image_key=os.environ.get("GOAL_REACHED_IMAGE_KEY", "front"),
        goal_image_key=os.environ.get("GOAL_REACHED_GOAL_IMAGE_KEY", "goal"),
        method=os.environ.get("GOAL_REACHED_METHOD", "lpips"),
        threshold=float(threshold_value),
        consecutive_hits=int(os.environ.get("GOAL_REACHED_CONSECUTIVE_HITS", "3")),
        min_steps=int(os.environ.get("GOAL_REACHED_MIN_STEPS", "2")),
        lpips_net=os.environ.get("GOAL_REACHED_LPIPS_NET", "alex"),
        lpips_device=os.environ.get("GOAL_REACHED_LPIPS_DEVICE", "auto"),
    )
    if config.method not in {"zncc", "rgb_cosine", "histogram_intersection", "lpips"}:
        raise ValueError(f"unsupported GOAL_REACHED_METHOD {config.method!r}")
    if not math.isfinite(float(config.threshold)):
        raise ValueError("GOAL_REACHED_THRESHOLD must be finite")
    if config.method == "lpips" and float(config.threshold) < 0:
        raise ValueError("GOAL_REACHED_THRESHOLD must be non-negative for LPIPS")
    if int(config.consecutive_hits) < 1:
        raise ValueError("GOAL_REACHED_CONSECUTIVE_HITS must be positive")
    if int(config.min_steps) < 0:
        raise ValueError("GOAL_REACHED_MIN_STEPS must be non-negative")
    return config
