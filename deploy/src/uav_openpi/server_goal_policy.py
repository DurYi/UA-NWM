"""Server-side fixed goal injection for image-goal policies."""

from __future__ import annotations

from typing import Any

import numpy as np

from uav_openpi.base_policy import BasePolicy


class ServerGoalPolicy(BasePolicy):
    """Inject one server-local goal image without accepting it from the client."""

    def __init__(self, policy: BasePolicy, goal_image: np.ndarray, goal_image_key: str = "goal") -> None:
        image = np.asarray(goal_image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
            raise ValueError("server goal image must be a uint8 HWC array with 1, 3, or 4 channels")
        if not goal_image_key:
            raise ValueError("goal_image_key must be non-empty")
        self._policy = policy
        self._goal_image = np.ascontiguousarray(image)
        self._goal_image_key = goal_image_key

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._policy.metadata)
        metadata["goal_image_source"] = {
            "location": "server",
            "key": self._goal_image_key,
            "shape": list(self._goal_image.shape),
            "sent_by_client": False,
        }
        return metadata

    def reset(self, episode_id: str | None = None) -> None:
        self._policy.reset(episode_id)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        observation = dict(obs)
        images = dict(observation.get("images", {}))
        if self._goal_image_key in images:
            raise ValueError(
                f"client observation must not contain server-owned image key {self._goal_image_key!r}"
            )
        images[self._goal_image_key] = self._goal_image
        observation["images"] = images
        return self._policy.infer(observation)
