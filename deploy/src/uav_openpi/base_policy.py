"""Policy interface derived from openpi_client.base_policy.

Modified for this project to expose optional metadata while preserving the
OpenPI ``infer(observation) -> result`` programming model.
"""

from __future__ import annotations

import abc
from typing import Any


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Infer an action dictionary from an observation dictionary."""

    def reset(self, episode_id: str | None = None) -> None:
        """Reset episode-local policy state, if any."""

    @property
    def metadata(self) -> dict[str, Any]:
        """Describe the model and its wire-level observation/action contract."""
        return {}
