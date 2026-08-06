"""Hardware-neutral onboard loop; vendor SDK adapters implement two protocols."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Protocol

from uav_openpi import protocol
from uav_openpi.base_policy import BasePolicy
from uav_openpi.goal_termination import GoalTerminationResult, ImageGoalTerminator
from uav_openpi.uav_schema import ActionGuard, LocalWaypointAction, validate_observation

logger = logging.getLogger(__name__)


class ObservationSource(Protocol):
    def read(self) -> dict[str, Any]:
        """Return one wire-ready UAV observation."""


class FlightController(Protocol):
    def send_local_waypoint(self, waypoint: LocalWaypointAction) -> None:
        """Send one bounded pose increment in the body-FLU frame."""

    def hover(self) -> None:
        """Enter a locally controlled safe hover."""


@dataclass(frozen=True)
class RunResult:
    episode_id: str
    status: str
    completed_steps: int


class OnboardRuntime:
    """Run remote inference while retaining all flight authority locally."""

    def __init__(
        self,
        observation_source: ObservationSource,
        flight_controller: FlightController,
        policy: BasePolicy,
        action_guard: ActionGuard | None = None,
        max_observation_age_s: float = 0.5,
        goal_terminator: ImageGoalTerminator | None = None,
        goal_result_callback: Callable[[GoalTerminationResult, dict[str, Any]], None] | None = None,
    ) -> None:
        self._observation_source = observation_source
        self._flight_controller = flight_controller
        self._policy = policy
        self._action_guard = action_guard or ActionGuard()
        self._max_observation_age_s = max_observation_age_s
        self._goal_terminator = goal_terminator
        self._goal_result_callback = goal_result_callback

    def run(self, *, episode_id: str, steps: int) -> RunResult:
        try:
            self._reset_policy(episode_id)
            if self._goal_terminator is not None:
                self._goal_terminator.reset()
            for step in range(steps):
                observation = self._observation_source.read()
                validate_observation(observation)
                age_s = (time.time_ns() - observation["timestamp_ns"]) / 1_000_000_000
                if age_s < -0.1 or age_s > self._max_observation_age_s:
                    raise ValueError(f"observation age {age_s:.3f}s is outside the safe window")
                if self._goal_terminator is not None:
                    termination = self._goal_terminator.evaluate(observation, step=step)
                    if self._goal_result_callback is not None:
                        self._goal_result_callback(termination, observation)
                    logger.info(
                        "Goal metric step=%s method=%s score=%.4f threshold=%.4f hits=%s/%s reached=%s",
                        step,
                        termination.method,
                        termination.score,
                        termination.threshold,
                        termination.consecutive_hits,
                        termination.required_consecutive_hits,
                        termination.reached,
                    )
                    if termination.reached:
                        self._flight_controller.hover()
                        return RunResult(episode_id, "goal_reached", step)
                result = self._policy.infer(observation)
                protocol.validate_inference_result(result)
                if result["goal_reached"]:
                    self._flight_controller.hover()
                    return RunResult(episode_id, "goal_reached", step)
                waypoint = self._action_guard.first_safe_waypoint(result)
                self._flight_controller.send_local_waypoint(waypoint)
            self._flight_controller.hover()
            return RunResult(episode_id, "completed", steps)
        except Exception as exc:
            logger.exception("Onboard inference loop failed; commanding hover")
            self._flight_controller.hover()
            return RunResult(episode_id, f"safe_hover:{type(exc).__name__}", locals().get("step", 0))

    def _reset_policy(self, episode_id: str) -> None:
        self._policy.reset(episode_id)
