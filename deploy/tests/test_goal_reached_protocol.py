from __future__ import annotations

import itertools
import logging
import threading

import numpy as np
import pytest

from uav_openpi import msgpack_numpy, protocol
from uav_openpi.goal_aware_policy import GoalAwarePolicy
from uav_openpi.goal_termination import GoalTerminationConfig
from uav_openpi.mock_policy import ConstantWaypointPolicy, WaypointPolicy
from uav_openpi.onboard_runtime import OnboardRuntime
from uav_openpi.server_goal_policy import ServerGoalPolicy
from uav_openpi.uav_schema import LocalWaypointAction, make_observation
from uav_openpi.websocket_client_policy import WebsocketClientPolicy


def observation(*, identical_goal: bool = False, position_x: float = 0.0) -> dict:
    front = np.zeros((24, 32, 3), dtype=np.uint8)
    front[..., 0] = np.arange(32, dtype=np.uint8)[None, :]
    goal = front.copy() if identical_goal else np.flip(front, axis=1).copy()
    state = np.zeros(12, dtype=np.float32)
    state[0] = position_x
    return make_observation(images={"front": front, "goal": goal}, state=state, prompt="test")


def test_inference_result_requires_native_bool() -> None:
    valid_false = {"actions": np.zeros((1, 4), dtype=np.float32), "goal_reached": False}
    valid_true = {"goal_reached": True}
    assert protocol.validate_inference_result(valid_false) is valid_false
    assert protocol.validate_inference_result(valid_true) is valid_true

    for invalid in (None, 0, 1, "false", np.bool_(False)):
        with pytest.raises(protocol.ProtocolError, match="native boolean"):
            protocol.validate_inference_result({"actions": np.zeros((1, 4)), "goal_reached": invalid})


def test_inference_result_requires_action_only_when_not_reached() -> None:
    with pytest.raises(protocol.ProtocolError, match="missing goal_reached"):
        protocol.validate_inference_result({"actions": np.zeros((1, 4))})
    with pytest.raises(protocol.ProtocolError, match="requires actions"):
        protocol.validate_inference_result({"goal_reached": False})
    assert protocol.validate_inference_result({"goal_reached": True})["goal_reached"] is True


def test_msgpack_preserves_native_goal_boolean() -> None:
    result = msgpack_numpy.unpackb(msgpack_numpy.packb({"goal_reached": True}))
    assert type(result["goal_reached"]) is bool


def test_goal_aware_policy_stops_before_second_policy_call_and_resets() -> None:
    class CountingPolicy(ConstantWaypointPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def infer(self, obs: dict) -> dict:
            self.calls += 1
            return super().infer(obs)

    base = CountingPolicy()
    policy = GoalAwarePolicy(
        base,
        GoalTerminationConfig(
            method="rgb_cosine",
            threshold=0.99,
            consecutive_hits=2,
            min_steps=0,
            resize_hw=(16, 16),
        ),
    )
    policy.reset("episode-a")
    first = policy.infer(observation(identical_goal=True))
    second = policy.infer(observation(identical_goal=True))
    sticky = policy.infer(observation(identical_goal=False))

    assert first["goal_reached"] is False
    assert first["actions"].shape == (1, 4)
    assert second["goal_reached"] is True
    assert second["debug"]["goal_termination"]["reached"] is True
    assert sticky == {"goal_reached": True}
    assert base.calls == 1

    policy.reset("episode-b")
    after_reset = policy.infer(observation(identical_goal=True))
    assert after_reset["goal_reached"] is False
    assert base.calls == 2


def test_server_fixed_goal_reaches_without_client_goal_image() -> None:
    front_only = observation(identical_goal=True)
    goal = front_only["images"].pop("goal")
    policy = ServerGoalPolicy(
        GoalAwarePolicy(
            ConstantWaypointPolicy(),
            GoalTerminationConfig(
                method="rgb_cosine",
                threshold=0.99,
                consecutive_hits=1,
                min_steps=0,
                resize_hw=(16, 16),
            ),
        ),
        goal,
    )

    result = policy.infer(front_only)
    assert result["goal_reached"] is True
    assert result["debug"]["goal_termination"]["reached"] is True
    assert "goal" not in front_only["images"]


def test_mock_policies_emit_native_goal_boolean() -> None:
    moving = ConstantWaypointPolicy().infer(observation())
    reached = WaypointPolicy(target_xyz=(1.0, 0.0, 0.0)).infer(observation(position_x=1.0))
    assert moving["goal_reached"] is False
    assert type(moving["goal_reached"]) is bool
    assert reached["goal_reached"] is True
    assert "actions" not in reached


def _client_with_results(results: list[dict]) -> WebsocketClientPolicy:
    client = WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    client._lock = threading.Lock()
    client._request_ids = itertools.count(1)
    client._episode_id = "test"
    client._inference_timeout_s = 1.0
    client._warned_missing_goal_reached = False
    client._ensure_connected = lambda: None
    responses = iter(results)

    def round_trip(request: dict, *, timeout_s: float) -> dict:
        del timeout_s
        result = next(responses)
        return protocol.make_success_response(request, result=result)

    client._round_trip = round_trip
    return client


def test_client_warns_once_for_missing_goal_reached(caplog: pytest.LogCaptureFixture) -> None:
    legacy = {"actions": np.zeros((1, 4), dtype=np.float32)}
    client = _client_with_results([legacy, legacy])
    with caplog.at_level(logging.WARNING):
        assert client.infer(observation())["goal_reached"] is False
        assert client.infer(observation())["goal_reached"] is False
    messages = [record.message for record in caplog.records if "omitted goal_reached" in record.message]
    assert len(messages) == 1


def test_client_rejects_non_boolean_goal_reached() -> None:
    client = _client_with_results([{"actions": np.zeros((1, 4)), "goal_reached": "false"}])
    with pytest.raises(protocol.ProtocolError, match="native boolean"):
        client.infer(observation())


def test_runtime_hovers_on_server_goal_without_action() -> None:
    class Source:
        def read(self) -> dict:
            return observation()

    class ReachedPolicy(ConstantWaypointPolicy):
        def infer(self, obs: dict) -> dict:
            return {"goal_reached": True, "actions": "must be ignored"}

    class Controller:
        def __init__(self) -> None:
            self.hovered = False
            self.waypoints: list[LocalWaypointAction] = []

        def send_local_waypoint(self, waypoint: LocalWaypointAction) -> None:
            self.waypoints.append(waypoint)

        def hover(self) -> None:
            self.hovered = True

    controller = Controller()
    result = OnboardRuntime(Source(), controller, ReachedPolicy()).run(episode_id="goal", steps=3)
    assert result.status == "goal_reached"
    assert result.completed_steps == 0
    assert controller.hovered
    assert not controller.waypoints


def test_runtime_safe_hovers_on_invalid_goal_type() -> None:
    class Source:
        def read(self) -> dict:
            return observation()

    class InvalidPolicy(ConstantWaypointPolicy):
        def infer(self, obs: dict) -> dict:
            return {"actions": np.zeros((1, 4)), "goal_reached": 0}

    class Controller:
        def __init__(self) -> None:
            self.hovered = False

        def send_local_waypoint(self, waypoint: LocalWaypointAction) -> None:
            raise AssertionError("invalid result must not reach the flight controller")

        def hover(self) -> None:
            self.hovered = True

    controller = Controller()
    result = OnboardRuntime(Source(), controller, InvalidPolicy()).run(episode_id="invalid", steps=1)
    assert result.status == "safe_hover:ProtocolError"
    assert controller.hovered
