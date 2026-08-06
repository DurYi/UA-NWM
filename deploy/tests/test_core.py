from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from PIL import Image

from uav_openpi import msgpack_numpy, protocol
from uav_openpi.image_tools import resize_long_edge
from uav_openpi.mock_policy import ConstantWaypointPolicy
from uav_openpi.onboard_runtime import OnboardRuntime
from uav_openpi.observation_recorder import ObservationRecorder
from uav_openpi.server_goal_policy import ServerGoalPolicy
from uav_openpi.uav_schema import ActionGuard, ActionLimits, LocalWaypointAction, SchemaError, make_observation
from uav_openpi.websocket_policy_server import WebsocketPolicyServer, _make_wire_result


def observation() -> dict:
    return make_observation(
        images={"front": np.zeros((32, 48, 3), dtype=np.uint8)},
        state=np.zeros(12, dtype=np.float32),
        prompt="test",
    )


def test_msgpack_round_trip_preserves_numpy() -> None:
    expected = observation()
    actual = msgpack_numpy.unpackb(msgpack_numpy.packb(expected))
    assert np.array_equal(actual["images"]["front"], expected["images"]["front"])
    assert actual["images"]["front"].dtype == np.uint8
    assert np.array_equal(actual["state"], expected["state"])


def test_resize_long_edge_preserves_aspect_and_small_images() -> None:
    landscape = np.zeros((300, 900, 3), dtype=np.uint8)
    portrait = np.zeros((900, 300, 3), dtype=np.uint8)
    small = np.zeros((32, 48, 3), dtype=np.uint8)

    assert resize_long_edge(landscape, 600).shape == (200, 600, 3)
    assert resize_long_edge(portrait, 600).shape == (600, 200, 3)
    assert resize_long_edge(small, 600).shape == small.shape


def test_protocol_request_and_response() -> None:
    request = protocol.make_infer_request(observation(), request_id=7, episode_id="ep", deadline_ms=500)
    protocol.validate_request(request)
    response = protocol.make_success_response(request, result={"actions": np.zeros((1, 4))})
    assert protocol.validate_response(response, request_id=7)["status"] == "ok"


def test_production_wire_result_is_small_and_language_neutral() -> None:
    result = {
        "actions": np.array([[0.1, -0.2, 0.3, 4.0]], dtype=np.float32),
        "goal_reached": False,
        "debug": {"best_action_sequence": [[1.0] * 4] * 8},
    }
    wire_result = _make_wire_result(result, include_debug=False)

    assert wire_result.keys() == {"actions", "goal_reached"}
    assert isinstance(wire_result["actions"], list)
    assert isinstance(wire_result["actions"][0], list)
    assert np.allclose(wire_result["actions"], result["actions"])


def test_verbose_wire_result_keeps_policy_debug() -> None:
    result = {
        "actions": np.zeros((1, 4), dtype=np.float32),
        "goal_reached": False,
        "debug": {"best_energy": 0.25},
    }

    assert _make_wire_result(result, include_debug=True)["debug"] == result["debug"]


def test_goal_reached_wire_result_omits_ignored_action() -> None:
    result = {
        "actions": np.ones((1, 4), dtype=np.float32),
        "goal_reached": True,
        "debug": {"score": 0.1},
    }

    assert _make_wire_result(result, include_debug=False) == {"goal_reached": True}


def test_observation_recorder_saves_images_state_and_metadata(tmp_path) -> None:
    recorder = ObservationRecorder(tmp_path, image_format="png", queue_size=2)
    obs = observation()
    obs["state"] = obs["state"].astype(np.float64)
    obs["extra"] = {"goal_delta_body_flu": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}

    assert recorder.record(obs, request_id=7, episode_id="flight/test", received_ns=123456789)
    result = {
        "actions": np.array([[0.4, -0.2, 0.1, 3.0]], dtype=np.float32),
        "goal_reached": False,
        "debug": {"best_energy": np.float32(0.25), "best_action_sequence": [[1.0] * 4] * 8},
    }
    assert recorder.record_inference(
        result,
        request_id=7,
        episode_id="flight/test",
        received_ns=123456789,
        completed_ns=123456999,
        queue_ms=1.25,
        infer_ms=42.5,
    )
    recorder.close()

    episode_dir = tmp_path / "flight_test"
    record_dir = episode_dir / "request_00000007_123456789"
    assert np.array_equal(np.load(record_dir / "state.npy", allow_pickle=False), obs["state"])
    with Image.open(record_dir / "front.png") as image:
        assert image.size == (48, 32)
    metadata = json.loads((record_dir / "observation.json").read_text(encoding="utf-8"))
    assert metadata["episode_id"] == "flight/test"
    assert metadata["request_id"] == 7
    assert metadata["image_files"] == {"front": "front.png"}
    assert metadata["fields"]["prompt"] == "test"
    assert metadata["fields"]["extra"]["goal_delta_body_flu"] == [1.0, 2.0, 3.0, 4.0]
    manifest = (episode_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest) == 1
    assert json.loads(manifest[0])["record_dir"] == record_dir.name
    inference = json.loads((record_dir / "inference.json").read_text(encoding="utf-8"))
    assert np.allclose(inference["actions"], result["actions"])
    assert inference["debug"]["best_energy"] == pytest.approx(0.25)
    assert inference["server_timing"] == {"queue_ms": 1.25, "infer_ms": 42.5}
    inference_manifest = (episode_dir / "inference_index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(inference_manifest) == 1
    assert json.loads(inference_manifest[0])["actions"] == inference["actions"]
    assert recorder.saved_count == 1
    assert recorder.saved_inference_count == 1
    assert recorder.dropped_count == 0


def test_observation_recorder_supports_high_quality_jpeg(tmp_path) -> None:
    recorder = ObservationRecorder(tmp_path, image_format="jpeg", jpeg_quality=97)

    assert recorder.record(observation(), request_id=1, episode_id="jpeg", received_ns=2)
    recorder.close()

    image_path = tmp_path / "jpeg" / "request_00000001_2" / "front.jpg"
    with Image.open(image_path) as image:
        assert image.format == "JPEG"
        assert image.size == (48, 32)


def test_policy_server_queues_raw_client_observation_before_inference() -> None:
    class RecordingSink:
        def __init__(self) -> None:
            self.calls = []
            self.inference_calls = []

        def record(self, observation, **kwargs) -> bool:
            self.calls.append((observation, kwargs))
            return True

        def record_inference(self, result, **kwargs) -> bool:
            self.inference_calls.append((result, kwargs))
            return True

    sink = RecordingSink()
    server = WebsocketPolicyServer(ConstantWaypointPolicy(), observation_recorder=sink)
    server._infer_lock = asyncio.Lock()
    request = protocol.make_infer_request(observation(), request_id=9, episode_id="ep", deadline_ms=500)

    response = asyncio.run(server._dispatch(request))

    assert response["status"] == "ok"
    assert len(sink.calls) == 1
    saved_observation, saved_fields = sink.calls[0]
    assert saved_observation is request["observation"]
    assert saved_fields["request_id"] == 9
    assert saved_fields["episode_id"] == "ep"
    assert len(sink.inference_calls) == 1
    saved_result, inference_fields = sink.inference_calls[0]
    assert saved_result["goal_reached"] is False
    assert inference_fields["received_ns"] == saved_fields["received_ns"]
    assert inference_fields["request_id"] == 9
    assert inference_fields["infer_ms"] >= 0.0


def test_policy_server_continues_inference_when_recorder_fails() -> None:
    class FailingSink:
        def record(self, observation, **kwargs) -> bool:
            del observation, kwargs
            raise OSError("disk unavailable")

    server = WebsocketPolicyServer(ConstantWaypointPolicy(), observation_recorder=FailingSink())
    server._infer_lock = asyncio.Lock()
    request = protocol.make_infer_request(observation(), request_id=10, episode_id="ep", deadline_ms=500)

    response = asyncio.run(server._dispatch(request))

    assert response["status"] == "ok"
    assert response["result"]["goal_reached"] is False


def test_server_goal_policy_injects_goal_without_mutating_client_observation() -> None:
    class RecordingPolicy(ConstantWaypointPolicy):
        def infer(self, obs: dict) -> dict:
            self.observation = obs
            return super().infer(obs)

    wrapped = RecordingPolicy()
    goal = np.full((20, 30, 3), 127, dtype=np.uint8)
    policy = ServerGoalPolicy(wrapped, goal)
    client_observation = observation()

    policy.infer(client_observation)

    assert "goal" not in client_observation["images"]
    assert np.array_equal(wrapped.observation["images"]["goal"], goal)


def test_server_goal_policy_rejects_client_goal_transfer() -> None:
    policy = ServerGoalPolicy(ConstantWaypointPolicy(), np.zeros((8, 8, 3), dtype=np.uint8))
    client_observation = observation()
    client_observation["images"]["goal"] = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="must not contain"):
        policy.infer(client_observation)


def test_expired_request_is_detected() -> None:
    request = protocol.make_infer_request(observation(), request_id=1, episode_id="ep", deadline_ms=1)
    assert protocol.request_deadline_expired(request, elapsed_ms=2.0)


def test_action_guard_clips_waypoint() -> None:
    guard = ActionGuard(ActionLimits(max_dx_m=0.3, max_dyaw_deg=10.0))
    waypoint = guard.first_safe_waypoint({"actions": np.array([[2.0, 0.0, 0.0, 30.0]], dtype=np.float32)})
    assert waypoint.dx_m == pytest.approx(0.3)
    assert waypoint.dyaw_deg == pytest.approx(10.0)


def test_action_guard_rejects_action_chunks() -> None:
    guard = ActionGuard()
    with pytest.raises(SchemaError, match=r"shape \(1, 4\)"):
        guard.first_safe_waypoint({"actions": np.zeros((2, 4), dtype=np.float32)})


def test_action_guard_rejects_nan() -> None:
    with pytest.raises(SchemaError):
        ActionGuard().first_safe_waypoint({"actions": np.array([[np.nan, 0, 0, 0]])})


def test_mock_policy_contract() -> None:
    result = ConstantWaypointPolicy().infer(observation())
    assert result["actions"].shape == (1, 4)


def test_onboard_runtime_hovers_after_policy_failure() -> None:
    class Source:
        def read(self) -> dict:
            return observation()

    class FailedPolicy(ConstantWaypointPolicy):
        def infer(self, obs: dict) -> dict:
            raise RuntimeError("network failed")

    class Controller:
        def __init__(self) -> None:
            self.waypoints: list[LocalWaypointAction] = []
            self.hovered = False

        def send_local_waypoint(self, waypoint: LocalWaypointAction) -> None:
            self.waypoints.append(waypoint)

        def hover(self) -> None:
            self.hovered = True

    controller = Controller()
    result = OnboardRuntime(Source(), controller, FailedPolicy()).run(episode_id="failure", steps=3)
    assert result.status == "safe_hover:RuntimeError"
    assert result.completed_steps == 0
    assert controller.hovered
    assert not controller.waypoints



def test_nomad_waypoints_to_local_flu_action() -> None:
    from uav_openpi.nomad_policy import (
        convert_local_waypoints_frame,
        local_waypoints_to_wire_actions,
        scale_waypoint_actions,
        waypoints_to_local_waypoints,
    )

    waypoints = np.array(
        [
            [
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    local = waypoints_to_local_waypoints(waypoints, metric_spacing=2.0, pos_dimension=3, learn_angle=True)
    assert local.shape == (1, 2, 4)
    assert np.allclose(local[0, :, :3], [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    actions = local_waypoints_to_wire_actions(local[0])
    assert actions.shape == (2, 4)
    scaled = scale_waypoint_actions(actions, translation_scale=0.4, yaw_scale=1.5)
    assert np.allclose(scaled[:, :3], actions[:, :3] * 0.4)
    assert np.allclose(scaled[:, 3], actions[:, 3] * 1.5)

    model_frd = np.array([[1.0, 2.0, 3.0, 0.4]], dtype=np.float32)
    output_flu = convert_local_waypoints_frame(model_frd, source_frame="body_frd", target_frame="body_flu")
    assert np.allclose(output_flu, [[1.0, -2.0, -3.0, -0.4]])


def test_nomad_action_scale_rejects_invalid_values() -> None:
    from uav_openpi.nomad_policy import scale_waypoint_actions

    actions = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="translation_scale"):
        scale_waypoint_actions(actions, translation_scale=-0.1, yaw_scale=1.0)


def test_image_goal_terminator_reaches_on_identical_images() -> None:
    from uav_openpi.goal_termination import GoalTerminationConfig, ImageGoalTerminator

    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[..., 0] = np.arange(48, dtype=np.uint8)[None, :]
    image[..., 1] = np.arange(32, dtype=np.uint8)[:, None]
    obs = make_observation(
        images={"front": image, "goal": image.copy()},
        state=np.zeros(12, dtype=np.float32),
        prompt="test",
    )
    terminator = ImageGoalTerminator(
        GoalTerminationConfig(threshold=0.99, consecutive_hits=1, min_steps=0, resize_hw=(16, 16))
    )
    result = terminator.evaluate(obs, step=0)
    assert result.reached
    assert result.score == pytest.approx(1.0)


def test_onboard_runtime_stops_when_goal_similarity_reached() -> None:
    from uav_openpi.goal_termination import GoalTerminationConfig, ImageGoalTerminator

    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[..., 2] = 128

    class Source:
        def read(self) -> dict:
            return make_observation(
                images={"front": image, "goal": image.copy()},
                state=np.zeros(12, dtype=np.float32),
                prompt="test",
            )

    class Policy(ConstantWaypointPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def infer(self, obs: dict) -> dict:
            self.calls += 1
            return super().infer(obs)

    class Controller:
        def __init__(self) -> None:
            self.waypoints: list[LocalWaypointAction] = []
            self.hovered = False

        def send_local_waypoint(self, waypoint: LocalWaypointAction) -> None:
            self.waypoints.append(waypoint)

        def hover(self) -> None:
            self.hovered = True

    policy = Policy()
    controller = Controller()
    terminator = ImageGoalTerminator(
        GoalTerminationConfig(method="rgb_cosine", threshold=0.99, consecutive_hits=1, min_steps=0)
    )
    result = OnboardRuntime(Source(), controller, policy, goal_terminator=terminator).run(
        episode_id="goal", steps=3
    )
    assert result.status == "goal_reached"
    assert result.completed_steps == 0
    assert policy.calls == 0
    assert controller.hovered
    assert not controller.waypoints


def test_lpips_threshold_uses_lower_distance() -> None:
    from uav_openpi.goal_termination import _meets_threshold

    assert _meets_threshold(0.1, 0.2, lower_is_better=True)
    assert not _meets_threshold(0.3, 0.2, lower_is_better=True)
    assert _meets_threshold(0.9, 0.8, lower_is_better=False)


def test_goal_result_callback_runs_before_stop() -> None:
    from uav_openpi.goal_termination import GoalTerminationConfig, ImageGoalTerminator

    image = np.full((24, 32, 3), 64, dtype=np.uint8)

    class Source:
        def read(self) -> dict:
            return make_observation(
                images={"front": image, "goal": image.copy()},
                state=np.zeros(12, dtype=np.float32),
            )

    class Controller:
        def __init__(self) -> None:
            self.hovered = False

        def send_local_waypoint(self, waypoint: LocalWaypointAction) -> None:
            raise AssertionError("action must not be sent after reaching the goal")

        def hover(self) -> None:
            self.hovered = True

    records = []
    controller = Controller()
    result = OnboardRuntime(
        Source(),
        controller,
        ConstantWaypointPolicy(),
        goal_terminator=ImageGoalTerminator(
            GoalTerminationConfig(method="rgb_cosine", threshold=0.99, consecutive_hits=1, min_steps=0)
        ),
        goal_result_callback=lambda termination, observation: records.append(
            (termination, observation["timestamp_ns"])
        ),
    ).run(episode_id="callback", steps=1)

    assert result.status == "goal_reached"
    assert controller.hovered
    assert len(records) == 1
    assert records[0][0].reached
