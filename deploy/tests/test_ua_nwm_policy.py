from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uav_openpi.ua_nwm_policy import (
    UANWMRealNavConfig,
    center_crop_square,
    config_from_env,
    goal_hint_from_observation,
    model_actions_to_flu_wire_actions,
)
from uav_openpi.uav_schema import make_observation


def config(**overrides) -> UANWMRealNavConfig:
    values = {
        "uanwm_root": Path("."),
        "model_config": Path("."),
        "checkpoint": Path("."),
        "action_stats": Path("."),
    }
    values.update(overrides)
    return UANWMRealNavConfig(**values)


def observation(extra=None) -> dict:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    return make_observation(
        images={"front": image, "goal": image.copy()},
        state=np.zeros(12, dtype=np.float32),
        extra=extra,
    )


def test_center_crop_square_matches_online_preprocessing() -> None:
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    cropped = center_crop_square(image)
    assert cropped.shape == (4, 4, 3)
    assert np.array_equal(cropped, image[:, 1:5])


def test_goal_hint_converts_flu_degrees_to_frd_radians() -> None:
    policy_config = config(goal_hint_mode="extra_flu")
    hint = goal_hint_from_observation(
        observation(extra={"goal_delta_body_flu": [8.0, 2.0, 1.0, 30.0]}),
        policy_config,
    )
    assert hint is not None
    assert np.allclose(hint, [8.0, -2.0, -1.0, -np.pi / 6])


def test_goal_hint_is_optional_in_image_only_mode() -> None:
    assert goal_hint_from_observation(observation(), config(goal_hint_mode="none")) is None


def test_goal_hint_mode_requires_explicit_flu_hint() -> None:
    with pytest.raises(ValueError, match="goal_delta_body_flu"):
        goal_hint_from_observation(observation(), config(goal_hint_mode="extra_flu"))


def test_ua_nwm_action_crosses_frd_to_flu_wire_boundary() -> None:
    actions = model_actions_to_flu_wire_actions(
        np.array([[2.0, 1.0, -0.5, np.pi / 6]], dtype=np.float32),
        model_frame="body_frd",
        translation_scale=0.25,
        yaw_scale=0.5,
    )
    assert actions.dtype == np.float32
    assert actions.shape == (1, 4)
    assert np.allclose(actions, [[0.5, -0.25, 0.125, -15.0]], atol=1e-5)


def test_root_environment_variable_relocates_default_model_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UANWM_ROOT", "/models/UA-NWM")
    policy_config = config_from_env()
    assert policy_config.uanwm_root == Path("/models/UA-NWM")
    assert policy_config.model_config == Path("/models/UA-NWM/configs/ua_nwm.yaml")
    assert policy_config.checkpoint == Path("/models/UA-NWM/pretrained/ua-nwm/ua-nwm_best.pt")
    assert policy_config.action_stats == Path("/models/UA-NWM/pretrained/ua-nwm/action_stats.json")
