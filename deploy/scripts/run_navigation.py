#!/usr/bin/env python3
"""Run real UAV navigation using server-side final-goal detection."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import json
import logging
import os
from typing import Any

from uav_openpi.image_tools import resize_long_edge
from uav_openpi.onboard_runtime import OnboardRuntime
from uav_openpi.uav_schema import ActionGuard, ActionLimits
from uav_openpi.websocket_client_policy import WebsocketClientPolicy


class ResizedObservationSource:
    """Bound image payload size before sending a fresh hardware observation."""

    def __init__(self, source: Any, max_image_edge: int | None = 640) -> None:
        self._source = source
        self._max_image_edge = max_image_edge

    def read(self) -> dict[str, Any]:
        observation = dict(self._source.read())
        images = dict(observation.get("images", {}))
        if self._max_image_edge is not None:
            images = {
                name: resize_long_edge(image, self._max_image_edge)
                for name, image in images.items()
            }
        observation["images"] = images
        return observation


def create_component(spec: str, kwargs_json: str, required_method: str) -> Any:
    try:
        module_name, factory_name = spec.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"component factory must use module:function syntax, got {spec!r}") from exc
    kwargs = json.loads(kwargs_json)
    if not isinstance(kwargs, dict):
        raise ValueError("component factory kwargs must decode to a JSON object")
    component = getattr(importlib.import_module(module_name), factory_name)(**kwargs)
    if not callable(getattr(component, required_method, None)):
        raise TypeError(f"{spec} returned an object without required {required_method}() method")
    return component


def manage_component(stack: ExitStack, component: Any) -> Any:
    if hasattr(component, "__enter__") and hasattr(component, "__exit__"):
        return stack.enter_context(component)
    close = getattr(component, "close", None)
    if callable(close):
        stack.callback(close)
    return component


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remote-policy UAV navigation with server goal detection.")
    parser.add_argument("--observation-source-factory", required=True)
    parser.add_argument("--observation-source-kwargs", default="{}")
    parser.add_argument("--flight-controller-factory", required=True)
    parser.add_argument("--flight-controller-kwargs", default="{}")
    parser.add_argument(
        "--max-image-edge",
        type=int,
        default=640,
        help="Downsample transmitted images to this maximum width or height; use 0 to disable.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--episode-id", default="real-navigation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--max-observation-age", type=float, default=0.5)
    parser.add_argument("--max-dx-m", type=float, default=1.0)
    parser.add_argument("--max-dy-m", type=float, default=1.0)
    parser.add_argument("--max-dz-m", type=float, default=0.5)
    parser.add_argument("--max-dyaw-deg", type=float, default=15.0)
    parser.add_argument("--api-key", default=os.environ.get("UAV_POLICY_API_KEY"))
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.max_image_edge < 0:
        raise ValueError("--max-image-edge must be non-negative")
    with ExitStack() as stack:
        source = manage_component(
            stack,
            create_component(args.observation_source_factory, args.observation_source_kwargs, "read"),
        )
        controller = manage_component(
            stack,
            create_component(args.flight_controller_factory, args.flight_controller_kwargs, "hover"),
        )
        if not callable(getattr(controller, "send_local_waypoint", None)):
            raise TypeError("flight controller must also implement send_local_waypoint()")
        resized_source = ResizedObservationSource(source, max_image_edge=args.max_image_edge or None)
        policy = stack.enter_context(
            WebsocketClientPolicy(
                host=args.host,
                port=args.port,
                api_key=args.api_key,
                connect_timeout_s=args.connect_timeout,
                inference_timeout_s=args.timeout,
                episode_id=args.episode_id,
            )
        )
        runtime = OnboardRuntime(
            resized_source,
            controller,
            policy,
            action_guard=ActionGuard(
                ActionLimits(
                    max_dx_m=args.max_dx_m,
                    max_dy_m=args.max_dy_m,
                    max_dz_m=args.max_dz_m,
                    max_dyaw_deg=args.max_dyaw_deg,
                )
            ),
            max_observation_age_s=args.max_observation_age,
        )
        result = runtime.run(episode_id=args.episode_id, steps=args.steps)

    print(
        f"navigation finished: episode_id={result.episode_id} status={result.status} "
        f"completed_steps={result.completed_steps}"
    )
    if result.status.startswith("safe_hover:"):
        raise SystemExit(2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
