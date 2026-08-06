#!/usr/bin/env python3
"""Start the reference UAV policy server, mirroring openpi/scripts/serve_policy.py."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from uav_openpi.base_policy import BasePolicy
from uav_openpi.image_tools import resize_long_edge
from uav_openpi.mock_policy import ConstantWaypointPolicy, WaypointPolicy
from uav_openpi.observation_recorder import ObservationRecorder
from uav_openpi.server_goal_policy import ServerGoalPolicy
from uav_openpi.websocket_policy_server import WebsocketPolicyServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an OpenPI-style UAV policy.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.0)
    parser.add_argument("--dyaw", type=float, default=0.0)
    parser.add_argument("--mock-policy", choices=("constant", "waypoint"), default="constant")
    parser.add_argument("--target-x", type=float, default=1.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.0)
    parser.add_argument("--api-key", default=os.environ.get("UAV_POLICY_API_KEY"))
    parser.add_argument("--goal-rgb", help="Server-local fixed goal image used for every inference request.")
    parser.add_argument(
        "--goal-image-key",
        default=None,
        help="Internal policy image key; defaults to the policy metadata value or 'goal'.",
    )
    parser.add_argument(
        "--goal-max-image-edge",
        type=int,
        default=640,
        help="Downsample the server-local goal to this maximum edge; use 0 to disable.",
    )
    parser.add_argument(
        "--verbose-wire",
        action="store_true",
        help="Send policy debug fields, full metadata, and timing to a development client.",
    )
    parser.add_argument(
        "--record-observations-dir",
        help="Asynchronously save each accepted client observation under this directory.",
    )
    parser.add_argument(
        "--record-image-format",
        choices=("png", "jpeg"),
        default="png",
        help="Image format used by the observation recorder.",
    )
    parser.add_argument(
        "--record-jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used when --record-image-format=jpeg.",
    )
    parser.add_argument(
        "--record-queue-size",
        type=int,
        default=64,
        help="Maximum pending observation writes before new records are dropped.",
    )
    parser.add_argument(
        "--policy-factory",
        help="Optional module:function that constructs the real server-side BasePolicy.",
    )
    args = parser.parse_args()

    if args.policy_factory:
        policy = load_policy_factory(args.policy_factory)
    elif args.mock_policy == "constant":
        policy = ConstantWaypointPolicy(dx_m=args.dx, dy_m=args.dy, dz_m=args.dz, dyaw_deg=args.dyaw)
    else:
        policy = WaypointPolicy(
            target_xyz=(args.target_x, args.target_y, args.target_z),
        )
    if args.goal_max_image_edge < 0:
        raise ValueError("--goal-max-image-edge must be non-negative")
    policy_goal_key = _policy_goal_key(policy)
    if args.policy_factory and policy_goal_key and not args.goal_rgb:
        parser.error(f"--goal-rgb is required by policy image key {policy_goal_key!r}")
    if args.goal_rgb:
        goal_path = Path(args.goal_rgb).expanduser().resolve()
        goal_image = load_rgb(goal_path)
        if args.goal_max_image_edge:
            goal_image = resize_long_edge(goal_image, args.goal_max_image_edge)
        goal_key = args.goal_image_key or policy_goal_key or "goal"
        policy = ServerGoalPolicy(policy, goal_image, goal_image_key=goal_key)
        logging.getLogger(__name__).info(
            "Loaded fixed server goal path=%s key=%s shape=%s",
            goal_path,
            goal_key,
            list(goal_image.shape),
        )
    recorder = None
    if args.record_observations_dir:
        recorder = ObservationRecorder(
            args.record_observations_dir,
            image_format=args.record_image_format,
            jpeg_quality=args.record_jpeg_quality,
            queue_size=args.record_queue_size,
        )
    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        verbose_wire=args.verbose_wire,
        observation_recorder=recorder,
    ).serve_forever()


def load_policy_factory(spec: str) -> BasePolicy:
    try:
        module_name, factory_name = spec.split(":", 1)
    except ValueError as exc:
        raise ValueError("--policy-factory must use module:function syntax") from exc
    factory = getattr(importlib.import_module(module_name), factory_name)
    policy = factory()
    if not isinstance(policy, BasePolicy):
        raise TypeError(f"{spec} returned {type(policy)!r}, expected BasePolicy")
    return policy


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _policy_goal_key(policy: BasePolicy) -> str | None:
    image_keys = policy.metadata.get("image_keys")
    if not isinstance(image_keys, dict):
        return None
    goal_key = image_keys.get("goal")
    return goal_key if isinstance(goal_key, str) and goal_key else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
