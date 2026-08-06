#!/usr/bin/env python3
"""Minimal synthetic UAV client based on openpi/examples/simple_client."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import time

import numpy as np
from PIL import Image

from uav_openpi.image_tools import resize_long_edge
from uav_openpi.uav_schema import ActionGuard, LocalWaypointAction, make_observation
from uav_openpi.websocket_client_policy import WebsocketClientPolicy


@dataclass
class SimState:
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    yaw_deg: float = 0.0
    step: int = 0


def random_observation(
    front_image: np.ndarray | None = None,
) -> dict:
    front = front_image if front_image is not None else np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    images = {"front": front}
    return make_observation(
        images=images,
        state=np.zeros(12, dtype=np.float32),
        prompt="fly forward slowly",
    )


def closed_loop_observation(
    state: SimState,
    *,
    prompt: str,
    front_image: np.ndarray | None = None,
) -> dict:
    state_vector = np.zeros(12, dtype=np.float32)
    state_vector[:5] = [state.x_m, state.y_m, state.z_m, state.yaw_deg, float(state.step)]
    images = {"front": front_image if front_image is not None else synthetic_front_image(state)}
    return make_observation(
        images=images,
        state=state_vector,
        prompt=prompt,
        extra={"sim_step": state.step},
    )


def synthetic_front_image(state: SimState) -> np.ndarray:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[..., 1] = 30
    center_x = int(np.clip(112 + state.x_m * 40, 0, 223))
    center_y = int(np.clip(112 - state.y_m * 40, 0, 223))
    image[max(0, center_y - 4) : min(224, center_y + 5), :, 0] = 80
    image[:, max(0, center_x - 4) : min(224, center_x + 5), 2] = 160
    return image



def load_rgb_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def apply_waypoint(state: SimState, waypoint: LocalWaypointAction) -> SimState:
    yaw_rad = np.radians(state.yaw_deg)
    world_dx = waypoint.dx_m * np.cos(yaw_rad) - waypoint.dy_m * np.sin(yaw_rad)
    world_dy = waypoint.dx_m * np.sin(yaw_rad) + waypoint.dy_m * np.cos(yaw_rad)
    return SimState(
        x_m=state.x_m + float(world_dx),
        y_m=state.y_m + float(world_dy),
        z_m=state.z_m + waypoint.dz_m,
        yaw_deg=((state.yaw_deg + waypoint.dyaw_deg + 180.0) % 360.0) - 180.0,
        step=state.step + 1,
    )


def format_state(state: SimState) -> str:
    return f"x={state.x_m:.3f}, y={state.y_m:.3f}, z={state.z_m:.3f}, yaw={state.yaw_deg:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a UAV remote policy server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--api-key", default=os.environ.get("UAV_POLICY_API_KEY"))
    parser.add_argument(
        "--closed-loop",
        action="store_true",
        help="Feed each returned action into the next observation.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each request/response leg explicitly.")
    parser.add_argument("--sleep-between-steps", type=float, default=0.0)
    parser.add_argument("--prompt", default="fly to the mock waypoint")
    parser.add_argument(
        "--front-rgb",
        default=None,
        help="Optional fixed front image path for smoke tests/calibration.",
    )
    parser.add_argument(
        "--max-image-edge",
        type=int,
        default=640,
        help="Downsample transmitted images to this maximum width or height; use 0 to disable.",
    )
    args = parser.parse_args()

    if args.max_image_edge < 0:
        raise ValueError("--max-image-edge must be non-negative")

    guard = ActionGuard()
    round_trip_timings: list[float] = []
    sim_state = SimState()
    front_image = load_rgb_image(args.front_rgb) if args.front_rgb else None
    if args.max_image_edge:
        front_image = resize_long_edge(front_image, args.max_image_edge) if front_image is not None else None
    with WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        inference_timeout_s=args.timeout,
        episode_id="simple-client",
        api_key=args.api_key,
    ) as policy:
        print("Server metadata:", policy.get_server_metadata())
        policy.reset("simple-client")
        for step in range(args.steps):
            observation = (
                closed_loop_observation(sim_state, prompt=args.prompt, front_image=front_image)
                if args.closed_loop
                else random_observation(front_image=front_image)
            )
            if args.verbose or args.closed_loop:
                print(f"round={step:02d} C->S obs_state=({format_state(sim_state)}) prompt={observation['prompt']!r}")
            result = policy.infer(observation)
            round_trip_ms = float(result["transport_timing"]["round_trip_ms"])
            round_trip_timings.append(round_trip_ms)
            if result["goal_reached"]:
                print(f"round={step:02d} S->C goal_reached=true; no action executed")
                break
            waypoint = guard.first_safe_waypoint(result)
            actions = np.asarray(result["actions"])
            if args.verbose or args.closed_loop:
                debug = result.get("debug", {})
                print(
                    f"round={step:02d} S->C waypoint={waypoint} action_shape={actions.shape} "
                    f"server_timing={result.get('server_timing', {})} debug={debug} "
                    f"round_trip_ms={round_trip_ms:.2f}"
                )
            else:
                print(f"step={step} waypoint={waypoint} round_trip_ms={round_trip_ms:.2f}")
            if args.closed_loop:
                sim_state = apply_waypoint(sim_state, waypoint)
                print(f"round={step:02d} SIM next_state=({format_state(sim_state)})")
            if args.sleep_between_steps > 0:
                time.sleep(args.sleep_between_steps)
    if round_trip_timings:
        print(f"mean round-trip: {np.mean(round_trip_timings):.2f} ms")
    else:
        print("no policy requests sent")


if __name__ == "__main__":
    main()
