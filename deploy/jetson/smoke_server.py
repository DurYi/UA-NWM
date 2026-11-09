"""Three local requests: real GPU inference, min-step gate, identical-goal stop.

Starts/stops only its own loopback server. Never imports a drone/ROS client.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
from PIL import Image

from uav_openpi.uav_schema import make_observation
from uav_openpi.websocket_client_policy import WebsocketClientPolicy

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", type=Path, required=True, help="Benchmark result folder with synthetic_current/goal.png")
args = parser.parse_args()
current = np.asarray(Image.open(args.inputs / "synthetic_current.png").convert("RGB"))
goal = np.asarray(Image.open(args.inputs / "synthetic_goal.png").convert("RGB"))
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
env = {**os.environ, "HOST": "127.0.0.1", "PORT": str(port),
       "GOAL_REACHED_METHOD": "lpips", "GOAL_REACHED_THRESHOLD": "0.5",
       "GOAL_REACHED_MIN_STEPS": "2", "GOAL_REACHED_CONSECUTIVE_HITS": "1"}
log_path = args.inputs / "server_smoke.log"
with log_path.open("w") as log:
    server = subprocess.Popen(["bash", str(root / "deploy/jetson/run_server.sh"),
                               str((args.inputs / "synthetic_goal.png").resolve()), "--verbose-wire"],
                              env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        with WebsocketClientPolicy(host="127.0.0.1", port=port,
                                   api_key=env.get("UAV_POLICY_API_KEY"),
                                   connect_timeout_s=120, inference_timeout_s=120) as client:
            client.reset("jetson-static-smoke")
            obs = make_observation(images={"front": current}, state=np.zeros(12, dtype=np.float32))
            first = client.infer(obs)
            assert first["goal_reached"] is False and np.isfinite(first["actions"]).all()
            assert np.asarray(first["actions"]).shape == (1, 4)
            # Goal checks use zero-based steps. min_steps=2 gates steps 0 and 1.
            gated = client.infer(make_observation(images={"front": goal}, state=np.zeros(12, dtype=np.float32)))
            assert gated["goal_reached"] is False
            reached = client.infer(make_observation(images={"front": goal}, state=np.zeros(12, dtype=np.float32)))
            assert reached["goal_reached"] is True and "actions" not in reached
            result = {"first_request": first, "min_steps_gated_request": gated, "identical_goal_request": reached,
                      "metadata": client.metadata, "endpoint": f"127.0.0.1:{port}",
                      "notes": "Cold first request is a functional check, not a steady-state benchmark; no flight client."}
            def convert(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                raise TypeError(type(value).__name__)
            (args.inputs / "server_smoke.json").write_text(json.dumps(result, indent=2, default=convert) + "\n")
            print("PASS: real GPU inference, zero-based min-step gate and identical-goal termination; no ROS / flight commands.")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
