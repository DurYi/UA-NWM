#!/usr/bin/env python3
"""Visualize saved UA-NWM CEM action sequences for a real-flight episode."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from uav_openpi.nomad_policy import convert_local_waypoints_frame


@dataclass(frozen=True)
class StepRecord:
    index: int
    request_id: int
    record_dir: Path
    image_path: Path | None
    state: np.ndarray
    model_frd: np.ndarray
    scaled_flu: np.ndarray
    wire_action: np.ndarray
    best_energy: float
    lpips_score: float | None
    translation_scale: float
    yaw_scale: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-dir",
        type=Path,
        help="Episode directory containing request_*/inference.json; defaults to the latest recorded episode.",
    )
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=repo_root / "records" / "real_flight",
        help="Root used to find the latest episode when --episode-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to <episode-dir>/action_visualizations.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_dir = args.episode_dir.expanduser().resolve() if args.episode_dir else latest_episode(args.records_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else episode_dir / "action_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = load_episode(episode_dir)
    if not steps:
        raise RuntimeError(f"no saved inference action sequences found under {episode_dir}")

    for index, step in enumerate(steps):
        next_step = steps[index + 1] if index + 1 < len(steps) else None
        output_path = output_dir / f"step_{step.index:03d}_request_{step.request_id:08d}.png"
        plot_step(step, next_step, output_path, dpi=args.dpi)
        print(f"wrote {output_path}")

    overview_path = output_dir / "episode_overview.png"
    plot_episode(episode_dir.name, steps, overview_path, dpi=args.dpi)
    print(f"wrote {overview_path}")


def latest_episode(records_dir: Path) -> Path:
    records_dir = records_dir.expanduser().resolve()
    candidates = [
        path
        for path in records_dir.iterdir()
        if path.is_dir() and any(path.glob("request_*/inference.json"))
    ]
    if not candidates:
        raise RuntimeError(f"no recorded episodes with inference.json found under {records_dir}")
    return max(candidates, key=lambda path: max(item.stat().st_mtime_ns for item in path.glob("request_*/inference.json")))


def load_episode(episode_dir: Path) -> list[StepRecord]:
    records: list[StepRecord] = []
    for index, record_dir in enumerate(sorted(episode_dir.glob("request_*"))):
        inference_path = record_dir / "inference.json"
        observation_path = record_dir / "observation.json"
        state_path = record_dir / "state.npy"
        if not inference_path.is_file() or not observation_path.is_file() or not state_path.is_file():
            continue
        inference = json.loads(inference_path.read_text(encoding="utf-8"))
        debug = inference.get("debug") or {}
        sequence_value = debug.get("best_action_sequence")
        if sequence_value is None:
            continue
        model_frd = np.asarray(sequence_value, dtype=np.float64)
        if model_frd.ndim != 2 or model_frd.shape[1] != 4:
            raise ValueError(f"invalid best_action_sequence shape {model_frd.shape} in {inference_path}")
        translation_scale = float(debug.get("translation_scale", 1.0))
        yaw_scale = float(debug.get("yaw_scale", 1.0))
        scaled_flu = convert_local_waypoints_frame(
            model_frd.astype(np.float32),
            source_frame="body_frd",
            target_frame="body_flu",
        ).astype(np.float64)
        scaled_flu[:, :3] *= translation_scale
        scaled_flu[:, 3] *= yaw_scale
        wire_action = np.asarray(inference.get("actions"), dtype=np.float64).reshape(-1, 4)[0]
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        image_path = _first_image_path(record_dir, observation)
        termination = debug.get("goal_termination") or {}
        records.append(
            StepRecord(
                index=index,
                request_id=int(inference["request_id"]),
                record_dir=record_dir,
                image_path=image_path,
                state=np.load(state_path, allow_pickle=False).astype(np.float64),
                model_frd=model_frd,
                scaled_flu=scaled_flu,
                wire_action=wire_action,
                best_energy=float(debug.get("best_energy", math.nan)),
                lpips_score=_optional_float(termination.get("score")),
                translation_scale=translation_scale,
                yaw_scale=yaw_scale,
            )
        )
    return records


def integrate_local_actions(actions_flu_m_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = [np.zeros(3, dtype=np.float64)]
    headings = [0.0]
    for dx, dy, dz, dyaw in np.asarray(actions_flu_m_rad, dtype=np.float64):
        yaw = headings[-1]
        c, s = math.cos(yaw), math.sin(yaw)
        delta = np.asarray([c * dx - s * dy, s * dx + c * dy, dz], dtype=np.float64)
        positions.append(positions[-1] + delta)
        headings.append(_wrap_angle(yaw + dyaw))
    return np.stack(positions), np.asarray(headings)


def plot_step(step: StepRecord, next_step: StepRecord | None, output_path: Path, *, dpi: int) -> None:
    positions, headings = integrate_local_actions(step.scaled_flu)
    horizon = np.arange(1, len(step.scaled_flu) + 1)
    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.15])
    image_ax = fig.add_subplot(grid[:, 0])
    path_ax = fig.add_subplot(grid[0, 1])
    component_ax = fig.add_subplot(grid[0, 2])
    yaw_ax = fig.add_subplot(grid[1, 1])
    info_ax = fig.add_subplot(grid[1, 2])

    if step.image_path is not None:
        with Image.open(step.image_path) as image:
            image_ax.imshow(image.convert("RGB"))
    image_ax.set_title("Observation received by server")
    image_ax.axis("off")

    path_ax.plot(positions[:, 0], positions[:, 1], "-o", color="#2563eb", linewidth=2.0, markersize=4)
    path_ax.plot(positions[:2, 0], positions[:2, 1], "-o", color="#dc2626", linewidth=3.2, label="executed first action")
    for idx, point in enumerate(positions):
        path_ax.annotate(str(idx), point[:2], xytext=(4, 4), textcoords="offset points", fontsize=8)
    arrow_scale = max(0.2, np.ptp(positions[:, :2], axis=0).max() * 0.06)
    path_ax.quiver(
        positions[:, 0],
        positions[:, 1],
        np.cos(headings) * arrow_scale,
        np.sin(headings) * arrow_scale,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        color="#475569",
        alpha=0.7,
    )
    if next_step is not None:
        actual = measured_delta_body_flu(step.state, next_step.state)
        path_ax.plot([0.0, actual[0]], [0.0, actual[1]], "--D", color="#0891b2", linewidth=2.0, label="measured next observation")
    path_ax.axhline(0, color="#94a3b8", linewidth=0.8)
    path_ax.axvline(0, color="#94a3b8", linewidth=0.8)
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.set_title("Selected 8-step plan (scaled body-FLU)")
    path_ax.set_xlabel("forward x (m)")
    path_ax.set_ylabel("left y (m)")
    path_ax.grid(True, alpha=0.25)
    path_ax.legend(fontsize=8)

    component_ax.plot(horizon, step.scaled_flu[:, 0], "-o", label="dx forward")
    component_ax.plot(horizon, step.scaled_flu[:, 1], "-o", label="dy left")
    component_ax.plot(horizon, step.scaled_flu[:, 2], "-o", label="dz up")
    component_ax.axhline(0, color="#64748b", linewidth=0.8)
    component_ax.set_xticks(horizon)
    component_ax.set_title("Per-horizon translation actions")
    component_ax.set_xlabel("planning horizon step")
    component_ax.set_ylabel("scaled meters")
    component_ax.grid(True, alpha=0.25)
    component_ax.legend(fontsize=8)

    yaw_delta_deg = np.degrees(step.scaled_flu[:, 3])
    cumulative_deg = np.degrees(headings[1:])
    yaw_ax.bar(horizon - 0.18, yaw_delta_deg, width=0.36, color="#f59e0b", label="delta yaw")
    yaw_ax.bar(horizon + 0.18, cumulative_deg, width=0.36, color="#7c3aed", alpha=0.75, label="cumulative yaw")
    yaw_ax.axhline(0, color="#64748b", linewidth=0.8)
    yaw_ax.set_xticks(horizon)
    yaw_ax.set_title("Planned yaw")
    yaw_ax.set_xlabel("planning horizon step")
    yaw_ax.set_ylabel("degrees")
    yaw_ax.grid(True, axis="y", alpha=0.25)
    yaw_ax.legend(fontsize=8)

    actual_text = "not available (last recorded step)"
    if next_step is not None:
        actual = measured_delta_body_flu(step.state, next_step.state)
        actual_yaw = math.degrees(_wrap_angle(float(next_step.state[5] - step.state[5])))
        actual_text = f"[{actual[0]:+.3f}, {actual[1]:+.3f}, {actual[2]:+.3f}, {actual_yaw:+.2f} deg]"
    info_lines = [
        f"request_id: {step.request_id}",
        f"best energy: {step.best_energy:.6f}",
        f"LPIPS: {_format_optional(step.lpips_score)}",
        f"translation scale: {step.translation_scale:g}",
        f"yaw scale: {step.yaw_scale:g}",
        "",
        "wire first action [x, y, z, yaw deg]",
        np.array2string(step.wire_action, precision=4, separator=", "),
        "",
        "model first action FRD [x, y, z, yaw rad]",
        np.array2string(step.model_frd[0], precision=4, separator=", "),
        "",
        "measured motion to next observation",
        actual_text,
    ]
    info_ax.text(0.0, 1.0, "\n".join(info_lines), va="top", ha="left", family="monospace", fontsize=9)
    info_ax.axis("off")
    fig.suptitle(
        f"{step.record_dir.parent.name} | planning step {step.index} | request {step.request_id}\n"
        "Blue is the hypothetical full selected plan; red is the only action sent this round.",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_episode(episode_id: str, steps: list[StepRecord], output_path: Path, *, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    actual_ax, plans_ax, action_ax, score_ax = axes.flat
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(steps))))

    states = np.stack([step.state for step in steps])
    actual_local = world_positions_to_initial_body(states)
    actual_ax.plot(actual_local[:, 0], actual_local[:, 1], "-o", color="#111827", linewidth=2.2)
    for step, point in zip(steps, actual_local):
        actual_ax.annotate(str(step.index), point[:2], xytext=(4, 4), textcoords="offset points")
    actual_ax.set_title("Measured observation positions")
    actual_ax.set_xlabel("initial-body forward x (m)")
    actual_ax.set_ylabel("initial-body left y (m)")
    actual_ax.set_aspect("equal", adjustable="datalim")
    actual_ax.grid(True, alpha=0.25)

    for color, step in zip(colors, steps):
        positions, _ = integrate_local_actions(step.scaled_flu)
        plans_ax.plot(positions[:, 0], positions[:, 1], "-o", color=color, markersize=3, label=f"step {step.index}")
        plans_ax.plot(positions[:2, 0], positions[:2, 1], color=color, linewidth=4.0)
    plans_ax.set_title("Selected 8-step plans aligned at each origin")
    plans_ax.set_xlabel("forward x (m)")
    plans_ax.set_ylabel("left y (m)")
    plans_ax.set_aspect("equal", adjustable="datalim")
    plans_ax.grid(True, alpha=0.25)
    plans_ax.legend(fontsize=8)

    step_ids = np.arange(len(steps))
    wire = np.stack([step.wire_action for step in steps])
    width = 0.25
    action_ax.bar(step_ids - width, wire[:, 0], width=width, label="dx forward")
    action_ax.bar(step_ids, wire[:, 1], width=width, label="dy left")
    action_ax.bar(step_ids + width, wire[:, 2], width=width, label="dz up")
    action_ax.axhline(0, color="#64748b", linewidth=0.8)
    action_ax.set_xticks(step_ids)
    action_ax.set_title("Executed first actions returned by server")
    action_ax.set_xlabel("planning step")
    action_ax.set_ylabel("meters")
    action_ax.grid(True, axis="y", alpha=0.25)
    action_ax.legend(fontsize=8)

    yaw = wire[:, 3]
    lpips = np.asarray([math.nan if step.lpips_score is None else step.lpips_score for step in steps])
    score_ax.plot(step_ids, yaw, "-o", color="#7c3aed", label="wire delta yaw (deg)")
    score_ax.set_xlabel("planning step")
    score_ax.set_ylabel("wire delta yaw (deg)", color="#7c3aed")
    score_ax.tick_params(axis="y", labelcolor="#7c3aed")
    score_ax.set_xticks(step_ids)
    score_ax.grid(True, alpha=0.25)
    lpips_ax = score_ax.twinx()
    lpips_ax.plot(step_ids, lpips, "-s", color="#dc2626", label="LPIPS")
    lpips_ax.set_ylabel("LPIPS (lower is closer)", color="#dc2626")
    lpips_ax.tick_params(axis="y", labelcolor="#dc2626")
    score_ax.set_title("Yaw command and goal-image distance")

    fig.suptitle(f"Episode {episode_id} | UA-NWM MPC-CEM action overview", fontsize=14)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def measured_delta_body_flu(current: np.ndarray, following: np.ndarray) -> np.ndarray:
    delta = np.asarray(following[:3] - current[:3], dtype=np.float64)
    yaw = float(current[5])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]])


def world_positions_to_initial_body(states: np.ndarray) -> np.ndarray:
    delta = states[:, :3] - states[0, :3]
    yaw = float(states[0, 5])
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.empty_like(delta)
    local[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    local[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    local[:, 2] = delta[:, 2]
    return local


def _first_image_path(record_dir: Path, observation: dict[str, Any]) -> Path | None:
    image_files = observation.get("image_files") or {}
    preferred = image_files.get("front")
    if preferred:
        path = record_dir / preferred
        return path if path.is_file() else None
    for name in image_files.values():
        path = record_dir / str(name)
        if path.is_file():
            return path
    return None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


if __name__ == "__main__":
    main()
