"""Shared navigation trajectory utilities for AirGoal evaluation and visualization."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from uanwm.datasets.trajectory_dataset import angle_difference, to_local_coords


def list_traj_dirs(root: str | Path) -> list[Path]:
    """List AirGoal trajectory directories from a split root."""
    root = Path(root)
    return sorted(root.glob("traj_*"))


def _poses6(record: dict[str, Any]) -> np.ndarray:
    """Reconstruct [T,6] poses as [x,y,z,roll,pitch,yaw] from a trajectory record."""
    point = np.asarray(record["point"], dtype=np.float64)
    n = int(point.shape[0])
    roll = np.asarray(record.get("roll", np.zeros(n)), dtype=np.float64).reshape(-1)
    pitch = np.asarray(record.get("pitch", np.zeros(n)), dtype=np.float64).reshape(-1)
    yaw = np.asarray(record["yaw"], dtype=np.float64).reshape(-1)
    return np.column_stack([point[:, 0], point[:, 1], point[:, 2], roll, pitch, yaw])


def load_traj_pkl(traj_dir: str | Path) -> tuple[np.ndarray, list[str], list[np.ndarray] | None]:
    """Load one AirGoal trajectory folder.

    Returns:
        gt_poses: [F,6] ground-truth poses, [x,y,z,roll,pitch,yaw].
        frame_stems: image basenames; frame i is ``f"{frame_stems[i]}.jpg"``.
        cand_pose_list: test candidate poses, or ``None`` when not present.
    """
    traj_dir = Path(traj_dir)
    with (traj_dir / "traj_data.pkl").open("rb") as f:
        data = pickle.load(f)
    gt_poses = _poses6(data)
    frame_stems = [str(x) for x in np.asarray(data["images"]).tolist()]
    candidates = data.get("candidate_trajectories")
    cand_pose_list = [_poses6(candidate) for candidate in candidates] if candidates is not None else None
    return gt_poses, frame_stems, cand_pose_list


def traj_ate_rpe(sel_poses, gt_poses, lo: int, hi: int) -> tuple[float, float, float]:
    """ATE/RPE between a selected trajectory and the GT future path over [lo, hi]."""
    selected = np.asarray(sel_poses, dtype=np.float64)[lo : hi + 1]
    gt = np.asarray(gt_poses, dtype=np.float64)[lo : hi + 1]
    pos_s, pos_g = selected[:, :3], gt[:, :3]
    yaw_s, yaw_g = selected[:, 5], gt[:, 5]
    ate = float(np.sqrt(np.mean(np.sum((pos_s - pos_g) ** 2, axis=1))))
    d_s = np.diff(pos_s, axis=0)
    d_g = np.diff(pos_g, axis=0)
    rpe_t = float(np.sqrt(np.mean(np.sum((d_s - d_g) ** 2, axis=1)))) if len(d_s) else 0.0
    dyaw_s = np.angle(np.exp(1j * np.diff(yaw_s)))
    dyaw_g = np.angle(np.exp(1j * np.diff(yaw_g)))
    drot = np.angle(np.exp(1j * (dyaw_s - dyaw_g)))
    rpe_r = float(np.degrees(np.sqrt(np.mean(drot**2)))) if len(drot) else 0.0
    return ate, rpe_t, rpe_r


def pose_traj_xyzyaw(poses, lo: int, hi: int) -> list[list[float]]:
    """Return [x,y,z,yaw] over [lo, hi] for raw metric recomputation."""
    pose = np.asarray(poses, dtype=np.float64)[lo : hi + 1]
    return np.stack([pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 5]], axis=1).tolist()


def candidate_local_actions(gt_poses, cand_poses, decision_idx: int, goal_idx: int, spacing: float) -> np.ndarray:
    """Build local-frame actions for one candidate, anchored at the GT decision pose."""
    gt = np.asarray(gt_poses, dtype=np.float64)
    candidate = np.asarray(cand_poses, dtype=np.float64)
    poses = np.concatenate(
        [gt[: decision_idx + 1], candidate[decision_idx + 1 : goal_idx + 1]],
        axis=0,
    )
    pos = poses[:, :3]
    yaw = poses[:, 5]
    ref_pos = gt[decision_idx, :3]
    ref_yaw = gt[decision_idx, 5]
    local_pos = to_local_coords(pos, ref_pos, float(ref_yaw)).astype(np.float32)
    local_yaw = angle_difference(ref_yaw, yaw).reshape(-1).astype(np.float32)
    xyz = (local_pos[1:] - local_pos[:-1]) / float(spacing)
    dyaw = angle_difference(local_yaw[:-1], local_yaw[1:]).reshape(-1, 1).astype(np.float32)
    return np.concatenate([xyz, dyaw], axis=-1).astype(np.float32)
