from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_image_transform


def angle_difference(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    delta = theta2 - theta1
    return delta - 2 * np.pi * np.floor((delta + np.pi) / (2 * np.pi))


def wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    return delta - 2 * np.pi * np.floor((delta + np.pi) / (2 * np.pi))


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def to_local_coords(positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float) -> np.ndarray:
    return (positions - curr_pos).dot(yaw_rotmat(float(curr_yaw)))


@dataclass(frozen=True)
class EpisodeIndex:
    episode_dir: Path
    episode_id: str
    num_frames: int


class AirVLNOnlineTrajectoryDataset(Dataset):
    """AirVLN adapter for online DINO feature extraction.

    The dataset reads image frames from ``traj_*/*.jpg`` and computes
    actions from ``traj_data.pkl`` as a_t = [dx, dy, dz, dyaw].
    """

    def __init__(
        self,
        raw_data: str | Path,
        split_file: str | Path | None,
        context_len: int,
        rollout_steps: int,
        image_size: int,
        action_dim: int = 4,
        metric_waypoint_spacing: float = 1.0,
        normalize_position_delta: bool = True,
        action_stride: int = 1,
        action_frame: str = "local",
        max_episodes: int | None = None,
        cache_episode_metadata: bool = True,
        external_depth_dir: str | Path | None = None,
        pad_short_context: bool = False,
    ) -> None:
        self.raw_data = Path(raw_data)
        # Optional external depth/geometry cache (e.g. VGGT-estimated depth for
        # datasets that ship without sim depth). When set, get_geometry_for_index
        # reads depth/pose/intrinsics from <external_depth_dir>/<episode_id>.npz
        # (keys: depth (F,H,W), pose (F,4,4) cam->world, intr (F,3,3), names).
        self.external_depth_dir = Path(external_depth_dir) if external_depth_dir else None
        self.context_len = int(context_len)
        self.rollout_steps = int(rollout_steps)
        self.action_dim = int(action_dim)
        self.metric_waypoint_spacing = float(metric_waypoint_spacing)
        self.normalize_position_delta = bool(normalize_position_delta)
        self.action_stride = int(action_stride)
        self.action_frame = str(action_frame)
        # When True, also emit windows whose context is shorter than ``context_len``
        # (down to a single real frame at episode start). The missing front context
        # positions are filled by REPEATING the earliest available frame — both the
        # image and its trajectory coordinate point — so padded inter-frame actions
        # are zero motion. This teaches the model to tolerate deficient context at
        # the very start of a real navigation episode (deploy-time only 1 frame).
        self.pad_short_context = bool(pad_short_context)
        if self.action_stride < 1:
            raise ValueError("action_stride must be >= 1.")
        if self.action_frame not in {"local", "global"}:
            raise ValueError("action_frame must be 'local' or 'global'.")
        self.cache_episode_metadata = bool(cache_episode_metadata)
        self.transform = build_image_transform(image_size)
        self._episode_cache: dict[str, dict[str, Any]] = {}

        self.episodes = self._discover_episodes(split_file, max_episodes)
        self.index: list[tuple[int, int]] = []
        # With padding, a window only needs >=1 real context frame (start_idx 0),
        # so the missing front positions get replicated from frame 0.
        min_start = 0 if self.pad_short_context else (self.context_len - 1) * self.action_stride
        stop_offset = self.rollout_steps * self.action_stride
        for ep_idx, ep in enumerate(self.episodes):
            for start_idx in range(min_start, ep.num_frames - stop_offset):
                self.index.append((ep_idx, start_idx))
        if not self.index:
            raise ValueError("No valid samples found for the requested context_len and rollout_steps.")

    def _discover_episodes(self, split_file: str | Path | None, max_episodes: int | None) -> list[EpisodeIndex]:
        if split_file:
            names = [line.strip() for line in Path(split_file).read_text().splitlines() if line.strip()]
            episode_dirs = [self.raw_data / name for name in names]
        else:
            episode_dirs = sorted(self.raw_data.glob("traj_*"))
        if max_episodes is not None:
            episode_dirs = episode_dirs[: int(max_episodes)]

        episodes: list[EpisodeIndex] = []
        for episode_dir in episode_dirs:
            if not episode_dir.is_dir():
                continue
            frames = sorted(episode_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            if self.pad_short_context:
                # only need 1 context frame + the future horizon
                required_frames = 1 + self.rollout_steps * self.action_stride
            else:
                required_frames = 1 + (self.context_len + self.rollout_steps - 1) * self.action_stride
            if len(frames) >= required_frames:
                episodes.append(EpisodeIndex(episode_dir, episode_dir.name, len(frames)))
        return episodes

    def __len__(self) -> int:
        return len(self.index)

    def _ctx_ids(self, context_start: int, start_idx: int) -> list[int]:
        """Context frame indices; with padding, negative (pre-episode) positions are
        clamped to frame 0 so the earliest frame is replicated."""
        ids = list(range(context_start, start_idx + 1, self.action_stride))
        if self.pad_short_context:
            ids = [max(0, i) for i in ids]
        return ids

    def _load_episode_meta(self, ep: EpisodeIndex) -> dict[str, Any]:
        cached = self._episode_cache.get(ep.episode_id)
        if cached is not None:
            return cached

        with (ep.episode_dir / "traj_data.pkl").open("rb") as f:
            data = pickle.load(f)
        points = np.asarray(data["point"], dtype=np.float32)
        yaw = np.asarray(data["yaw"], dtype=np.float32).reshape(-1)
        image_names = [str(x) for x in np.asarray(data.get("images", np.arange(len(points)))).tolist()]

        deltas = points[1:] - points[:-1]
        dyaw = angle_difference(yaw[:-1], yaw[1:]).reshape(-1, 1).astype(np.float32)
        if self.normalize_position_delta:
            deltas = deltas / self.metric_waypoint_spacing
        actions = np.concatenate([deltas, dyaw], axis=-1).astype(np.float32)
        if self.action_dim != 4:
            raise ValueError("AirVLN online adapter currently supports action_dim=4.")

        meta = {
            "image_names": image_names,
            "actions": actions,
            "points": points,
            "yaw": yaw,
            "poses": np.asarray(data.get("pose", [])),
        }
        if self.cache_episode_metadata:
            self._episode_cache[ep.episode_id] = meta
        return meta

    def _load_image(self, ep: EpisodeIndex, name: str) -> torch.Tensor:
        path = ep.episode_dir / f"{name}.jpg"
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def get_actions_for_index(self, idx: int) -> torch.Tensor:
        ep_idx, start_idx = self.index[idx]
        ep = self.episodes[ep_idx]
        meta = self._load_episode_meta(ep)

        stride = self.action_stride
        context_start = start_idx - (self.context_len - 1) * stride
        context_frame_ids = self._ctx_ids(context_start, start_idx)
        target_frame_ids = range(start_idx + stride, start_idx + (self.rollout_steps + 1) * stride, stride)
        interval_starts = range(context_start, start_idx + self.rollout_steps * stride, stride)

        if self.action_frame == "local":
            frame_ids = list(context_frame_ids) + list(target_frame_ids)
            local_pos = to_local_coords(
                meta["points"][frame_ids],
                meta["points"][start_idx],
                float(meta["yaw"][start_idx]),
            ).astype(np.float32)
            local_yaw = angle_difference(meta["yaw"][start_idx], meta["yaw"][frame_ids]).reshape(-1).astype(np.float32)
            xyz_actions = local_pos[1:] - local_pos[:-1]
            if self.normalize_position_delta:
                xyz_actions = xyz_actions / self.metric_waypoint_spacing
            dyaw_actions = angle_difference(local_yaw[:-1], local_yaw[1:]).reshape(-1, 1).astype(np.float32)
            actions_np = np.concatenate([xyz_actions, dyaw_actions], axis=-1).astype(np.float32)
        else:
            actions_np = []
            for action_start in interval_starts:
                if self.pad_short_context and action_start < 0:
                    # padded (pre-episode) interval == repeated frame 0 -> zero motion
                    actions_np.append(np.zeros(self.action_dim, dtype=np.float32))
                    continue
                action = meta["actions"][action_start : action_start + stride].sum(axis=0).astype(np.float32)
                action[3] = wrap_angle_delta(np.asarray(action[3], dtype=np.float32))
                actions_np.append(action)
            actions_np = np.stack(actions_np, axis=0)
        return torch.as_tensor(actions_np, dtype=torch.float32)

    def _frame_ids(self, start_idx: int) -> tuple[list[int], list[int]]:
        """Context / target absolute frame indices for a sample (matches __getitem__)."""
        stride = self.action_stride
        context_start = start_idx - (self.context_len - 1) * stride
        context_frame_ids = self._ctx_ids(context_start, start_idx)
        target_frame_ids = list(range(start_idx + stride, start_idx + (self.rollout_steps + 1) * stride, stride))
        return context_frame_ids, target_frame_ids

    def max_future_yaw_delta_deg(self, idx: int) -> float:
        """Max |yaw change| (degrees) from the last context frame to any future frame.
        Uses only the cached yaw array (no depth); for turning-oversampling weights."""
        ep_idx, start_idx = self.index[idx]
        ep = self.episodes[ep_idx]
        yaw = self._load_episode_meta(ep)["yaw"]
        _, target_frame_ids = self._frame_ids(start_idx)
        y0 = float(yaw[start_idx])
        deltas = [abs(float(angle_difference(y0, yaw[t]))) for t in target_frame_ids]
        return float(np.degrees(max(deltas))) if deltas else 0.0

    def get_geometry_for_index(self, idx: int) -> dict[str, Any]:
        """Load depth + pose + intrinsics for one sample, for GT-visibility reprojection.

        Returns numpy arrays:
            K: (3, 3)
            depth_context: (C, H, W), pose_context: (C, 4, 4)
            depth_future: (H_steps, H, W), pose_future: (H_steps, 4, 4)
            context_frame_ids, target_frame_ids, episode_id, episode_dir
        """
        import pickle

        ep_idx, start_idx = self.index[idx]
        ep = self.episodes[ep_idx]
        cache = getattr(self, "_geometry_cache", None)
        if cache is None or cache.get("episode_id") != ep.episode_id:
            with (ep.episode_dir / "traj_data.pkl").open("rb") as f:
                data = pickle.load(f)
            if self.external_depth_dir is not None:
                # Geometry from external cache (e.g. VGGT). Pose/intrinsics come
                # from the cache (its own self-consistent, metric-scaled frame),
                # NOT from the GT pkl. K = per-traj mean intrinsics (frames share
                # the same camera, so per-frame intr are ~identical).
                npz = np.load(self.external_depth_dir / f"{ep.episode_id}.npz")
                cache = {
                    "episode_id": ep.episode_id,
                    "K": np.asarray(npz["intr"], dtype=np.float64).mean(axis=0),
                    "depth": np.asarray(npz["depth"], dtype=np.float32),
                    "pose": np.asarray(npz["pose"], dtype=np.float64),
                    "yaw": np.asarray(data["yaw"], dtype=np.float64).reshape(-1),
                }
            else:
                cache = {
                    "episode_id": ep.episode_id,
                    "K": np.asarray(data["K"], dtype=np.float64),
                    "depth": np.asarray(data["depth"], dtype=np.float32),
                    "pose": np.asarray(data["pose"], dtype=np.float64),
                    "yaw": np.asarray(data["yaw"], dtype=np.float64).reshape(-1),
                }
            self._geometry_cache = cache  # keep only the most recent episode

        context_frame_ids, target_frame_ids = self._frame_ids(start_idx)
        depth, pose = cache["depth"], cache["pose"]
        image_names = self._load_episode_meta(ep)["image_names"]
        return {
            "K": cache["K"],
            "depth_context": depth[context_frame_ids],
            "pose_context": pose[context_frame_ids],
            "depth_future": depth[target_frame_ids],
            "pose_future": pose[target_frame_ids],
            "context_frame_ids": context_frame_ids,
            "target_frame_ids": target_frame_ids,
            "context_image_names": [image_names[i] for i in context_frame_ids],
            "target_image_names": [image_names[i] for i in target_frame_ids],
            "yaw_context": cache["yaw"][context_frame_ids],
            "yaw_future": cache["yaw"][target_frame_ids],
            "episode_id": ep.episode_id,
            "episode_dir": str(ep.episode_dir),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx, start_idx = self.index[idx]
        ep = self.episodes[ep_idx]
        meta = self._load_episode_meta(ep)
        image_names = meta["image_names"]

        stride = self.action_stride
        context_start = start_idx - (self.context_len - 1) * stride
        context_frame_ids = self._ctx_ids(context_start, start_idx)
        target_frame_ids = range(start_idx + stride, start_idx + (self.rollout_steps + 1) * stride, stride)

        context_images = torch.stack([self._load_image(ep, image_names[i]) for i in context_frame_ids], dim=0)
        target_images = torch.stack([self._load_image(ep, image_names[i]) for i in target_frame_ids], dim=0)
        actions = self.get_actions_for_index(idx)

        return {
            "context_images": context_images,
            "target_images": target_images,
            "actions": actions,
            "episode_id": ep.episode_id,
            "start_idx": start_idx,
        }
