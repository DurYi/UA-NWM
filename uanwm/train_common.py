"""Shared helpers for the UA-NWM staged training pipeline."""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from uanwm.datasets import AirVLNOnlineTrajectoryDataset


class ActionNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std.clamp_min(1e-6)

    @classmethod
    def identity(cls, action_dim: int) -> "ActionNormalizer":
        return cls(torch.zeros(action_dim), torch.ones(action_dim))

    @classmethod
    def from_json(cls, path: str | Path) -> "ActionNormalizer":
        data = json.loads(Path(path).read_text())
        return cls(torch.tensor(data["mean"], dtype=torch.float32), torch.tensor(data["std"], dtype=torch.float32))

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.detach().cpu().tolist(), "std": self.std.detach().cpu().tolist()}

    def to(self, device: torch.device) -> "ActionNormalizer":
        return ActionNormalizer(self.mean.to(device), self.std.to(device))

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.mean.to(actions.device)) / self.std.to(actions.device)


def aggregate_action_windows(actions: torch.Tensor, stride: int) -> torch.Tensor:
    actions = actions.float()
    stride = int(stride)
    if stride <= 1:
        return actions
    if actions.shape[0] < stride:
        return actions[:0]
    windows = []
    for start in range(0, actions.shape[0] - stride + 1):
        action = actions[start : start + stride].sum(dim=0)
        if action.numel() >= 4:
            action[3] = torch.remainder(action[3] + torch.pi, 2 * torch.pi) - torch.pi
        windows.append(action)
    return torch.stack(windows, dim=0)


def build_dataset(cfg: dict[str, Any], split: str):
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    if train_cfg.get("mode", "online") != "online":
        raise ValueError("UA-NWM release training expects training.mode: online.")
    split_key = "train_split" if split == "train" else "val_split"
    max_key = "max_train_episodes" if split == "train" else "max_val_episodes"
    return AirVLNOnlineTrajectoryDataset(
        raw_data=cfg["paths"]["raw_data"],
        split_file=cfg["paths"].get(split_key),
        context_len=train_cfg["context_len"],
        rollout_steps=train_cfg["rollout_steps"],
        image_size=model_cfg["image_size"],
        action_dim=model_cfg["action_dim"],
        metric_waypoint_spacing=cfg["data"].get("metric_waypoint_spacing", 1.0),
        normalize_position_delta=cfg["data"].get("normalize_position_delta", True),
        action_stride=cfg["data"].get("action_stride", 1),
        action_frame=cfg["data"].get("action_frame", "local"),
        max_episodes=cfg["data"].get(max_key),
        cache_episode_metadata=cfg["data"].get("cache_episode_metadata", True),
        pad_short_context=cfg["data"].get("pad_short_context", False),
    )


def estimate_action_stats(dataset, action_dim: int, max_samples: int = 4096) -> ActionNormalizer:
    action_stride = int(getattr(dataset, "action_stride", 1))
    if getattr(dataset, "action_frame", "global") == "local" and hasattr(dataset, "get_actions_for_index"):
        actions = []
        limit = min(len(dataset), max_samples)
        for i in range(limit):
            actions.append(dataset.get_actions_for_index(i))
        stacked = torch.cat(actions, dim=0).float()
        return ActionNormalizer(stacked.mean(dim=0), stacked.std(dim=0))
    if hasattr(dataset, "episodes") and hasattr(dataset, "_load_episode_meta"):
        chunks = []
        for ep in dataset.episodes:
            actions = torch.as_tensor(dataset._load_episode_meta(ep)["actions"], dtype=torch.float32)
            chunks.append(aggregate_action_windows(actions, action_stride))
        stacked = torch.cat(chunks, dim=0)
        return ActionNormalizer(stacked.mean(dim=0), stacked.std(dim=0))
    actions = []
    limit = min(len(dataset), max_samples)
    for i in range(limit):
        actions.append(dataset[i]["actions"])
    stacked = torch.cat(actions, dim=0).float()
    return ActionNormalizer(stacked.mean(dim=0), stacked.std(dim=0))


def build_loaders(cfg: dict[str, Any], rank: int, world_size: int):
    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    pin_memory = bool(cfg["training"].get("pin_memory", True))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_ds, train_loader, val_loader, train_sampler


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def prepare_features(model, batch: dict[str, Any], mode: str, device: torch.device, amp_dtype: torch.dtype | None):
    core_model = unwrap_model(model)
    context_images = batch["context_images"].to(device, non_blocking=True)
    target_images = batch["target_images"].to(device, non_blocking=True)
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype is not None
        else contextlib.nullcontext()
    )
    with ctx:
        context_raw = core_model.encode_images(context_images)
        target_raw = core_model.encode_images(target_images)
    return context_raw, target_raw
