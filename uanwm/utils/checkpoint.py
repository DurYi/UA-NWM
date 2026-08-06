from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    step: int,
    cfg: dict[str, Any],
    action_stats: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "config": cfg,
        "action_stats": action_stats,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    target_model = model.module if hasattr(model, "module") else model
    result = target_model.load_state_dict(state["model"], strict=strict)
    if not strict and result is not None:
        missing = [m for m in result.missing_keys if not m.startswith("encoder.")]
        unexpected = [u for u in result.unexpected_keys if not u.startswith("encoder.")]
        if missing or unexpected:
            print(f"[load_checkpoint] non-encoder missing={missing} unexpected={unexpected}")
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state
