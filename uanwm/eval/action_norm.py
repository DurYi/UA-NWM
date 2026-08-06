from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from uanwm.train_common import ActionNormalizer


def load_action_normalizer(cfg: dict[str, Any], checkpoint_state: dict[str, Any] | None, device: torch.device) -> ActionNormalizer:
    action_dim = int(cfg["model"]["action_dim"])
    stats_path = cfg["paths"].get("action_stats")
    if stats_path and Path(stats_path).exists():
        return ActionNormalizer.from_json(stats_path).to(device)
    output_stats = Path(cfg.get("output_dir", "outputs")) / cfg["exp_name"] / "action_stats.json"
    if output_stats.exists():
        return ActionNormalizer.from_json(output_stats).to(device)
    if checkpoint_state and checkpoint_state.get("action_stats"):
        stats = checkpoint_state["action_stats"]
        return ActionNormalizer(
            torch.tensor(stats["mean"], dtype=torch.float32),
            torch.tensor(stats["std"], dtype=torch.float32),
        ).to(device)
    return ActionNormalizer.identity(action_dim).to(device)
