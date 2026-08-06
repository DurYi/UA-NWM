from __future__ import annotations

import torch


def get_autocast_dtype(name: str) -> torch.dtype | None:
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return None
    raise ValueError(f"Unsupported AMP mode: {name}")


def tensor_norm_mean(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().norm(dim=-1).mean()
