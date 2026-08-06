from __future__ import annotations

import torch
from torch import nn


class AdaLNModulation(nn.Module):
    def __init__(self, action_embed_dim: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_embed_dim, 6 * d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, action_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        chunks = self.net(action_emb).chunk(6, dim=-1)
        return tuple(chunk.unsqueeze(1) for chunk in chunks)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift
