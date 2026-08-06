from __future__ import annotations

import torch
from torch import nn


class TokenCompressor(nn.Module):
    def __init__(self, d_model: int, compressed_tokens: int, heads: int, dropout: float = 0.0,
                 input_dim: int | None = None) -> None:
        super().__init__()
        self.compressed_tokens = int(compressed_tokens)
        # input_dim != d_model: ingest raw features (e.g. DINO dino_dim) directly as keys/values.
        # Queries live in d_model; MHA's kdim/vdim project keys/values.
        input_dim = int(input_dim) if input_dim is not None else d_model
        self.queries = nn.Parameter(torch.randn(self.compressed_tokens, d_model) * 0.02)
        # Only pass kdim/vdim when they DIFFER from d_model: passing them (even if equal) flips
        # MHA to separate q/k/v projections and renames params (in_proj_weight -> {q,k,v}_proj_weight),
        # which would change checkpoint key names.
        if input_dim == d_model:
            self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        else:
            self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True,
                                              kdim=input_dim, vdim=input_dim)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_x = nn.LayerNorm(input_dim)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress patch tokens with learned cross-attention queries.

        Args:
            x: [B, T, N, input_dim]
        Returns:
            compressed: [B, T, K, d_model]
        """
        b, t, _, _ = x.shape
        flat = x.reshape(b * t, x.shape[2], x.shape[3])
        queries = self.queries.unsqueeze(0).expand(b * t, -1, -1)
        xn = self.norm_x(flat)
        compressed, _ = self.attn(self.norm_q(queries), xn, xn, need_weights=False)
        compressed = compressed + self.out(compressed)
        return compressed.reshape(b, t, self.compressed_tokens, self.queries.shape[-1])
