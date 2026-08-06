from __future__ import annotations

import torch
from torch import nn

from .adaln import AdaLNModulation, modulate


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, mlp_ratio: float, dropout: float, action_embed_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.mlp = MLP(d_model, mlp_ratio, dropout)
        self.modulation = AdaLNModulation(action_embed_dim, d_model)

    def forward(self, x: torch.Tensor, action_emb: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = self.modulation(action_emb)
        attn_in = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=attn_mask, need_weights=False)
        x = x + gate_attn * attn_out
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(mlp_in)
        return x


class CausalTransformerPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        action_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        compressed_tokens: int,
        context_len: int,
        predict_delta: bool = True,
        action_embed_dim: int | None = None,
        delta_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.compressed_tokens = compressed_tokens
        self.context_len = context_len
        self.predict_delta = predict_delta
        # ABLATION (3): emit M != K delta tokens. M learned query tokens cross-attend the K
        # per-slot outputs to pool K -> M before the delta head. delta_tokens is set ONLY in
        # delta-aware mode (predict_delta=False); when None the predictor is unchanged (K tokens).
        self.delta_tokens = int(delta_tokens) if delta_tokens else None
        action_embed_dim = action_embed_dim or d_model
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.SiLU(),
            nn.Linear(action_embed_dim, action_embed_dim),
        )
        self.frame_pos = nn.Parameter(torch.zeros(1, context_len, 1, d_model))
        self.token_pos = nn.Parameter(torch.zeros(1, 1, compressed_tokens, d_model))
        self.blocks = nn.ModuleList(
            [AdaLNTransformerBlock(d_model, heads, mlp_ratio, dropout, action_embed_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model)
        if self.delta_tokens is not None:
            self.delta_query = nn.Parameter(torch.zeros(1, self.delta_tokens, d_model))
            nn.init.normal_(self.delta_query, std=0.02)
            self.delta_pool = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        nn.init.normal_(self.frame_pos, std=0.02)
        nn.init.normal_(self.token_pos, std=0.02)

    def _block_causal_mask(self, context_len: int, k: int, device: torch.device) -> torch.Tensor:
        frame_ids = torch.arange(context_len, device=device).repeat_interleave(k)
        return frame_ids[:, None] < frame_ids[None, :]

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict the next compressed state.

        Args:
            states: [B, C, K, d_model]
            actions: [B, C] or longer action sequence with last context action at index C-1
        Returns:
            next_state: [B, K, d_model]
        """
        b, c, k, d = states.shape
        if c > self.frame_pos.shape[1]:
            raise ValueError(f"context_len={c} exceeds configured maximum {self.frame_pos.shape[1]}")
        x = states + self.frame_pos[:, :c] + self.token_pos[:, :, :k]
        x = x.reshape(b, c * k, d)
        action_emb = self.action_embed(actions[:, c - 1])
        attn_mask = self._block_causal_mask(c, k, states.device)
        for block in self.blocks:
            x = block(x, action_emb, attn_mask)
        last = self.norm(x[:, -k:, :])                       # [B, K, d]
        if self.delta_tokens is not None:
            q = self.delta_query.expand(b, -1, -1)
            last, _ = self.delta_pool(q, last, last, need_weights=False)  # [B, M, d]
            return self.head(last)                           # M-token delta (delta-aware mode)
        delta = self.head(last)
        return states[:, -1] + delta if self.predict_delta else delta
