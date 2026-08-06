from __future__ import annotations

import torch
from torch import nn

from .hep import HierarchicalErrorProjection


class TokenFuserBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, compressed_tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            self.norm_q(queries),
            self.norm_kv(compressed_tokens),
            self.norm_kv(compressed_tokens),
            need_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm_mlp(queries))
        return queries


class TokenFuser(nn.Module):
    def __init__(
        self,
        d_model: int,
        dino_dim: int,
        num_output_tokens: int = 196,
        num_heads: int = 6,
        depth: int = 2,
        dropout: float = 0.0,
        hep_mode: str = "none",
        hep_rank: int = 4,
        hep_scales=(1, 2, 7, 14),
        hep_use_context: bool = False,
        hep_context_mode: str = "mean",
        hep_context_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_output_tokens = int(num_output_tokens)
        self.output_queries = nn.Parameter(torch.randn(self.num_output_tokens, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TokenFuserBlock(d_model=d_model, num_heads=num_heads, dropout=dropout) for _ in range(int(depth))]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, dino_dim)

        self.hep_mode = str(hep_mode)
        self.hep: nn.Module | None = None
        if self.hep_mode == "multiscale":
            base_grid = int(round(self.num_output_tokens ** 0.5))
            self.hep = HierarchicalErrorProjection(
                d_model=d_model, dino_dim=dino_dim, base_grid=base_grid,
                scales=tuple(int(g) for g in hep_scales), rank=int(hep_rank),
                use_context=bool(hep_use_context),
                context_mode=str(hep_context_mode), context_heads=int(hep_context_heads))
        elif self.hep_mode != "none":
            raise ValueError(f"Unknown hep_mode: {self.hep_mode}")

    def hep_cost(self, compressed_tokens: torch.Tensor, target: torch.Tensor,
                   normalize: bool = True, context_summary: torch.Tensor | None = None,
                   parallel_weight: float = 0.0, active_scales=None) -> torch.Tensor:
        """Per-patch orthogonal-residual cost of `target` vs the uncertainty subspace [B,N].

        `context_summary` [B,d] is required only for the context-conditioned ('res')
        head (hep_use_context=true); ignored by the plain head.
        `parallel_weight`>0 (inference-only): add w*||e_par||^2 so far-along-plane deviations are
        also penalized (see HierarchicalErrorProjection.ortho_cost)."""
        if self.hep is None:
            raise RuntimeError("hep_cost called but TokenFuser has no hep (hep_mode='none').")
        hidden = self._hidden(compressed_tokens)
        mean = self.out_proj(hidden)
        kw = {"normalize": normalize, "parallel_weight": parallel_weight}
        if active_scales is not None and hasattr(self.hep, "scales"):  # multiscale head only
            kw["active_scales"] = active_scales
        if getattr(self.hep, "use_context", False):
            return self.hep.ortho_cost(hidden, mean, target, context_summary=context_summary, **kw)
        return self.hep.ortho_cost(hidden, mean, target, **kw)

    def _hidden(self, compressed_tokens: torch.Tensor) -> torch.Tensor:
        """Per-patch fuser hidden (shared by all read-out heads)."""
        b = compressed_tokens.shape[0]
        queries = self.output_queries.unsqueeze(0).expand(b, -1, -1)
        for block in self.blocks:
            queries = block(queries, compressed_tokens)
        return self.norm(queries)

    def forward(self, compressed_tokens: torch.Tensor) -> torch.Tensor:
        """Fuse K compressed tokens into dense DINO-like patch features.

        Args:
            compressed_tokens: [B, K, d_model]
        Returns:
            dense_features: [B, N, D_dino].
        """
        return self.out_proj(self._hidden(compressed_tokens))
