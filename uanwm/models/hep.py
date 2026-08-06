from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HierarchicalErrorProjection(nn.Module):
    """Hierarchical Error Projection (HEP).

    HEP decomposes the normalized goal discrepancy into an uncertainty-explainable
    component and an orthogonal residual by predicting a low-rank subspace S over
    a coarse-to-fine grid pyramid.
    """

    def __init__(
        self,
        d_model: int,
        dino_dim: int,
        base_grid: int = 14,
        scales=(1, 2, 7, 14),
        rank: int = 2,
        use_context: bool = False,
        context_mode: str = "mean",
        context_heads: int = 4,
    ) -> None:
        super().__init__()
        self.base_grid = int(base_grid)
        self.dino_dim = int(dino_dim)
        self.rank = int(rank)
        self.scales = [int(g) for g in scales]
        for g in self.scales:
            if self.base_grid % g != 0:
                raise ValueError(f"scale {g} does not divide base_grid {self.base_grid}")

        self.use_context = bool(use_context)
        self.context_mode = str(context_mode) if self.use_context else "none"
        in_dim = d_model * 2 if self.use_context else d_model
        if self.use_context:
            self.nets = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_dim, d_model),
                        nn.GELU(),
                        nn.Linear(d_model, d_model),
                        nn.GELU(),
                        nn.Linear(d_model, self.rank * dino_dim),
                    )
                    for _ in self.scales
                ]
            )
            if self.context_mode == "xattn":
                self.ctx_attn = nn.ModuleList(
                    [nn.MultiheadAttention(d_model, context_heads, batch_first=True) for _ in self.scales]
                )
                self.ctx_qnorm = nn.ModuleList([nn.LayerNorm(d_model) for _ in self.scales])
                self.ctx_knorm = nn.LayerNorm(d_model)
        else:
            self.nets = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_model, d_model),
                        nn.GELU(),
                        nn.Linear(d_model, self.rank * dino_dim),
                    )
                    for _ in self.scales
                ]
            )

    def _context_feature(self, gi: int, g: int, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return a per-cell context feature with shape [B, g*g, d]."""
        _, ncell, _ = query.shape
        if self.context_mode == "mean":
            cm = context if context.dim() == 2 else context.mean(dim=1)
            return cm.unsqueeze(1).expand(-1, ncell, -1)
        if self.context_mode == "xattn":
            k = self.ctx_knorm(context.float())
            att, _ = self.ctx_attn[gi](self.ctx_qnorm[gi](query.float()), k, k, need_weights=False)
            return att
        if self.context_mode == "spatial":
            xt = context.float().transpose(1, 2)
            pooled = F.adaptive_avg_pool1d(xt, ncell)
            return pooled.transpose(1, 2)
        raise ValueError(f"unknown context_mode {self.context_mode}")

    def _to_cells(self, x: torch.Tensor, g: int) -> torch.Tensor:
        """[B,N,D] -> [B, g*g, bs*bs, D]."""
        b, _, d = x.shape
        G = self.base_grid
        bs = G // g
        return (
            x.reshape(b, G, G, d)
            .reshape(b, g, bs, g, bs, d)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, g * g, bs * bs, d)
        )

    def _from_cells(self, x: torch.Tensor, g: int) -> torch.Tensor:
        """[B, g*g, bs*bs, D] -> [B,N,D]."""
        b = x.shape[0]
        G = self.base_grid
        bs = G // g
        d = x.shape[-1]
        return (
            x.reshape(b, g, g, bs, bs, d)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, G, G, d)
            .reshape(b, G * G, d)
        )

    def ortho_cost(
        self,
        hidden: torch.Tensor,
        mean: torch.Tensor,
        target: torch.Tensor,
        normalize: bool = True,
        context_summary: torch.Tensor | None = None,
        parallel_weight: float = 0.0,
        active_scales=None,
    ) -> torch.Tensor:
        act = None if active_scales is None else {int(g) for g in active_scales}
        b, _, _ = hidden.shape
        D = self.dino_dim
        e = (F.normalize(target, dim=-1) - F.normalize(mean, dim=-1)).float()
        r = e
        for gi, g in enumerate(self.scales):
            if act is not None and g not in act:
                continue
            hc = self._to_cells(hidden, g).mean(dim=2)
            if self.use_context:
                if context_summary is None:
                    raise RuntimeError("context HEP module needs a context_summary ([B,d] mean or [B,M,d] tokens).")
                ctx = self._context_feature(gi, g, hc, context_summary).to(hc.dtype)
                hc = torch.cat([hc, ctx], dim=-1)
            basis = self.nets[gi](hc).reshape(b, g * g, self.rank, D).float()
            rc = self._to_cells(r, g)
            rbar = rc.mean(dim=2)
            rhs = torch.einsum("bcrd,bcd->bcr", basis, rbar).float()
            gram = torch.einsum("bcrd,bcsd->bcrs", basis, basis).float()
            eye = torch.eye(self.rank, device=basis.device, dtype=gram.dtype).expand_as(gram)
            coords = torch.linalg.solve(gram + 1e-3 * eye, rhs.unsqueeze(-1)).squeeze(-1)
            proj = torch.einsum("bcrd,bcr->bcd", basis, coords)
            rc = rc - proj.unsqueeze(2)
            r = self._from_cells(rc, g)
        rperp2 = (r ** 2).sum(-1)
        e2 = (e ** 2).sum(-1)
        if parallel_weight:
            cost = rperp2 + float(parallel_weight) * (e2 - rperp2).clamp_min(0.0)
        else:
            cost = rperp2
        return cost / e2.clamp_min(1e-9) if normalize else cost


__all__ = ["HierarchicalErrorProjection"]
