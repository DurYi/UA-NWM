from __future__ import annotations

import torch
from torch import nn


class DeltaEncoder(nn.Module):
    """Encode a latent transition z_t -> z_{t+1} into a delta latent.

    Ported from the junior repo (dino_feature_ar_wm) delta-aware tokenizer. Unlike
    DeltaToken (one token summarising the whole change), this keeps K delta tokens,
    one per compressed slot -> a MULTI-token delta. Used only as a TEACHER at
    training time (produces delta_gt); never called at inference.

    Inputs:
        z_t:    [B, K, d]
        z_next: [B, K, d]
    Output:
        delta:  [B, K, delta_dim]
    """

    def __init__(
        self,
        d_model: int,
        delta_dim: int | None = None,
        layers: int = 2,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_delta_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.delta_dim = int(delta_dim or d_model)
        # ABLATION (3): decouple the DELTA token count M from the K compressed slots. When
        # num_delta_tokens is set, M learned query tokens cross-attend the K per-slot transition
        # features to pool K -> M delta tokens (the count the predictor is trained to emit).
        # When None (default), the encoder is 1-to-1 (K -> K) exactly as before.
        self.num_delta_tokens = int(num_delta_tokens) if num_delta_tokens else None
        self.input_proj = nn.Linear(3 * self.d_model, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(heads),
            dim_feedforward=int(self.d_model * float(mlp_ratio)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=int(layers))
        if self.num_delta_tokens is not None:
            self.query = nn.Parameter(torch.zeros(1, self.num_delta_tokens, self.d_model))
            nn.init.normal_(self.query, std=0.02)
            self.pool = nn.MultiheadAttention(self.d_model, int(heads), dropout=float(dropout), batch_first=True)
        self.norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.delta_dim)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        if z_t.shape != z_next.shape:
            raise ValueError(f"z_t and z_next must have the same shape, got {z_t.shape} and {z_next.shape}")
        x = torch.cat([z_t, z_next, z_next - z_t], dim=-1)
        x = self.input_proj(x)
        x = self.blocks(x)                                   # [B, K, d]
        if self.num_delta_tokens is not None:
            q = self.query.expand(x.shape[0], -1, -1)
            x, _ = self.pool(q, x, x, need_weights=False)    # [B, M, d]
        return self.output_proj(self.norm(x))


class DeltaDecoder(nn.Module):
    """Decode current absolute latent plus delta latent into next absolute latent.

    z_{t+1} = z_t + update(z_t, delta)   (residual=True)

    This is the "reconstructor" of the delta-aware tokenizer: the predictor emits a
    delta latent, and this module composes it with the current absolute state to
    recover the next absolute state fed back to the AR context.

    Inputs:
        z_t:   [B, K, d]
        delta: [B, K, delta_dim]
    Output:
        z_next: [B, K, d]
    """

    def __init__(
        self,
        d_model: int,
        delta_dim: int | None = None,
        layers: int = 2,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        residual: bool = True,
        num_delta_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.delta_dim = int(delta_dim or d_model)
        self.residual = bool(residual)
        # ABLATION (3): when the delta has M != K tokens, compose z_{t+1} by letting the K state
        # slots (queries) CROSS-ATTEND the M delta tokens, instead of the 1-to-1 concat path.
        self.num_delta_tokens = int(num_delta_tokens) if num_delta_tokens else None
        if self.num_delta_tokens is not None:
            self.delta_kv = nn.Linear(self.delta_dim, self.d_model)
            self.cross = nn.MultiheadAttention(self.d_model, int(heads), dropout=float(dropout), batch_first=True)
        else:
            self.input_proj = nn.Linear(self.d_model + self.delta_dim, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(heads),
            dim_feedforward=int(self.d_model * float(mlp_ratio)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=int(layers))
        self.norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, z_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.num_delta_tokens is not None:
            kv = self.delta_kv(delta)                        # [B, M, d]
            attn, _ = self.cross(z_t, kv, kv, need_weights=False)  # K queries attend M deltas -> [B, K, d]
            x = z_t + attn
        else:
            if z_t.shape[:2] != delta.shape[:2]:
                raise ValueError(f"z_t and delta must share [B, K], got {z_t.shape} and {delta.shape}")
            x = self.input_proj(torch.cat([z_t, delta], dim=-1))
        x = self.blocks(x)
        update = self.output_proj(self.norm(x))
        return z_t + update if self.residual else update
