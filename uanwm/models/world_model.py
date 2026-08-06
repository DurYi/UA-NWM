from __future__ import annotations

import torch
from torch import nn

from .causal_transformer import CausalTransformerPredictor
from .delta_tokenizer import DeltaDecoder, DeltaEncoder
from .dino_encoder import DINOEncoder
from .token_compressor import TokenCompressor
from .token_fuser import TokenFuser


class UncertaintyAwareWorldModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        train_cfg = cfg["training"]
        self.context_len = int(train_cfg["context_len"])
        self.encoder = DINOEncoder(
            model_name=model_cfg["encoder_name"],
            freeze=bool(model_cfg.get("freeze_encoder", True)),
            image_size=int(model_cfg["image_size"]),
            raw_num_tokens=int(model_cfg.get("raw_num_tokens", 196)),
            num_register_tokens=model_cfg.get("num_register_tokens"),
        )
        self.compressor = TokenCompressor(
            d_model=int(model_cfg["d_model"]),
            compressed_tokens=int(model_cfg["compressed_tokens"]),
            heads=int(model_cfg["predictor_heads"]),
            dropout=float(model_cfg["dropout"]),
            input_dim=int(model_cfg["dino_dim"]),
        )
        fuser_cfg = model_cfg.get("token_fuser", {})
        self.fuser = TokenFuser(
            d_model=int(model_cfg["d_model"]),
            dino_dim=int(model_cfg["dino_dim"]),
            num_output_tokens=int(fuser_cfg.get("num_output_tokens", model_cfg.get("raw_num_tokens", 196))),
            num_heads=int(fuser_cfg.get("num_heads", model_cfg["predictor_heads"])),
            depth=int(fuser_cfg.get("depth", 2)),
            dropout=float(model_cfg["dropout"]),
            hep_mode=str(model_cfg.get("hep", {}).get("mode", "none")),
            hep_rank=int(model_cfg.get("hep", {}).get("rank", 4)),
            hep_scales=tuple(model_cfg.get("hep", {}).get("scales", (1, 2, 7, 14))),
            hep_use_context=bool(model_cfg.get("hep", {}).get("use_context", False)),
            hep_context_mode=str(model_cfg.get("hep", {}).get("context_mode", "mean")),
            hep_context_heads=int(model_cfg.get("hep", {}).get("context_heads", 4)),
        )
        # --- Delta-aware tokenizer (ported from dino_feature_ar_wm). When enabled the
        # predictor emits a DELTA latent (predict_delta=False -> raw head output), and
        # a learned DeltaDecoder composes z_{t+1} = z_t (+) delta. A DeltaEncoder gives
        # the teacher delta at training time. The AR context stays ABSOLUTE latent z, so
        # every downstream read-out head (fuser mean, HEP uncertainty) is unchanged. ---
        delta_cfg = model_cfg.get("delta_aware", {})
        self.delta_aware_enabled = bool(delta_cfg.get("enabled", False))
        self.delta_encoder: DeltaEncoder | None = None
        self.delta_decoder: DeltaDecoder | None = None
        # ABLATION (3): optional M != K delta-token count (default None -> M = K (UA-NWM default))
        num_delta_tokens = delta_cfg.get("delta_tokens", None)
        num_delta_tokens = int(num_delta_tokens) if num_delta_tokens else None
        if self.delta_aware_enabled:
            delta_dim = int(delta_cfg.get("delta_dim", model_cfg["d_model"]))
            delta_heads = int(delta_cfg.get("heads", model_cfg["predictor_heads"]))
            delta_mlp_ratio = float(delta_cfg.get("mlp_ratio", model_cfg.get("mlp_ratio", 4.0)))
            self.delta_encoder = DeltaEncoder(
                d_model=int(model_cfg["d_model"]), delta_dim=delta_dim,
                layers=int(delta_cfg.get("delta_encoder_layers", 2)), heads=delta_heads,
                mlp_ratio=delta_mlp_ratio, dropout=float(delta_cfg.get("dropout", model_cfg["dropout"])),
                num_delta_tokens=num_delta_tokens)
            self.delta_decoder = DeltaDecoder(
                d_model=int(model_cfg["d_model"]), delta_dim=delta_dim,
                layers=int(delta_cfg.get("delta_decoder_layers", 2)), heads=delta_heads,
                mlp_ratio=delta_mlp_ratio, dropout=float(delta_cfg.get("dropout", model_cfg["dropout"])),
                residual=bool(delta_cfg.get("delta_decoder_residual", True)),
                num_delta_tokens=num_delta_tokens)
        # In delta mode the predictor outputs the raw delta latent (no z_{t-1}+delta).
        predict_delta = bool(model_cfg.get("predict_delta", True)) and not self.delta_aware_enabled
        self.predictor = CausalTransformerPredictor(
            d_model=int(model_cfg["d_model"]),
            action_dim=int(model_cfg["action_dim"]),
            depth=int(model_cfg["predictor_depth"]),
            heads=int(model_cfg["predictor_heads"]),
            mlp_ratio=float(model_cfg["mlp_ratio"]),
            dropout=float(model_cfg["dropout"]),
            compressed_tokens=int(model_cfg["compressed_tokens"]),
            context_len=int(train_cfg["context_len"]),
            predict_delta=predict_delta,
            action_embed_dim=int(model_cfg.get("action_embed_dim", model_cfg["d_model"])),
            delta_tokens=(num_delta_tokens if self.delta_aware_enabled else None),
        )

    def encode_state(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Encode raw DINO features into compressed latent state.

        Args:
            raw_features: [B, T, N, D_dino]
        Returns:
            compressed: [B, T, K, d_model]
        """
        return self.compressor(raw_features)

    def fuse_state(self, compressed_state: torch.Tensor) -> torch.Tensor:
        """Fuse one compressed latent state back to dense DINO-like features.

        Args:
            compressed_state: [B, K, d_model]
        Returns:
            dense_features: [B, N, D_dino].
        """
        return self.fuser(compressed_state)

    def fuse_sequence(self, compressed_states: torch.Tensor) -> torch.Tensor:
        """Fuse a sequence of compressed latent states.

        Args:
            compressed_states: [B, T, K, d_model]
        Returns:
            dense_features: [B, T, N, D_dino].
        """
        b, t, k, d = compressed_states.shape
        flat = compressed_states.reshape(b * t, k, d)
        dense = self.fuse_state(flat)
        return dense.reshape(b, t, dense.shape[1], dense.shape[2])

    def load_state_dict(self, state_dict, strict: bool = True):
        # Back-compat: the HEP module (paper: Hierarchical Error Projection) was formerly named
        # `resid_head`. Remap any legacy `...resid_head...` keys to `...hep...` so older checkpoints
        # (e.g. the released pretrained/ua-nwm/*.pt) still load under the paper-aligned naming.
        if any("resid_head." in k for k in state_dict):
            state_dict = {k.replace("resid_head.", "hep."): v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        context_raw: torch.Tensor,
        target_raw: torch.Tensor,
        actions: torch.Tensor,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict rollout targets from raw DINO feature sequences.

        Args:
            context_raw: [B, C, N, D_dino]
            target_raw: [B, H, N, D_dino]
            actions: [B, C + H - 1, action_dim]
            steps: rollout horizon H
        Returns:
            outputs with pred/target latents and dense DINO features.
        """
        context_states = self.encode_state(context_raw)
        with torch.no_grad():
            target_states = self.encode_state(target_raw).detach()
        rollout_outputs = self.rollout(context_states, actions, steps=steps)
        rec_features = self.fuse_state(context_states[:, -1])
        rollout_outputs.update(
            {
                "target_latents": target_states,
                "target_features": target_raw.detach(),
                "context_states": context_states,
                "reconstruction_features": rec_features,
                "reconstruction_target": context_raw[:, -1].detach(),
            }
        )
        return rollout_outputs

    def predict_next(self, context_states: torch.Tensor, context_actions: torch.Tensor) -> torch.Tensor:
        """Predict one future latent state.

        Args:
            context_states: [B, C, K, d_model]
            context_actions: [B, C, action_dim]
        Returns:
            next_state: [B, K, d_model]
        """
        return self.predictor(context_states, context_actions)

    def encode_delta(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        if self.delta_encoder is None:
            raise RuntimeError("encode_delta requires model.delta_aware.enabled=true.")
        return self.delta_encoder(z_t, z_next)

    def decode_delta(self, z_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.delta_decoder is None:
            raise RuntimeError("decode_delta requires model.delta_aware.enabled=true.")
        return self.delta_decoder(z_t, delta)

    def rollout(
        self,
        init_states: torch.Tensor,
        actions: torch.Tensor,
        steps: int,
    ) -> dict[str, torch.Tensor]:
        """Autoregressively roll out compressed states.

        Args:
            init_states: [B, C, K, d_model]
            actions: [B, C + steps - 1, action_dim]
        Returns:
            pred_latents: [B, steps, K, d_model]
            pred_features: [B, steps, N, D_dino]
        """
        context = init_states
        preds = []
        c = init_states.shape[1]
        for step in range(steps):
            action_window = actions[:, step : step + c]
            if self.delta_aware_enabled:
                # predictor emits a delta latent; decoder composes the next absolute state
                delta_hat = self.predictor(context, action_window)
                next_state = self.decode_delta(context[:, -1], delta_hat)
            else:
                next_state = self.predict_next(context, action_window)
            preds.append(next_state)
            context = torch.cat([context[:, 1:], next_state.unsqueeze(1)], dim=1)
        pred_latents = torch.stack(preds, dim=1)
        return {"pred_latents": pred_latents, "pred_features": self.fuse_sequence(pred_latents)}

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode image sequences.

        Args:
            images: [B, T, 3, H, W]
        Returns:
            patch_tokens: [B, T, N, D_dino]
        """
        b, t, c, h, w = images.shape
        tokens = self.encoder(images.reshape(b * t, c, h, w))
        return tokens.reshape(b, t, tokens.shape[1], tokens.shape[2])
