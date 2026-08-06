from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_prediction_loss(preds: torch.Tensor, targets: torch.Tensor, loss_cfg: dict) -> dict[str, torch.Tensor]:
    """Compute rollout prediction losses.

    Args:
        preds: [B, H, K, d_model]
        targets: [B, H, K, d_model]
    """
    mse_weight = float(loss_cfg.get("mse_weight", 1.0))
    cos_weight = float(loss_cfg.get("cos_weight", 0.0))
    mse_steps = ((preds - targets) ** 2).mean(dim=(0, 2, 3))
    cos_steps = 1.0 - F.cosine_similarity(preds.float(), targets.float(), dim=-1).mean(dim=(0, 2))
    mse_loss = mse_steps.mean()
    cos_loss = cos_steps.mean()
    loss = mse_weight * mse_loss + cos_weight * cos_loss
    out = {"loss": loss, "mse": mse_loss, "cos": cos_loss}
    for i, value in enumerate(mse_steps, start=1):
        out[f"mse_step_{i}"] = value
    for i, value in enumerate(cos_steps, start=1):
        out[f"cos_step_{i}"] = value
    return out


def _step_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return ((preds - targets) ** 2).mean(dim=tuple(range(preds.ndim))[0:1] + tuple(range(2, preds.ndim)))


def _step_cos(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(preds.float(), targets.float(), dim=-1).mean(dim=(0, 2))


def compute_dense_dino_prediction_loss(outputs: dict[str, torch.Tensor], loss_cfg: dict) -> dict[str, torch.Tensor]:
    """Compute dense DINO supervision losses.

    Required output shapes:
        pred_features: [B, H, N, D_dino]
        target_features: [B, H, N, D_dino]
        pred_latents: [B, H, K, d_model]
        target_latents: [B, H, K, d_model]
        reconstruction_features: [B, N, D_dino]
        reconstruction_target: [B, N, D_dino]
    """
    pred_features = outputs["pred_features"]
    target_features = outputs["target_features"].detach()
    pred_latents = outputs["pred_latents"]
    target_latents = outputs["target_latents"].detach()
    rec_features = outputs["reconstruction_features"]
    rec_target = outputs["reconstruction_target"].detach()

    dino_mse_weight = float(loss_cfg.get("dino_mse_weight", loss_cfg.get("mse_weight", 1.0)))
    dino_cos_weight = float(loss_cfg.get("dino_cos_weight", 0.0))
    latent_mse_weight = float(loss_cfg.get("latent_mse_weight", 0.0))
    reconstruction_mse_weight = float(loss_cfg.get("reconstruction_mse_weight", 0.0))

    dino_mse_steps = _step_mse(pred_features, target_features)
    dino_cos_steps = _step_cos(pred_features, target_features)
    latent_mse_steps = _step_mse(pred_latents, target_latents)
    rec_mse = ((rec_features - rec_target) ** 2).mean()

    dino_mse = dino_mse_steps.mean()
    dino_cos = dino_cos_steps.mean()
    latent_mse = latent_mse_steps.mean()
    loss = (
        dino_mse_weight * dino_mse
        + dino_cos_weight * dino_cos
        + latent_mse_weight * latent_mse
        + reconstruction_mse_weight * rec_mse
    )

    out = {
        "loss": loss,
        "dino_mse": dino_mse,
        "dino_cos": dino_cos,
        "latent_mse": latent_mse,
        "reconstruction_mse": rec_mse,
    }
    for i, value in enumerate(dino_mse_steps, start=1):
        out[f"dino_mse_step_{i}"] = value
    for i, value in enumerate(dino_cos_steps, start=1):
        out[f"dino_cos_step_{i}"] = value
    for i, value in enumerate(latent_mse_steps, start=1):
        out[f"latent_mse_step_{i}"] = value
    return out
