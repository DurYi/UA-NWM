from __future__ import annotations

import torch
import torch.nn.functional as F


def _step_weights(num_steps: int, loss_cfg: dict, device: torch.device) -> torch.Tensor:
    mode = str(loss_cfg.get("later_step_weighting", "none")).lower()
    if mode in {"none", "false", "0"} or num_steps <= 1:
        return torch.ones(num_steps, dtype=torch.float32, device=device)
    if mode != "linear":
        raise ValueError(f"Unsupported later_step_weighting: {mode}")
    max_weight = float(loss_cfg.get("max_step_weight", 1.0))
    return torch.linspace(1.0, max_weight, num_steps, dtype=torch.float32, device=device)


def _weighted_mean(step_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(step_values.device, dtype=step_values.dtype)
    return (step_values * weights).sum() / weights.sum().clamp_min(1e-8)


def compute_self_rollout_post_training_loss(outputs: dict[str, torch.Tensor], loss_cfg: dict) -> dict[str, torch.Tensor]:
    """Compute post-training losses for scheduled self-rollout outputs."""
    pred_features = outputs["pred_features"]
    target_features = outputs["target_features"].detach()
    pred_latents = outputs["pred_latents"]
    target_latents = outputs["target_latents"].detach()
    context_features = outputs["context_features"].detach()

    steps = pred_features.shape[1]
    weights = _step_weights(steps, loss_cfg, pred_features.device)

    dino_mse_steps = ((pred_features - target_features) ** 2).mean(dim=(0, 2, 3))
    dino_cos_steps = 1.0 - F.cosine_similarity(pred_features.float(), target_features.float(), dim=-1).mean(dim=(0, 2))
    latent_mse_steps = ((pred_latents - target_latents) ** 2).mean(dim=(0, 2, 3))

    dino_mse = _weighted_mean(dino_mse_steps, weights)
    dino_cos = _weighted_mean(dino_cos_steps, weights)
    latent_mse = _weighted_mean(latent_mse_steps, weights)

    dino_mse_weight = float(loss_cfg.get("dino_mse_weight", 1.0))
    dino_cos_weight = float(loss_cfg.get("dino_cos_weight", 0.0))
    latent_aux_weight = float(loss_cfg.get("latent_aux_weight", loss_cfg.get("latent_mse_weight", 0.0)))

    loss = dino_mse_weight * dino_mse + dino_cos_weight * dino_cos + latent_aux_weight * latent_mse

    init_feature = context_features[:, -1:].to(pred_features.dtype)
    pred_prev = torch.cat([init_feature, pred_features[:, :-1]], dim=1)
    target_prev = torch.cat([init_feature, target_features[:, :-1]], dim=1)
    pred_delta = pred_features - pred_prev
    target_delta = target_features - target_prev
    delta_loss_type = str(loss_cfg.get("delta_loss_type", "smooth_l1")).lower()
    if delta_loss_type in {"smooth_l1", "huber"}:
        delta_per_step = F.smooth_l1_loss(pred_delta, target_delta, reduction="none").mean(dim=(0, 2, 3))
    elif delta_loss_type == "mse":
        delta_per_step = ((pred_delta - target_delta) ** 2).mean(dim=(0, 2, 3))
    else:
        raise ValueError(f"Unsupported delta_loss_type: {delta_loss_type}")
    delta_loss = _weighted_mean(delta_per_step, weights)
    loss = loss + float(loss_cfg.get("delta_loss_weight", 0.0)) * delta_loss

    pred_std = pred_features.float().std(dim=(0, 1, 2))
    target_std = target_features.float().std(dim=(0, 1, 2)).detach()
    variance_loss = (pred_std - target_std).abs().mean()
    loss = loss + float(loss_cfg.get("variance_loss_weight", 0.0)) * variance_loss

    rec_features = outputs.get("fuser_gt_reconstruction_features")
    if rec_features is not None:
        fuser_gt_reconstruction_mse = ((rec_features - target_features) ** 2).mean()
    else:
        fuser_gt_reconstruction_mse = pred_features.new_zeros(())
    loss = loss + float(loss_cfg.get("fuser_gt_reconstruction_weight", 0.0)) * fuser_gt_reconstruction_mse

    reconstruction_features = outputs.get("reconstruction_features")
    reconstruction_target = outputs.get("reconstruction_target")
    if reconstruction_features is not None and reconstruction_target is not None:
        context_reconstruction_mse = ((reconstruction_features - reconstruction_target.detach()) ** 2).mean()
    else:
        context_reconstruction_mse = pred_features.new_zeros(())
    loss = loss + float(loss_cfg.get("reconstruction_mse_weight", 0.0)) * context_reconstruction_mse

    out = {
        "loss": loss,
        "dino_mse": dino_mse,
        "dino_cos": dino_cos,
        "latent_mse": latent_mse,
        "delta_loss": delta_loss,
        "variance_loss": variance_loss,
        "fuser_gt_reconstruction_mse": fuser_gt_reconstruction_mse,
        "reconstruction_mse": context_reconstruction_mse,
        "feature_variance_ratio": pred_std.mean() / target_std.mean().clamp_min(1e-8),
    }
    for i, value in enumerate(dino_mse_steps, start=1):
        out[f"dino_mse_step_{i}"] = value
    for i, value in enumerate(dino_cos_steps, start=1):
        out[f"dino_cos_step_{i}"] = value
    for i, value in enumerate(latent_mse_steps, start=1):
        out[f"latent_mse_step_{i}"] = value
    for i, value in enumerate(delta_per_step, start=1):
        out[f"delta_loss_step_{i}"] = value
    return out
