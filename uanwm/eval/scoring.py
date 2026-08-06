from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from uanwm.models import UncertaintyAwareWorldModel
from uanwm.train_common import ActionNormalizer


@torch.no_grad()
def score_candidates(
    model: UncertaintyAwareWorldModel,
    normalizer: ActionNormalizer,
    context_images: torch.Tensor,
    goal_image: torch.Tensor,
    candidates: torch.Tensor,
    eval_steps: int,
    amp_dtype: torch.dtype | None,
    score_metric: str,
    repeats: int,
    hep_active_scales=None,
) -> dict[str, torch.Tensor]:
    """Score candidate action sequences with the final UA-NWM scorers.

    Supported metrics:
      - ``hep``: HEP orthogonal-residual cost.
      - ``dinov3_cosine``: deterministic mean-predictor cosine baseline.
    """
    device = context_images.device
    k, t, a = candidates.shape
    if score_metric not in {"hep", "dinov3_cosine"}:
        raise ValueError(f"Unknown score_metric: {score_metric}. Expected 'hep' or 'dinov3_cosine'.")
    use_hep = score_metric == "hep"
    if use_hep and getattr(model.fuser, "hep", None) is None:
        raise ValueError("HEP scoring needs a hep (model.hep.mode!=none).")

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype is not None
        else contextlib.nullcontext()
    )
    with amp_ctx:
        context_raw = model.encode_images(context_images.unsqueeze(0))
        goal_raw = model.encode_images(goal_image.unsqueeze(0).unsqueeze(1))[:, 0]
        context_states = model.encode_state(context_raw)

    context_states = context_states.repeat(k, 1, 1, 1)
    goal_raw = goal_raw.repeat(k, 1, 1)
    scores = []
    mse_values = []
    cos_values = []
    for _ in range(max(1, int(repeats))):
        action_seq = normalizer(candidates.reshape(k, t, a)).to(device)
        with amp_ctx:
            outputs = model.rollout(context_states, action_seq, steps=eval_steps)
            final_pred = outputs["pred_features"][:, -1]
            if use_hep:
                hep_ctx = (
                    context_states.flatten(1, 2)
                    if getattr(model.fuser.hep, "use_context", False)
                    else None
                )
                hep_c = model.fuser.hep_cost(
                    outputs["pred_latents"][:, -1],
                    goal_raw.float(),
                    normalize=False,
                    context_summary=hep_ctx,
                    active_scales=hep_active_scales,
                ).float().mean(dim=1)

        pred = final_pred.float()
        goal = goal_raw.float()
        mse = F.mse_loss(pred, goal, reduction="none").mean(dim=(1, 2))
        cos = 1.0 - F.cosine_similarity(pred.flatten(1), goal.flatten(1), dim=1)

        if use_hep:
            score = hep_c
        else:
            score = cos

        scores.append(score)
        mse_values.append(mse)
        cos_values.append(cos)
    return {
        "score": torch.stack(scores, dim=0).mean(dim=0),
        "dino_mse": torch.stack(mse_values, dim=0).mean(dim=0),
        "dino_cos": torch.stack(cos_values, dim=0).mean(dim=0),
    }
