from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uanwm.losses import compute_dense_dino_prediction_loss
from uanwm.models import UncertaintyAwareWorldModel
from uanwm.train_common import (
    ActionNormalizer,
    build_loaders,
    estimate_action_stats,
    prepare_features,
    unwrap_model,
)
from uanwm.utils.checkpoint import load_checkpoint, save_checkpoint
from uanwm.utils.config import load_config
from uanwm.utils.distributed import barrier, cleanup_distributed, init_distributed, is_main_process
from uanwm.utils.logging import setup_logging
from uanwm.utils.seed import seed_everything
from uanwm.utils.tensor import get_autocast_dtype


class Stage1AutoEncoderModule(torch.nn.Module):
    """Train Adapter + TokenCompressor + TokenFuser as a per-frame DINO feature autoencoder."""

    def __init__(self, model: torch.nn.Module, stage_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.stage_cfg = stage_cfg

    def forward(self, context_raw: torch.Tensor, target_raw: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        parts = []
        if bool(self.stage_cfg.get("reconstruct_context", True)):
            parts.append(context_raw)
        if bool(self.stage_cfg.get("reconstruct_target", True)):
            parts.append(target_raw)
        if not parts:
            raise ValueError("Stage 1 needs at least one of reconstruct_context/reconstruct_target enabled.")

        raw_features = torch.cat(parts, dim=1)
        b, t, n, d = raw_features.shape
        frame_features = raw_features.reshape(b * t, n, d)

        latents = self.model.encode_state(frame_features.unsqueeze(1)).squeeze(1)
        fuser_input = latents
        noise_std = float(self.stage_cfg.get("latent_noise_std", 0.0))
        if self.training and noise_std > 0.0:
            fuser_input = fuser_input + torch.randn_like(fuser_input) * noise_std
        rec_features = self.model.fuse_state(fuser_input)

        target = frame_features.detach()
        rec_mse = ((rec_features - target) ** 2).mean()
        rec_cos = 1.0 - F.cosine_similarity(rec_features.float(), target.float(), dim=-1).mean()
        loss = (
            float(self.stage_cfg.get("reconstruction_mse_weight", 1.0)) * rec_mse
            + float(self.stage_cfg.get("reconstruction_cos_weight", 0.0)) * rec_cos
        )

        loss_dict = {
            "loss": loss,
            "reconstruction_mse": rec_mse,
            "reconstruction_cos": rec_cos,
            "latent_norm": latents.detach().float().norm(dim=-1).mean(),
        }

        outputs = {
            "reconstruction_features": rec_features,
            "reconstruction_target": target,
            "reconstruction_latents": latents,
        }
        return loss_dict, outputs


class Stage1DeltaModule(torch.nn.Module):
    """Delta-aware Stage 1: train Adapter + TokenCompressor + TokenFuser + DeltaEncoder +
    DeltaDecoder. On top of the plain per-frame autoencoder loss (keeps the tokenizer
    sharp) it adds the delta-transition objective ported from dino_feature_ar_wm:
        z_t, z_{t+1} -> DeltaEncoder -> delta ; z_t, delta -> DeltaDecoder -> z_rec_{t+1}
    supervised in both latent space (delta_z) and dense DINO space (delta_dense). Adjacent
    frames of [context ; target] are the transitions."""

    def __init__(self, model: torch.nn.Module, stage_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.stage_cfg = stage_cfg

    def forward(self, context_raw: torch.Tensor, target_raw: torch.Tensor):
        raw = torch.cat([context_raw, target_raw], dim=1)         # [B,T,N,D]
        b, t, n, d = raw.shape
        if t < 2:
            raise ValueError("Stage 1 delta needs >=2 frames per sample (context+target).")
        z = self.model.encode_state(raw)                          # [B,T,K,d]
        k, dm = z.shape[2], z.shape[3]

        # --- absolute per-frame reconstruction (unchanged tokenizer objective) ---
        flat = z.reshape(b * t, k, dm)
        noise_std = float(self.stage_cfg.get("latent_noise_std", 0.0))
        fin = flat + torch.randn_like(flat) * noise_std if (self.training and noise_std > 0.0) else flat
        rec = self.model.fuse_state(fin)                          # [B*T,N,D]
        tgt = raw.reshape(b * t, n, d).detach()
        abs_mse = ((rec - tgt) ** 2).mean()
        abs_cos = 1.0 - F.cosine_similarity(rec.float(), tgt.float(), dim=-1).mean()

        # --- delta transition (adjacent frames) ---
        z_t = z[:, :-1].reshape(b * (t - 1), k, dm)
        z_tp1 = z[:, 1:].reshape(b * (t - 1), k, dm)
        delta = self.model.encode_delta(z_t, z_tp1)
        delta_std = float(self.stage_cfg.get("delta_noise_std", 0.0))
        if self.training and delta_std > 0.0:
            delta = delta + torch.randn_like(delta) * delta_std
        z_next_rec = self.model.decode_delta(z_t, delta)
        delta_z_mse = ((z_next_rec - z_tp1.detach()) ** 2).mean()
        dfrec = self.model.fuse_state(z_next_rec)                 # [B*(T-1),N,D]
        dtgt = raw[:, 1:].reshape(b * (t - 1), n, d).detach()
        delta_dense_mse = ((dfrec - dtgt) ** 2).mean()
        delta_dense_cos = 1.0 - F.cosine_similarity(dfrec.float(), dtgt.float(), dim=-1).mean()

        loss = (
            float(self.stage_cfg.get("abs_mse_weight", 1.0)) * abs_mse
            + float(self.stage_cfg.get("abs_cos_weight", 0.0)) * abs_cos
            + float(self.stage_cfg.get("delta_z_weight", 1.0)) * delta_z_mse
            + float(self.stage_cfg.get("delta_dense_mse_weight", 0.5)) * delta_dense_mse
            + float(self.stage_cfg.get("delta_dense_cos_weight", 0.2)) * delta_dense_cos
        )
        loss_dict = {
            "loss": loss,
            "reconstruction_mse": abs_mse,
            "reconstruction_cos": abs_cos,
            "delta_z_mse": delta_z_mse,
            "delta_dense_mse": delta_dense_mse,
            "delta_dense_cos": delta_dense_cos,
            "latent_norm": z.detach().float().norm(dim=-1).mean(),
        }
        outputs = {
            "reconstruction_features": rec,
            "reconstruction_target": tgt,
            "reconstruction_latents": z,
        }
        return loss_dict, outputs


class Stage2PredictorModule(torch.nn.Module):
    """Train Predictor with frozen Adapter + TokenCompressor + TokenFuser."""

    def __init__(self, model: torch.nn.Module, stage_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.stage_cfg = stage_cfg

    def forward(
        self,
        context_raw: torch.Tensor,
        target_raw: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        steps = int(self.stage_cfg.get("rollout_steps", 2))
        outputs = self.model(context_raw, target_raw, actions, steps)
        loss_dict = compute_dense_dino_prediction_loss(outputs, self.stage_cfg.get("loss", {}))
        return loss_dict, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage training for DINO feature AR world models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Optional checkpoint to load before the selected start stage.")
    parser.add_argument("--start_stage", choices=["stage1", "stage2"], default="stage1")
    return parser.parse_args()


def normalize_staged_config(cfg: dict[str, Any]) -> dict[str, Any]:
    staged = cfg.setdefault("staged_training", {})
    stage1 = staged.setdefault("stage1", {})
    stage2 = staged.setdefault("stage2", {})

    stage1.setdefault("enabled", True)
    stage1.setdefault("max_steps", 20000)
    stage1.setdefault("lr", cfg["training"].get("lr", 1e-4))
    stage1.setdefault("min_lr_ratio", 0.01)
    stage1.setdefault("weight_decay", cfg["training"].get("weight_decay", 0.05))
    stage1.setdefault("warmup_steps", cfg["training"].get("warmup_steps", 1000))
    stage1.setdefault("grad_clip_norm", cfg["training"].get("grad_clip_norm", 1.0))
    stage1.setdefault("log_every", cfg["training"].get("log_every", 50))
    stage1.setdefault("eval_every_steps", 1000)
    stage1.setdefault("save_every_steps", 5000)
    stage1.setdefault("reconstruct_context", True)
    stage1.setdefault("reconstruct_target", True)
    stage1.setdefault("reconstruction_mse_weight", 1.0)
    stage1.setdefault("reconstruction_cos_weight", 0.0)
    stage1.setdefault("latent_noise_std", 0.0)
    stage1.setdefault("best_metric", "val/reconstruction_mse")

    stage2.setdefault("enabled", True)
    stage2.setdefault("max_steps", 50000)
    stage2.setdefault("rollout_steps", 2)
    stage2.setdefault("lr", cfg["training"].get("lr", 1e-4))
    stage2.setdefault("min_lr_ratio", 0.01)
    stage2.setdefault("weight_decay", cfg["training"].get("weight_decay", 0.05))
    stage2.setdefault("warmup_steps", cfg["training"].get("warmup_steps", 1000))
    stage2.setdefault("grad_clip_norm", cfg["training"].get("grad_clip_norm", 1.0))
    stage2.setdefault("log_every", cfg["training"].get("log_every", 50))
    stage2.setdefault("eval_every_steps", 1000)
    stage2.setdefault("save_every_steps", 5000)
    stage2.setdefault("best_metric", f"val/dino_mse_step_{int(stage2['rollout_steps'])}")
    stage2.setdefault(
        "loss",
        {
            "dino_mse_weight": 1.0,
            "dino_cos_weight": 0.05,
            "latent_mse_weight": 0.1,
            "reconstruction_mse_weight": 0.0,
        },
    )

    cfg["training"]["rollout_steps"] = int(stage2.get("rollout_steps", cfg["training"].get("rollout_steps", 2)))
    cfg.setdefault("output_dir", "outputs")
    return cfg


def freeze_module(module: torch.nn.Module | None, frozen: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = not frozen


def apply_stage_trainability(model: torch.nn.Module, stage_name: str, joint: bool = False) -> list[str]:
    core = unwrap_model(model)
    for param in core.parameters():
        param.requires_grad = False

    if stage_name == "stage1":
        freeze_module(getattr(core, "encoder", None), True)
        freeze_module(getattr(core, "compressor", None), False)
        freeze_module(getattr(core, "fuser", None), False)
        freeze_module(getattr(core, "predictor", None), True)
        # delta-aware modules (present only for delta variants) train alongside the tokenizer
        freeze_module(getattr(core, "delta_encoder", None), False)
        freeze_module(getattr(core, "delta_decoder", None), False)
        return ["encoder", "predictor"]
    if stage_name == "stage2":
        # ABLATION (4): joint=True MERGES Stage 1+2 -> the tokenizer is NOT frozen, so the
        # whole mean predictor (tokenizer + predictor + delta decoder) is trained END-TO-END in one
        # stage. Only the DINO encoder stays frozen. joint=False keeps the tokenizer frozen (UA-NWM default).
        freeze_module(getattr(core, "encoder", None), True)
        freeze_module(getattr(core, "compressor", None), not joint)
        freeze_module(getattr(core, "fuser", None), not joint)
        freeze_module(getattr(core, "predictor", None), False)
        if joint:
            # delta_decoder IS used in the rollout (decode_delta) -> train it. delta_encoder is the
            # training-time TEACHER, never called in Stage 2 -> keep it FROZEN, else DDP
            # (find_unused_parameters=False) errors on its grad-less params.
            freeze_module(getattr(core, "delta_encoder", None), True)
            freeze_module(getattr(core, "delta_decoder", None), False)
            return ["encoder", "delta_encoder"]
        return ["encoder", "compressor", "fuser"]
    raise ValueError(f"Unknown stage: {stage_name}")


def build_stage_optimizer(model: torch.nn.Module, stage_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in unwrap_model(model).parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters for this stage.")
    return torch.optim.AdamW(
        params,
        lr=float(stage_cfg["lr"]),
        betas=tuple(stage_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(stage_cfg.get("weight_decay", 0.05)),
    )


def build_stage_scheduler(optimizer: torch.optim.Optimizer, stage_cfg: dict[str, Any]):
    max_steps = max(1, int(stage_cfg["max_steps"]))
    warmup = int(stage_cfg.get("warmup_steps", 0))
    min_ratio = float(stage_cfg.get("min_lr_ratio", 0.01))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-8, float(step + 1) / float(warmup))
        progress = (step - warmup) / max(1, max_steps - warmup)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def reduce_metrics(totals: dict[str, float], count: int, device: torch.device) -> dict[str, float]:
    if dist.is_available() and dist.is_initialized():
        keys = sorted(totals.keys())
        values = torch.tensor([totals[key] for key in keys] + [float(count)], dtype=torch.float64, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        totals = {key: float(values[i].item()) for i, key in enumerate(keys)}
        count = int(values[-1].item())
    return {f"val/{key}": value / max(1, count) for key, value in totals.items()}


def forward_stage1(module, core_model, batch, cfg, device, amp_dtype):
    context_raw, target_raw = prepare_features(core_model, batch, cfg["training"]["mode"], device, amp_dtype)
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype is not None
        else contextlib.nullcontext()
    )
    with ctx:
        return module(context_raw, target_raw)


def forward_stage2(module, core_model, batch, normalizer, cfg, device, amp_dtype):
    actions = normalizer(batch["actions"].to(device, non_blocking=True))
    context_raw, target_raw = prepare_features(core_model, batch, cfg["training"]["mode"], device, amp_dtype)
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype is not None
        else contextlib.nullcontext()
    )
    with ctx:
        return module(context_raw, target_raw, actions)


@torch.no_grad()
def evaluate_stage(stage_name, module, core_model, loader, normalizer, cfg, device, amp_dtype) -> dict[str, float]:
    module.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        if stage_name == "stage1":
            loss_dict, outputs = forward_stage1(module, core_model, batch, cfg, device, amp_dtype)
            bs = outputs["reconstruction_features"].shape[0]
        else:
            loss_dict, outputs = forward_stage2(module, core_model, batch, normalizer, cfg, device, amp_dtype)
            bs = outputs["pred_features"].shape[0]
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * bs
        count += bs
    return reduce_metrics(totals, count, device)


def format_stage_log(stage_name: str, global_step: int, loss_dict: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> str:
    parts = [f"stage={stage_name}", f"step={global_step}", f"loss={float(loss_dict['loss'].detach().cpu()):.6f}"]
    for key in [
        "reconstruction_mse",
        "reconstruction_cos",
        "latent_norm",
        "dino_mse",
        "dino_cos",
        "latent_mse",
    ]:
        if key in loss_dict:
            parts.append(f"{key}={float(loss_dict[key].detach().cpu()):.6f}")
    for key in sorted(k for k in loss_dict if k.endswith("_step_1") or k.endswith("_step_2")):
        parts.append(f"{key}={float(loss_dict[key].detach().cpu()):.6f}")
    parts.append("lr=" + ",".join(f"{group['lr']:.3e}" for group in optimizer.param_groups))
    return " ".join(parts)


def save_stage_checkpoint(path, model, optimizer, scheduler, stage_name, global_step, cfg, action_stats, metrics, frozen_modules):
    save_checkpoint(path, model, optimizer, scheduler, epoch=0, step=global_step, cfg=cfg, action_stats=action_stats)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["stage_name"] = stage_name
    state["global_step"] = int(global_step)
    state["metrics"] = metrics
    state["frozen_modules"] = list(frozen_modules)
    torch.save(state, path)


def run_stage(
    stage_name: str,
    core_model: torch.nn.Module,
    train_loader,
    val_loader,
    train_sampler,
    normalizer: ActionNormalizer,
    cfg: dict[str, Any],
    device: torch.device,
    local_rank: int,
    world_size: int,
    amp_dtype: torch.dtype | None,
    logger,
) -> Path | None:
    stage_cfg = cfg["staged_training"][stage_name]
    if not bool(stage_cfg.get("enabled", True)):
        return None

    frozen_modules = apply_stage_trainability(core_model, stage_name, joint=bool(stage_cfg.get("joint", False)))
    trainable = sum(p.numel() for p in core_model.parameters() if p.requires_grad)
    optimizer = build_stage_optimizer(core_model, stage_cfg)
    scheduler = build_stage_scheduler(optimizer, stage_cfg)
    module: torch.nn.Module
    if stage_name == "stage1":
        if getattr(unwrap_model(core_model), "delta_aware_enabled", False):
            module = Stage1DeltaModule(core_model, stage_cfg).to(device)
        else:
            module = Stage1AutoEncoderModule(core_model, stage_cfg).to(device)
    else:
        module = Stage2PredictorModule(core_model, stage_cfg).to(device)
    if world_size > 1:
        module = DDP(module, device_ids=[local_rank], find_unused_parameters=False)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["training"].get("amp") == "fp16")
    max_steps = int(stage_cfg["max_steps"])
    log_every = int(stage_cfg.get("log_every", 50))
    eval_every = int(stage_cfg.get("eval_every_steps", 1000))
    save_every = int(stage_cfg.get("save_every_steps", 5000))
    best_metric_name = str(stage_cfg.get("best_metric", "val/loss"))
    best_value = float("inf")
    ckpt_dir = Path(cfg["output_dir"]) / cfg["exp_name"] / "checkpoints"
    best_path = ckpt_dir / f"{stage_name}_best.pt"
    global_step = 0
    train_epoch = 0
    data_iter = iter(train_loader)

    if is_main_process():
        logger.info("%s frozen modules: %s", stage_name, frozen_modules)
        logger.info("%s trainable parameters: %.2fM", stage_name, trainable / 1e6)

    while global_step < max_steps:
        module.train()
        try:
            batch = next(data_iter)
        except StopIteration:
            train_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(train_epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)
        if stage_name == "stage1":
            loss_dict, _ = forward_stage1(module, core_model, batch, cfg, device, amp_dtype)
        else:
            loss_dict, _ = forward_stage2(module, core_model, batch, normalizer, cfg, device, amp_dtype)
        loss = loss_dict["loss"]
        scaler.scale(loss).backward()
        if stage_cfg.get("grad_clip_norm"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in core_model.parameters() if p.requires_grad],
                float(stage_cfg["grad_clip_norm"]),
            )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        if is_main_process() and global_step % log_every == 0:
            logger.info(format_stage_log(stage_name, global_step, loss_dict, optimizer))

        should_eval = global_step % eval_every == 0 or global_step == max_steps
        if should_eval:
            metrics = evaluate_stage(stage_name, module, core_model, val_loader, normalizer, cfg, device, amp_dtype)
            current = metrics.get(best_metric_name, metrics.get("val/loss", float("inf")))
            improved = current < best_value
            if improved:
                best_value = current
            if is_main_process():
                logger.info("%s validation metrics: %s", stage_name, {k: round(v, 6) for k, v in sorted(metrics.items())})
                action_stats = normalizer.as_dict()
                save_stage_checkpoint(
                    ckpt_dir / f"{stage_name}_last.pt",
                    core_model,
                    optimizer,
                    scheduler,
                    stage_name,
                    global_step,
                    cfg,
                    action_stats,
                    metrics,
                    frozen_modules,
                )
                if improved:
                    save_stage_checkpoint(
                        best_path,
                        core_model,
                        optimizer,
                        scheduler,
                        stage_name,
                        global_step,
                        cfg,
                        action_stats,
                        metrics,
                        frozen_modules,
                    )
                if global_step % save_every == 0:
                    save_stage_checkpoint(
                        ckpt_dir / f"{stage_name}_step_{global_step:06d}.pt",
                        core_model,
                        optimizer,
                        scheduler,
                        stage_name,
                        global_step,
                        cfg,
                        action_stats,
                        metrics,
                        frozen_modules,
                    )
            barrier()

    return best_path


def main() -> None:
    args = parse_args()
    cfg = normalize_staged_config(load_config(args.config))
    rank, world_size, local_rank = init_distributed(cfg.get("distributed", {}).get("enabled", True))
    seed_everything(int(cfg.get("seed", 42)) + rank)

    exp_dir = Path(cfg["output_dir"]) / cfg["exp_name"]
    logger = setup_logging(exp_dir, rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_ds, train_loader, val_loader, train_sampler = build_loaders(cfg, rank, world_size)

    action_stats_path = cfg["paths"].get("action_stats")
    if action_stats_path and Path(action_stats_path).exists():
        normalizer = ActionNormalizer.from_json(action_stats_path)
    else:
        if action_stats_path and is_main_process():
            logger.info("action_stats not found at %s; estimating from the training split", action_stats_path)
        normalizer = estimate_action_stats(train_ds, cfg["model"]["action_dim"])
        if is_main_process():
            normalizer.to_json(exp_dir / "action_stats.json")
    normalizer = normalizer.to(device)

    core_model = UncertaintyAwareWorldModel(cfg)
    core_model.to(device)
    if args.resume:
        # non-strict: warm-starting a delta-aware model from a plain-tokenizer checkpoint
        # (baseline has no DeltaEncoder/DeltaDecoder -> those stay freshly initialised).
        load_checkpoint(args.resume, core_model, map_location=device, strict=False)

    amp_dtype = get_autocast_dtype(cfg["training"].get("amp", "bf16"))
    try:
        stage1_best = None
        if args.start_stage == "stage1":
            stage1_best = run_stage(
                "stage1",
                core_model,
                train_loader,
                val_loader,
                train_sampler,
                normalizer,
                cfg,
                device,
                local_rank,
                world_size,
                amp_dtype,
                logger,
            )
            if stage1_best is not None:
                barrier()
                load_checkpoint(stage1_best, core_model, map_location=device)

        if bool(cfg["staged_training"]["stage2"].get("enabled", True)):
            init_stage2 = cfg["staged_training"]["stage2"].get("init_checkpoint")
            if args.start_stage == "stage2" and init_stage2 and not args.resume:
                load_checkpoint(init_stage2, core_model, map_location=device)
            run_stage(
                "stage2",
                core_model,
                train_loader,
                val_loader,
                train_sampler,
                normalizer,
                cfg,
                device,
                local_rank,
                world_size,
                amp_dtype,
                logger,
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
