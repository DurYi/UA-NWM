"""Train the HEP uncertainty subspace on top of the frozen mean predictor.

The head learns per-patch uncertainty directions so the GT future deviation lies
in the predicted subspace.
"""
from __future__ import annotations
import argparse, contextlib, math, sys
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from uanwm.models import UncertaintyAwareWorldModel
from uanwm.train_common import ActionNormalizer, build_loaders, estimate_action_stats, prepare_features
from uanwm.utils.checkpoint import save_checkpoint
from uanwm.utils.config import load_config
from uanwm.utils.distributed import barrier, cleanup_distributed, init_distributed, is_main_process
from uanwm.utils.logging import setup_logging
from uanwm.utils.seed import seed_everything
from uanwm.utils.tensor import get_autocast_dtype


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--init_checkpoint", default=None)
    return p.parse_args()


def normalize_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    r = cfg.setdefault("model", {}).setdefault("hep", {})
    r.setdefault("mode", "subspace"); r.setdefault("rank", 4)
    t = cfg.setdefault("hep_training", {})
    t.setdefault("phase", "phase1"); t.setdefault("max_steps", 8000); t.setdefault("lr", 5.0e-4)
    t.setdefault("min_lr_ratio", 0.02); t.setdefault("weight_decay", 0.0); t.setdefault("warmup_steps", 300)
    t.setdefault("grad_clip_norm", 1.0); t.setdefault("log_every", 50); t.setdefault("eval_every_steps", 500)
    t.setdefault("save_every_steps", 4000); t.setdefault("rollout_steps", cfg["training"].get("rollout_steps", 2))
    t.setdefault("init_checkpoint", None)
    cfg["training"]["rollout_steps"] = int(t["rollout_steps"]); cfg.setdefault("output_dir", "outputs")
    return cfg


def apply_trainability(core, phase):
    for p in core.parameters():
        p.requires_grad = False
    if core.fuser.hep is None:
        raise RuntimeError("no hep; set model.hep.mode!='none'.")
    for p in core.fuser.hep.parameters():
        p.requires_grad = True
    return ["fuser.hep"]


def build_opt_sched(core, t):
    opt = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],
                            lr=float(t["lr"]), betas=(0.9, 0.95), weight_decay=float(t["weight_decay"]))
    ms = max(1, int(t["max_steps"])); wu = int(t["warmup_steps"]); mr = float(t["min_lr_ratio"])
    def f(s):
        if wu > 0 and s < wu: return max(1e-8, (s + 1) / wu)
        pr = (s - wu) / max(1, ms - wu); return mr + 0.5 * (1 - mr) * (1 + math.cos(math.pi * min(1.0, pr)))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, f)


def hep_forward(core, batch, normalizer, cfg, device, amp_dtype, t):
    actions = normalizer(batch["actions"].to(device, non_blocking=True))
    context_raw, target_raw = prepare_features(core, batch, cfg["training"]["mode"], device, amp_dtype)
    steps = int(t["rollout_steps"])
    ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
           if device.type == "cuda" and amp_dtype is not None else contextlib.nullcontext())
    with ctx:
        cs = core.encode_state(context_raw)
        roll = core.rollout(cs, actions, steps)
        states = roll["pred_latents"]; tgt = target_raw.detach().float()
        # Pass full context tokens [B, frames*Kc, d]; the head reduces them according to context_mode.
        ctx_sum = cs.flatten(1, 2) if getattr(core.fuser.hep, "use_context", False) else None
        cost = 0.0
        for s in range(steps):
            cost = cost + core.fuser.hep_cost(states[:, s], tgt[:, s], context_summary=ctx_sum).mean()
        cost = cost / steps
        if not t.get("contrastive"):
            return {"loss": cost, "ortho_cost": cost.detach()}
        # ---- CONTRASTIVE: S should explain the GT deviation (low ortho cost) but NOT a
        # wrong-rollout deviation toward the SAME goal (high ortho). This is exactly the
        # selection mechanism (eval re-ranks candidates by ortho cost of goal-vs-mean), so a
        # correct contrastive should sharpen selection. Knobs (hep_training.*):
        #   contrastive_neg: 'perm' (in-batch action permutation, a wildly different traj) |
        #                    'perturb' (GT action + Gaussian noise -> a NEAR-MISS traj, the hard
        #                     negative that mirrors close-but-wrong candidates) | 'both'
        #   contrastive_perturb_std : noise scale on the (normalized) actions for 'perturb'
        #   contrastive_num_neg     : negatives sampled per positive
        #   contrastive_all_steps   : score the contrast over ALL rollout steps (matches roll8)
        #                             instead of only the goal step
        #   contrastive_type: 'margin' (softplus rank hinge) | 'infonce' (softmax, temp)
        b = actions.shape[0]
        neg_kind = str(t.get("contrastive_neg", "perm"))
        num_neg = int(t.get("contrastive_num_neg", 1))
        sidx = list(range(steps)) if t.get("contrastive_all_steps") else [steps - 1]
        neg_states = []
        if neg_kind in ("perm", "both"):
            for _ in range(1 if neg_kind == "both" else num_neg):
                perm = (torch.arange(b, device=device) + 1 + torch.randint(0, b - 1, (1,), device=device)) % b
                neg_states.append(core.rollout(cs, actions[perm], steps)["pred_latents"])
        if neg_kind in ("perturb", "both"):
            std = float(t.get("contrastive_perturb_std", 0.5))
            for _ in range(1 if neg_kind == "both" else num_neg):
                neg_states.append(core.rollout(cs, actions + std * torch.randn_like(actions), steps)["pred_latents"])

        def pcost(st):                            # per-sample ortho cost, mean over patches & selected steps -> [B]
            c = 0.0
            for s in sidx:
                c = c + core.fuser.hep_cost(st[:, s], tgt[:, s], context_summary=ctx_sum).mean(dim=1)
            return c / len(sidx)
        og = pcost(states)                        # GT rollout (want low)
        on = torch.stack([pcost(ns) for ns in neg_states], 0)   # [num_neg, B] (want high)
        rw = float(t.get("recon_weight", 1.0)); cw = float(t.get("contrastive_weight", 1.0))
        if str(t.get("contrastive_type", "margin")) == "infonce":
            temp = float(t.get("contrastive_temp", 0.1))
            logits = torch.cat([(-og).unsqueeze(0), -on], 0).t() / temp     # [B, 1+num_neg], higher=better; pos=col0
            ctr_loss = torch.nn.functional.cross_entropy(logits, torch.zeros(b, dtype=torch.long, device=device))
        else:
            margin = float(t.get("contrastive_margin", 0.1))
            ctr_loss = torch.nn.functional.softplus(margin - (on - og.unsqueeze(0))).mean()  # want on - og >= margin
        loss = rw * cost + cw * ctr_loss
    return {"loss": loss, "ortho_cost": cost.detach(), "rank_loss": ctr_loss.detach(),
            "ortho_gt": og.mean().detach(), "ortho_neg": on.mean().detach()}


@torch.no_grad()
def evaluate(core, loader, normalizer, cfg, device, amp_dtype, t):
    core.eval(); tot = {}; cnt = 0
    for batch in loader:
        d = hep_forward(core, batch, normalizer, cfg, device, amp_dtype, t); bs = batch["actions"].shape[0]
        for k, v in d.items(): tot[k] = tot.get(k, 0.0) + float(v) * bs
        cnt += bs
    if dist.is_available() and dist.is_initialized():
        keys = sorted(tot); vals = torch.tensor([tot[k] for k in keys] + [float(cnt)], dtype=torch.float64, device=device)
        dist.all_reduce(vals, op=dist.ReduceOp.SUM); tot = {k: float(vals[i]) for i, k in enumerate(keys)}; cnt = int(vals[-1])
    return {f"val/{k}": v / max(1, cnt) for k, v in tot.items()}


def load_mean_checkpoint(path, core, device):
    state = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = core.load_state_dict(state.get("model", state), strict=False)
    missing = [m for m in missing if "hep" not in m and not m.startswith("encoder.")]
    if missing or [u for u in unexpected if not u.startswith("encoder.")]:
        print(f"[load] WARNING missing={missing}")


def main():
    args = parse_args()
    cfg = normalize_cfg(load_config(args.config)); t = cfg["hep_training"]
    rank, world_size, local_rank = init_distributed(cfg.get("distributed", {}).get("enabled", True))
    seed_everything(int(cfg.get("seed", 42)) + rank)
    exp_dir = Path(cfg["output_dir"]) / cfg["exp_name"]; logger = setup_logging(exp_dir, rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_ds, train_loader, val_loader, train_sampler = build_loaders(cfg, rank, world_size)
    action_stats_path = cfg["paths"].get("action_stats")
    if action_stats_path and Path(action_stats_path).exists():
        normalizer = ActionNormalizer.from_json(action_stats_path)
    else:
        if action_stats_path and is_main_process():
            logger.info("action_stats not found at %s; estimating from the training split", action_stats_path)
        normalizer = estimate_action_stats(train_ds, cfg["model"]["action_dim"])
        if is_main_process(): normalizer.to_json(exp_dir / "action_stats.json")
    normalizer = normalizer.to(device)
    core = UncertaintyAwareWorldModel(cfg)
    core.to(device)
    init_ckpt = args.init_checkpoint or t.get("init_checkpoint")
    if init_ckpt:
        load_mean_checkpoint(init_ckpt, core, device)
        if is_main_process(): logger.info("Loaded mean predictor from %s", init_ckpt)
    trained = apply_trainability(core, str(t["phase"]))
    if is_main_process():
        logger.info("stage=stage3 HEP phase=%s rank=%d | trained: %s | params: %.3fM", t["phase"], cfg["model"]["hep"]["rank"],
                    trained, sum(p.numel() for p in core.parameters() if p.requires_grad) / 1e6)
    opt, sched = build_opt_sched(core, t)
    amp_dtype = get_autocast_dtype(cfg["training"].get("amp", "bf16"))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["training"].get("amp") == "fp16")
    max_steps = int(t["max_steps"]); ckpt_dir = exp_dir / "checkpoints"; best = float("inf")
    gstep = 0; epoch = 0; it = iter(train_loader)
    try:
        while gstep < max_steps:
            core.train()
            try: batch = next(it)
            except StopIteration:
                epoch += 1
                if train_sampler is not None: train_sampler.set_epoch(epoch)
                it = iter(train_loader); batch = next(it)
            opt.zero_grad(set_to_none=True)
            d = hep_forward(core, batch, normalizer, cfg, device, amp_dtype, t)
            scaler.scale(d["loss"]).backward()
            if t.get("grad_clip_norm"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], float(t["grad_clip_norm"]))
            scaler.step(opt); scaler.update(); sched.step(); gstep += 1
            if is_main_process() and gstep % int(t["log_every"]) == 0:
                logger.info(" ".join(["stage=stage3", f"step={gstep}", f"lr={sched.get_last_lr()[0]:.3e}"] + [f"{k}={float(v):.5f}" for k, v in d.items()]))
            if gstep % int(t["eval_every_steps"]) == 0 or gstep == max_steps:
                metrics = evaluate(core, val_loader, normalizer, cfg, device, amp_dtype, t)
                sel_key = "val/rank_loss" if t.get("contrastive") else "val/ortho_cost"
                cur = metrics.get(sel_key, float("inf")); improved = cur < best; best = min(best, cur)
                if is_main_process():
                    logger.info("validation: %s", {k: round(v, 6) for k, v in sorted(metrics.items())})
                    save_checkpoint(ckpt_dir / "ua-nwm_last.pt", core, opt, sched, epoch, gstep, cfg, normalizer.as_dict())
                    if improved:
                        save_checkpoint(ckpt_dir / "ua-nwm_best.pt", core, opt, sched, epoch, gstep, cfg, normalizer.as_dict())
                barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
