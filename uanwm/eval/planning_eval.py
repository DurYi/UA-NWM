"""offline_planning — standalone CEM planning (MPC) on the airgoal10k test set.

Instead of selecting among a fixed candidate pool (offline_ranking), we OPTIMIZE a continuous
action sequence with the Cross-Entropy Method so the world model's predicted final-frame features
match the goal image. The trajectory is a straight line parametrized by the endpoint: per-step
translation (sx,sy,sz) held constant over H steps + a terminal yaw phi applied at the last step
(4 vars). CEM: sample N from N(mu,diag(sigma^2)), roll out each via the WM, keep the top-k
lowest-cost elites, refit (mu,sigma), iterate. The final mu is the planned trajectory.

The CEM cost is the scorer's goal-distance — the SAME model is planned under two scorers:
  * UA-NWM               : orthogonal-residual cost (hep, HEP module)
  * UA-NWM_deterministic : free-mean cosine        (dinov3_cosine, Stage 1+2 mean predictor)

Metrics (vs GT future path): ATE, RPE. The airgoal10k "32x3" CEM config = --num_samples 32
--topk 16 --opt_steps 3 --sampler anchor.

Usage (see scripts/eval_planning.sh):
  python uanwm/eval/planning_eval.py --config configs/ua_nwm.yaml \
     --checkpoint pretrained/ua-nwm/ua-nwm_best.pt --root data/airgoal10k/test --score_metric hep \
     --sampler anchor --num_samples 32 --topk 16 --opt_steps 3 --dump_raw \
     --output_dir results/airgoal10k/offline_planning/UA-NWM
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import math
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uanwm.datasets.trajectory_dataset import angle_difference, to_local_coords, yaw_rotmat
from uanwm.datasets.transforms import build_image_transform
from uanwm.models import UncertaintyAwareWorldModel
from uanwm.utils.config import load_config
from uanwm.utils.tensor import get_autocast_dtype
from uanwm.eval.action_norm import load_action_normalizer
from uanwm.eval.nav_utils import candidate_local_actions, list_traj_dirs, pose_traj_xyzyaw, load_traj_pkl
from uanwm.eval.scoring import score_candidates
from uanwm.eval.validate_from_raw import eval_planning

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "airgoal10k" / "test"
_CEM_RNG_ANCHOR = 241  # CEM sampling anchor kept here so planning stays reproducible run-to-run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="model config")
    p.add_argument("--checkpoint", required=True, help="model checkpoint")
    p.add_argument("--score_metric", choices=["hep", "dinov3_cosine"], default="hep",
                   help="CEM cost scorer (hep | dinov3_cosine)")
    p.add_argument("--hep_active_scales", default="",
                   help="HEP pyramid ablation: comma-separated grid sizes to keep, e.g. '14' or '2,7,14'. "
                        "Empty uses all trained scales.")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_cases", type=int, default=0, help="0 = full set; else evenly-spaced subset (smoke only)")
    p.add_argument("--shard", type=int, default=0, help="this shard index in [0,num_shards)")
    p.add_argument("--num_shards", type=int, default=1, help="split traj list across this many shards (merge after)")
    p.add_argument("--num_samples", type=int, default=128, help="CEM samples per iter (paper N=120)")
    p.add_argument("--topk", type=int, default=16, help="CEM elites")
    p.add_argument("--opt_steps", type=int, default=5, help="CEM iterations (paper found 1 effective)")
    p.add_argument("--prior_n", type=int, default=500, help="#GT trajs to estimate the action prior")
    p.add_argument("--sampler", default="anchor", choices=["gaussian", "anchor"],
                   help="CEM candidate parametrization. 'anchor' (default): mimic the trajectory-collection "
                        "sampler -- a cone target direction (theta,phi) around forward + smoothstep-slerp curve "
                        "+ per-step scale, Gaussian over 3 vars (a,b,step_scale); yaw follows the heading (the "
                        "curved, in-distribution prior the WM was trained on). 'gaussian': straight line = "
                        "constant per-step translation (sx,sy,sz) + terminal yaw, Gaussian over 4 vars (OOD).")
    p.add_argument("--theta_max_deg", type=float, default=85.0,
                   help="anchor sampler: max polar deviation of the cone from forward (collect default = 85)")
    p.add_argument("--seed", type=int, default=_CEM_RNG_ANCHOR, help="CEM sampling seed")
    p.add_argument("--deterministic", action="store_true",
                   help="fix all RNG seeds + cudnn.deterministic + use_deterministic_algorithms so a "
                        "given (config,ckpt,data,seed,cem-params) is bit-reproducible run-to-run.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--nonstrict_load", action="store_true", default=True)
    p.add_argument("--dump_raw", action="store_true", default=False,
                   help="deprecated compatibility flag; per-case output always stores the minimal "
                        "raw fields needed to recompute ATE/RPE.")
    return p.parse_args()


def load_member(cfg_path, ckpt_path, device, nonstrict):
    cfg = load_config(cfg_path)
    core = UncertaintyAwareWorldModel(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    core.load_state_dict(state.get("model", state), strict=not nonstrict)
    core.eval()
    normalizer = load_action_normalizer(cfg, state, device)
    return cfg, core, normalizer


def reconstruct_poses(gt_poses, action_seq, decision_idx, goal_idx, spacing):
    """Inverse of candidate_local_actions: integrate per-step local deltas back to world poses.
    Frames 0..decision_idx are GT; decision_idx+1..goal_idx come from action_seq (H,4)."""
    H = goal_idx - decision_idx
    ref_pos = gt_poses[decision_idx, :3]; ref_yaw = float(gt_poses[decision_idx, 5])
    local_pos = np.zeros((H + 1, 3)); local_yaw = np.zeros(H + 1)
    for t in range(H):
        local_pos[t + 1] = local_pos[t] + action_seq[t, :3] * spacing
        local_yaw[t + 1] = local_yaw[t] + action_seq[t, 3]
    R = yaw_rotmat(ref_yaw)                     # local = (p-ref) @ R  =>  p = local @ R^T + ref
    world_pos = local_pos @ R.T + ref_pos
    world_yaw = ref_yaw + local_yaw
    poses = np.zeros((goal_idx + 1, 6))
    poses[: decision_idx + 1] = gt_poses[: decision_idx + 1]
    poses[decision_idx + 1:, :3] = world_pos[1:]
    poses[decision_idx + 1:, 5] = world_yaw[1:]
    return poses


def theta_to_actions(theta, H):
    """theta (N,4)=(sx,sy,sz,phi) -> action seq (N,H,4): constant per-step translation + terminal yaw."""
    N = theta.shape[0]
    act = np.zeros((N, H, 4), dtype=np.float32)
    act[:, :, 0] = theta[:, 0:1]; act[:, :, 1] = theta[:, 1:2]; act[:, :, 2] = theta[:, 2:3]
    act[:, -1, 3] = theta[:, 3]
    return act


# --- anchor sampler (mimics the trajectory-collection sampler) -----------------------
def smoothstep(u):
    return 3.0 * u * u - 2.0 * u * u * u


def _slerp_from_forward(d_target, alpha):
    """Vectorized slerp(e_f=(1,0,0), d_target, alpha): d_target (N,3) unit, alpha scalar -> (N,3) unit."""
    ef = np.array([1.0, 0.0, 0.0])
    dot = np.clip(d_target @ ef, -1.0, 1.0)                       # (N,)
    omega = np.arccos(dot); so = np.sin(omega)
    small = so < 1e-8; sod = np.where(small, 1.0, so)
    w0 = np.sin((1.0 - alpha) * omega) / sod
    w1 = np.sin(alpha * omega) / sod
    out = w0[:, None] * ef[None, :] + w1[:, None] * d_target
    out[small] = ef
    return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)


def anchor_z_to_actions(z, H, theta_max):
    """Dataset-style candidate actions, mirroring wm_collect/trajectory_sampler.generate_trajectory:
    a CONE target direction (theta,phi) around forward + a smoothstep-eased SLERP curve (heading turns
    gradually, not a straight line) + a per-trajectory step scale. Yaw follows the movement heading
    (NOT an independent terminal yaw). Reproduces the exact distribution the world model was trained on.

    z (N,3) = (a, b, step_scale) -> theta = min(hypot(a,b), theta_max), phi = atan2(b,a). The (a,b) disk
    is a Gaussian-friendly, non-periodic chart of the cone (mu=(0,0)=forward). Local decision frame:
    forward=+x, right=+y, up=-z (AirSim NED). Translation is in `spacing` units (step_scale=1 == base_step,
    which equals the dataset spacing), so it matches the gaussian sampler's ~unit scale. Per-step jitter is
    omitted for the planned mean (it is dataset augmentation noise, not a controllable DOF)."""
    N = z.shape[0]
    a, b, s = z[:, 0], z[:, 1], z[:, 2]
    theta = np.minimum(np.hypot(a, b), theta_max)
    phi = np.arctan2(b, a)
    s = np.clip(s, 0.05, None)
    st = np.sin(theta)
    d_target = np.stack([np.cos(theta), st * np.cos(phi), -st * np.sin(phi)], axis=1)  # e_u=(0,0,-1)
    d_target /= np.clip(np.linalg.norm(d_target, axis=1, keepdims=True), 1e-12, None)
    act = np.zeros((N, H, 4), dtype=np.float32)
    prev_yaw = np.zeros(N)
    for t in range(H):
        u = 1.0 if H == 1 else t / float(H - 1)
        direction = _slerp_from_forward(d_target, smoothstep(u))        # (N,3) unit heading at step t
        act[:, t, :3] = s[:, None] * direction                         # per-step local translation (spacing units)
        yaw = np.arctan2(direction[:, 1], direction[:, 0])
        act[:, t, 3] = (yaw - prev_yaw + np.pi) % (2 * np.pi) - np.pi   # wrapped per-step yaw delta
        prev_yaw = yaw
    return act


def main():
    args = parse_args()
    if args.deterministic:
        import os, random
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"[deterministic] use_deterministic_algorithms warn_only failed: {e}")
        print("[deterministic] RNG fixed; cudnn.deterministic=True", flush=True)
    active_scales = [int(x) for x in args.hep_active_scales.split(",") if x.strip()] or None
    if active_scales is not None:
        print(f"[HEP ablation] active_scales={active_scales}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    members = []
    cfg0 = None
    member_specs = [(args.config, args.checkpoint, args.score_metric)]
    for cfg_path, ckpt_path, metric in member_specs:
        cfg, core, normalizer = load_member(cfg_path, ckpt_path, device, args.nonstrict_load)
        members.append((core, normalizer, metric))
        cfg0 = cfg0 or cfg
    ensemble = len(members) > 1

    ctx_len = int(cfg0["training"]["context_len"]); decision_idx = ctx_len - 1
    spacing = float(cfg0["data"].get("metric_waypoint_spacing", 1.0))
    image_size = int(cfg0["model"]["image_size"])
    transform = build_image_transform(image_size)
    amp_dtype = get_autocast_dtype(cfg0["training"].get("amp", "bf16"))

    all_dirs = list_traj_dirs(args.root)
    if args.max_cases and args.max_cases < len(all_dirs):
        idx = np.linspace(0, len(all_dirs) - 1, args.max_cases).round().astype(int)
        traj_dirs = [all_dirs[i] for i in idx]
    else:
        traj_dirs = all_dirs
    if args.num_shards > 1:
        traj_dirs = traj_dirs[args.shard :: args.num_shards]  # interleaved shards keep splits balanced
        print(f"[shard] {args.shard}/{args.num_shards}: {len(traj_dirs)} cases")

    # ---- dataset-level action prior (paper: per-dataset mu/sigma) from a GT sample (aggregate, not per-case) ----
    rng = np.random.default_rng(args.seed)
    prior_dirs = [all_dirs[i] for i in np.linspace(0, len(all_dirs) - 1, min(args.prior_n, len(all_dirs))).round().astype(int)]
    thetas = []
    for td in prior_dirs:
        gt, _, _ = load_traj_pkl(td)
        gi = gt.shape[0] - 1
        fut = candidate_local_actions(gt, gt, decision_idx, gi, spacing)[decision_idx:]  # (H,4) GT future
        thetas.append([fut[:, 0].mean(), fut[:, 1].mean(), fut[:, 2].mean(), fut[:, 3].sum()])
    thetas = np.array(thetas)
    mu_init = thetas.mean(0); sigma_init = thetas.std(0) + 1e-3

    # ---- choose candidate parametrization / sampler ----
    if args.sampler == "anchor":
        theta_max = math.radians(args.theta_max_deg)
        z_to_actions = lambda z, H: anchor_z_to_actions(z, H, theta_max)
        # (a, b, step_scale): forward (theta=0), broad cone spread, unit step scale (base_step==spacing)
        mu_init = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        sigma_init = np.array([theta_max * 0.5, theta_max * 0.5, 0.15], dtype=np.float64)
    else:
        z_to_actions = lambda z, H: theta_to_actions(z, H)

    def score_pool(ctx_imgs, goal_img, cand_t, H):
        """combined per-candidate cost (N,) — z-norm sum over members for the ensemble."""
        total = None
        for core, normalizer, metric in members:
            out = score_candidates(core, normalizer, ctx_imgs, goal_img, cand_t, H,
                                   amp_dtype, metric, repeats=1,
                                   hep_active_scales=active_scales)
            s = out["score"].float().cpu().numpy()
            if ensemble:
                s = (s - s.mean()) / (s.std() + 1e-9)
            total = s if total is None else total + s
        return total

    res_rows = []
    metric_rows = []
    pbar = tqdm(traj_dirs, desc=f"offline_planning[{args.score_metric}]", unit="case")
    for ci, td in enumerate(pbar):
        gt_poses, frame_stems, _ = load_traj_pkl(td)
        goal_idx = gt_poses.shape[0] - 1; H = goal_idx - decision_idx
        ctx_imgs = torch.stack([transform(Image.open(td / f"{frame_stems[i]}.jpg").convert("RGB"))
                                for i in range(ctx_len)], dim=0).to(device)
        goal_img = transform(Image.open(td / f"{frame_stems[goal_idx]}.jpg").convert("RGB")).to(device)
        # fixed GT context transitions (first decision_idx of the C+H-1 sliding-window actions)
        ctx_actions = candidate_local_actions(gt_poses, gt_poses, decision_idx, goal_idx, spacing)[:decision_idx]  # (C-1,4)

        mu = mu_init.copy(); sigma = sigma_init.copy(); nvars = mu.shape[0]
        for _ in range(args.opt_steps):
            theta = rng.normal(mu, sigma, size=(args.num_samples, nvars)).astype(np.float32)
            theta[0] = mu  # keep current mean as a candidate (elitism)
            fut = z_to_actions(theta, H)                                       # (N,H,4) future
            ctx_b = np.broadcast_to(ctx_actions[None], (args.num_samples, decision_idx, 4))
            full = np.concatenate([ctx_b, fut], axis=1).astype(np.float32)     # (N, C-1+H, 4)
            cand_t = torch.as_tensor(full, dtype=torch.float32, device=device)
            with torch.no_grad():
                cost = score_pool(ctx_imgs, goal_img, cand_t, H)
            elite = np.argsort(cost)[: args.topk]
            mu = theta[elite].mean(0); sigma = theta[elite].std(0) + 1e-4

        planned = reconstruct_poses(gt_poses, z_to_actions(mu[None], H)[0], decision_idx, goal_idx, spacing)
        metric_gt_traj = pose_traj_xyzyaw(gt_poses, decision_idx, goal_idx)
        metric_pred_traj = pose_traj_xyzyaw(planned, decision_idx, goal_idx)
        row = {
            "traj": td.name,
            "mu": [float(x) for x in mu],
            "decision_idx": int(decision_idx),
            "goal_idx": int(goal_idx),
            "spacing": float(spacing),
            "gt_traj": metric_gt_traj,
            "pred_traj": metric_pred_traj,
        }
        res_rows.append(row)
        metric_rows.append({
            "traj": td.name,
            "spacing": float(spacing),
            "gt_traj": metric_gt_traj,
            "pred_traj": metric_pred_traj,
        })

    aligned = eval_planning(metric_rows)["all"]
    res = {"ate": aligned["ate"], "rpe": aligned["rpe"]}
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    (out_dir / "per_case.json").write_text(json.dumps(res_rows))
    print(f"[offline_planning:{args.score_metric}] wrote {out_dir}/metrics.json ({len(res_rows)} cases)")


if __name__ == "__main__":
    main()
