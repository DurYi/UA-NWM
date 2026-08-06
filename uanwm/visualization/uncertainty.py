from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/uanwm_matplotlib")

from uanwm.datasets.transforms import build_image_transform
from uanwm.eval.action_norm import load_action_normalizer
from uanwm.eval.nav_utils import candidate_local_actions, load_traj_pkl
from uanwm.models import UncertaintyAwareWorldModel
from uanwm.utils.config import load_config
from uanwm.utils.tensor import get_autocast_dtype
from uanwm.visualization.raev2_decoder import load_raev2_decoder


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UA-NWM uncertainty-subspace visualization with the fine-tuned RAEv2 decoder."
    )
    parser.add_argument("--traj", required=True, help="Trajectory id (traj_000123) or a path to a trajectory folder.")
    parser.add_argument(
        "--mode",
        choices=["random", "goal_oriented", "both"],
        default="both",
        help="random samples from the predicted HEP subspace, goal-oriented pseudo-goal projection, or both.",
    )
    parser.add_argument(
        "--data_root",
        default="data/airgoal10k/test",
        help="AirGoal split root or a direct trajectory directory.",
    )
    parser.add_argument("--pseudo_goal", default="", help="Pseudo-goal image file or directory of images.")
    parser.add_argument("--outdir", default="uncertainty", help="Subfolder under ./visualization.")
    parser.add_argument("--config", default="configs/ua_nwm.yaml")
    parser.add_argument("--checkpoint", default="pretrained/ua-nwm/ua-nwm_best.pt")
    parser.add_argument(
        "--raev2_checkpoint",
        default="pretrained/raev2-dinov3b-k1-airgoal10k-ft/step_0010000.pt",
    )
    parser.add_argument("--candidate_idx", type=int, default=-1, help="-1 = GT rollout; otherwise test candidate index.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Random subspace sample radius.")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of random samples to render.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.7, help="Display gamma baked into saved RGB panels.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def resolve_traj_dir(data_root: str | Path, traj: str) -> Path:
    traj_path = Path(traj)
    if traj_path.exists():
        return traj_path
    root = Path(data_root)
    if root.name.startswith("traj_") and (root / "traj_data.pkl").is_file():
        return root
    candidates = [root / traj]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"trajectory not found: traj={traj!r}, data_root={str(root)!r}")


def poses6(record: dict[str, Any]) -> np.ndarray:
    point = np.asarray(record["point"], dtype=np.float64)
    roll = np.asarray(record.get("roll", np.zeros(len(point))), dtype=np.float64).reshape(-1)
    pitch = np.asarray(record.get("pitch", np.zeros(len(point))), dtype=np.float64).reshape(-1)
    yaw = np.asarray(record["yaw"], dtype=np.float64).reshape(-1)
    return np.column_stack([point[:, 0], point[:, 1], point[:, 2], roll, pitch, yaw])


def load_model(args: argparse.Namespace, device: torch.device):
    cfg = load_config(args.config)
    core = UncertaintyAwareWorldModel(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = core.load_state_dict(state.get("model", state), strict=False)
    missing = [key for key in missing if not key.startswith("encoder.")]
    unexpected = [key for key in unexpected if not key.startswith("encoder.")]
    if missing or unexpected:
        print(f"[load] non-strict checkpoint load: missing={missing[:8]} unexpected={unexpected[:8]}", flush=True)
    core.eval()
    normalizer = load_action_normalizer(cfg, state, device)
    transform = build_image_transform(int(cfg["model"]["image_size"]))
    amp_dtype = get_autocast_dtype(cfg["training"].get("amp", "bf16"))
    decoder = load_raev2_decoder(core, args.raev2_checkpoint, device)
    return cfg, core, normalizer, transform, amp_dtype, decoder


@torch.no_grad()
def rollout_case(core, normalizer, transform, cfg, traj_dir: Path, candidate_idx: int, device, amp_dtype):
    gt_poses, frame_stems, candidate_poses = load_traj_pkl(traj_dir)
    ctx_len = int(cfg["training"]["context_len"])
    decision_idx = ctx_len - 1
    goal_idx = gt_poses.shape[0] - 1
    spacing = float(cfg["data"].get("metric_waypoint_spacing", 1.0))
    if candidate_idx >= 0:
        if candidate_poses is None:
            raise ValueError("--candidate_idx requires candidate_trajectories in traj_data.pkl")
        if candidate_idx >= len(candidate_poses):
            raise IndexError(f"candidate_idx={candidate_idx} but only {len(candidate_poses)} candidates are available")
        rollout_poses = candidate_poses[candidate_idx]
        candidate_label = f"candidate_{candidate_idx:02d}"
    else:
        rollout_poses = gt_poses
        candidate_label = "gt"
    actions = candidate_local_actions(gt_poses, rollout_poses, decision_idx, goal_idx, spacing)
    context_images = torch.stack(
        [transform(Image.open(traj_dir / f"{frame_stems[i]}.jpg").convert("RGB")) for i in range(ctx_len)],
        dim=0,
    ).to(device)
    goal_pil = Image.open(traj_dir / f"{frame_stems[goal_idx]}.jpg").convert("RGB")
    goal_image = transform(goal_pil).to(device)
    with autocast_context(device, amp_dtype):
        context_raw = core.encode_images(context_images.unsqueeze(0))
        goal_feat = core.encode_images(goal_image.unsqueeze(0).unsqueeze(1))[0, 0].float()
        context_states = core.encode_state(context_raw)
        action_tensor = normalizer(torch.as_tensor(actions[None], dtype=torch.float32, device=device))
        pred_latent = core.rollout(context_states, action_tensor, steps=goal_idx - decision_idx)["pred_latents"][:, -1]
        hidden = core.fuser._hidden(pred_latent)
        mean = core.fuser.out_proj(hidden).float()
        context_summary = context_states.flatten(1, 2) if getattr(core.fuser.hep, "use_context", False) else None
    return {
        "gt_poses": gt_poses,
        "frame_stems": frame_stems,
        "decision_idx": decision_idx,
        "goal_idx": goal_idx,
        "spacing": spacing,
        "candidate_label": candidate_label,
        "context_images": context_images,
        "goal_pil": goal_pil,
        "goal_feat": goal_feat,
        "hidden": hidden,
        "mean": mean,
        "context_summary": context_summary,
    }


@torch.no_grad()
def decompose_hep(head, hidden: torch.Tensor, mean: torch.Tensor, goal: torch.Tensor, context_summary):
    dino_dim = head.dino_dim
    mu_n = F.normalize(mean, dim=-1).float()
    goal_n = F.normalize(goal, dim=-1).float()
    e = goal_n - mu_n
    residual = e.clone()
    for scale_index, scale in enumerate(head.scales):
        hidden_cells = head._to_cells(hidden, scale).mean(dim=2)
        if getattr(head, "use_context", False):
            if context_summary is None:
                raise RuntimeError("context-conditioned HEP requires context_summary")
            ctx = head._context_feature(scale_index, scale, hidden_cells, context_summary).to(hidden_cells.dtype)
            hidden_cells = torch.cat([hidden_cells, ctx], dim=-1)
        basis = head.nets[scale_index](hidden_cells).reshape(
            hidden.shape[0], scale * scale, head.rank, dino_dim
        ).float()
        residual_cells = head._to_cells(residual, scale)
        residual_bar = residual_cells.mean(dim=2)
        rhs = torch.einsum("bcrd,bcd->bcr", basis, residual_bar).float()
        gram = torch.einsum("bcrd,bcsd->bcrs", basis, basis).float()
        eye = torch.eye(head.rank, device=basis.device, dtype=gram.dtype).expand_as(gram)
        coords = torch.linalg.solve(gram + 1e-3 * eye, rhs.unsqueeze(-1)).squeeze(-1)
        projected = torch.einsum("bcrd,bcr->bcd", basis, coords)
        residual_cells = residual_cells - projected.unsqueeze(2)
        residual = head._from_cells(residual_cells, scale)
    e_perp = residual
    e_par = e - e_perp
    goal_norm = goal.norm(dim=-1, keepdim=True)
    feat_par = goal_norm * F.normalize(mu_n + e_par, dim=-1)
    e_norm = e.norm(dim=-1)
    epar_norm = e_par.norm(dim=-1)
    eperp_norm = e_perp.norm(dim=-1)
    frac = epar_norm.square() / e_norm.square().clamp_min(1e-9)
    return {
        "e": e_norm[0],
        "epar": epar_norm[0],
        "eperp": eperp_norm[0],
        "frac": frac[0],
        "feat_par": feat_par[0],
        "cost": e_perp.square().sum(-1)[0],
    }


@torch.no_grad()
def sample_hep_subspace(head, hidden, context_summary, mean, n_samples: int, sigma: float, seed: int, device):
    dino_dim = head.dino_dim
    batch, num_tokens, _ = hidden.shape
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    total_deviation = torch.zeros(n_samples, batch, num_tokens, dino_dim, device=device)
    for scale_index, scale in enumerate(head.scales):
        hidden_cells = head._to_cells(hidden, scale).mean(dim=2)
        if getattr(head, "use_context", False):
            if context_summary is None:
                raise RuntimeError("context-conditioned HEP requires context_summary")
            ctx = head._context_feature(scale_index, scale, hidden_cells, context_summary).to(hidden_cells.dtype)
            hidden_cells = torch.cat([hidden_cells, ctx], dim=-1)
        basis = head.nets[scale_index](hidden_cells).reshape(batch, scale * scale, head.rank, dino_dim).float()
        coords = torch.randn(
            n_samples, batch, scale * scale, head.rank, device=device, generator=generator
        )
        cell_deviation = torch.einsum("nbcr,bcrd->nbcd", coords, basis)
        cell_size = (head.base_grid // scale) ** 2
        dense_deviation = head._from_cells(
            cell_deviation.reshape(n_samples * batch, scale * scale, 1, dino_dim).expand(
                n_samples * batch, scale * scale, cell_size, dino_dim
            ),
            scale,
        )
        total_deviation = total_deviation + dense_deviation.reshape(n_samples, batch, num_tokens, dino_dim)
    return (mean.unsqueeze(0) + float(sigma) * total_deviation)[:, 0]


def feature_to_chw(feature: torch.Tensor, grid: int) -> torch.Tensor:
    if feature.dim() == 2:
        feature = feature.unsqueeze(0)
    return feature.reshape(feature.shape[0], grid, grid, feature.shape[-1]).permute(0, 3, 1, 2).float()


@torch.no_grad()
def decode_feature(decoder, feature: torch.Tensor, grid: int, gamma: float) -> np.ndarray:
    image = decoder(feature_to_chw(feature, grid))[0].permute(1, 2, 0).detach().cpu().numpy()
    image = np.clip(image, 0, 1)
    if gamma != 1.0:
        image = image ** float(gamma)
    return image


def save_image_array(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)


def heatmap_array(values: np.ndarray, grid: int, vmin: float, vmax: float, cmap: str) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normed = np.clip((values.reshape(grid, grid) - vmin) / (vmax - vmin + 1e-12), 0, 1)
    colors = plt.get_cmap(cmap)(normed)[..., :3]
    image = Image.fromarray((np.clip(colors, 0, 1) * 255).astype(np.uint8))
    image = image.resize((grid * 16, grid * 16), Image.Resampling.NEAREST)
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_heatmap(values: np.ndarray, grid: int, path: Path, vmin: float, vmax: float, cmap: str) -> np.ndarray:
    image = heatmap_array(values, grid, vmin, vmax, cmap)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)
    return image


def to_display_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()


def cosine_to_goal(feature: torch.Tensor, goal: torch.Tensor) -> float:
    return float(F.cosine_similarity(feature.float(), goal.float(), dim=-1).mean().detach().cpu())


def draw_trajectory(ax, poses: np.ndarray, decision_idx: int, goal_idx: int) -> None:
    xyz = poses[:, :3]
    yaw = poses[:, 5]
    ax.plot(xyz[:, 0], xyz[:, 1], color="0.55", linewidth=1.4)
    ax.scatter(xyz[:, 0], xyz[:, 1], c=np.arange(len(xyz)), cmap="viridis", s=18)
    for idx, color, label in ((decision_idx, "lime", "start"), (goal_idx, "red", "goal")):
        ax.scatter([xyz[idx, 0]], [xyz[idx, 1]], marker="*", s=180, c=color, edgecolors="k", linewidths=0.5, label=label)
        span = max(float(np.ptp(xyz[:, 0])), float(np.ptp(xyz[:, 1])), 1.0)
        ax.arrow(
            xyz[idx, 0],
            xyz[idx, 1],
            0.16 * span * np.cos(yaw[idx]),
            0.16 * span * np.sin(yaw[idx]),
            color=color,
            width=0.004 * span,
            head_width=0.04 * span,
            length_includes_head=True,
        )
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ax.set_title("GT trajectory XY", fontsize=8)
    ax.legend(fontsize=6)


def collect_pseudo_images(path: str | Path) -> list[Path]:
    if not path:
        return []
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"pseudo goal path not found: {p}")


@torch.no_grad()
def encode_pil(core, transform, image: Image.Image, device: torch.device, amp_dtype):
    tensor = transform(image.convert("RGB")).to(device).unsqueeze(0).unsqueeze(0)
    with autocast_context(device, amp_dtype):
        return core.encode_images(tensor)[0, 0].float()


def render_random_overview(out_dir: Path, title: str, refs: dict[str, np.ndarray], samples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_samples = len(samples)
    rows = 1 + int(np.ceil(max(n_samples, 1) / 4))
    fig = plt.figure(figsize=(18, 3.1 * rows))
    gs = fig.add_gridspec(rows, 4, hspace=0.34, wspace=0.08)
    fig.suptitle(title, fontsize=14)
    for idx, (name, image) in enumerate(refs.items()):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for idx, (image, label) in enumerate(samples):
        row = 1 + idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.savefig(out_dir / "overview_random.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_goal_overview(out_dir: Path, title: str, refs: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    ncols = 6
    fig = plt.figure(figsize=(21, 3.1 * (len(rows) + 1)))
    gs = fig.add_gridspec(len(rows) + 1, ncols, hspace=0.34, wspace=0.06)
    fig.suptitle(title, fontsize=14)
    for idx, (name, image) in enumerate(refs.items()):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for idx in range(len(refs), ncols):
        fig.add_subplot(gs[0, idx]).axis("off")
    for i, row in enumerate(rows, start=1):
        panels = [
            (row["target_label"], row["target"]),
            (f"{row['target_label']} DINO->RGB", row["target_dino"]),
            ("goal-by-S->RGB", row["goal_by_s"]),
            ("|e| heat", row["e_heat"]),
            ("|e_parallel| heat", row["epar_heat"]),
            ("|e_perp| heat", row["eperp_heat"]),
        ]
        for j, (label, image) in enumerate(panels):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(np.clip(image, 0, 1))
            ax.set_title(f"{row['name']}\n{label}", fontsize=8)
            ax.axis("off")
    fig.savefig(out_dir / "overview_goal_oriented.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg, core, normalizer, transform, amp_dtype, decoder = load_model(args, device)
    traj_dir = resolve_traj_dir(args.data_root, args.traj)
    case = rollout_case(core, normalizer, transform, cfg, traj_dir, args.candidate_idx, device, amp_dtype)

    head = core.fuser.hep
    if head is None:
        raise RuntimeError("checkpoint/config has no HEP module")
    grid = int(round(int(cfg["model"].get("raw_num_tokens", 196)) ** 0.5))

    out_dir = ROOT / "visualization" / args.outdir / traj_dir.name
    if args.mode == "goal_oriented":
        shutil.rmtree(out_dir / "random", ignore_errors=True)
        (out_dir / "overview_random.png").unlink(missing_ok=True)
    elif args.mode == "random":
        shutil.rmtree(out_dir / "goal_oriented", ignore_errors=True)
        (out_dir / "overview_goal_oriented.png").unlink(missing_ok=True)

    goal = case["goal_feat"].unsqueeze(0)
    mean = case["mean"]
    hidden = case["hidden"]
    ctx_sum = case["context_summary"]
    decomposition = decompose_hep(head, hidden, mean, goal, ctx_sum)

    decision_image = to_display_image(case["context_images"][case["decision_idx"]])
    goal_image = np.asarray(case["goal_pil"].resize((256, 256), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    goal_rgb = decode_feature(decoder, case["goal_feat"], grid, args.gamma)
    mean_rgb = decode_feature(decoder, mean[0], grid, args.gamma)
    goal_by_s_rgb = decode_feature(decoder, decomposition["feat_par"], grid, args.gamma)

    heat_grid = grid
    e_np = decomposition["e"].detach().cpu().numpy().reshape(heat_grid, heat_grid)
    epar_np = decomposition["epar"].detach().cpu().numpy().reshape(heat_grid, heat_grid)
    eperp_np = decomposition["eperp"].detach().cpu().numpy().reshape(heat_grid, heat_grid)
    frac_np = decomposition["frac"].detach().cpu().numpy().reshape(heat_grid, heat_grid)
    e_max = max(float(e_np.max()), 1e-6)

    refs = {
        "decision frame": decision_image,
        "goal image": goal_image,
        "goal DINO->RGB": goal_rgb,
        f"mean mu->RGB\ncos={cosine_to_goal(mean[0], case['goal_feat']):.3f}": mean_rgb,
    }

    random_records = []
    if args.mode in {"random", "both"}:
        random_dir = out_dir / "random"
        shutil.rmtree(random_dir, ignore_errors=True)
        random_dir.mkdir(parents=True, exist_ok=True)
        save_image_array(decision_image, random_dir / "00_decision_frame.png")
        save_image_array(goal_image, random_dir / "01_goal_image_resized.png")
        save_image_array(goal_rgb, random_dir / "02_goal_dino_rgb.png")
        save_image_array(mean_rgb, random_dir / "03_mean_mu_rgb.png")
        samples = sample_hep_subspace(head, hidden, ctx_sum, mean, args.n_samples, args.sigma, args.seed, device)
        sample_panels = []
        for idx, sample in enumerate(samples):
            image = decode_feature(decoder, sample, grid, args.gamma)
            cos_value = cosine_to_goal(sample, case["goal_feat"])
            save_image_array(image, random_dir / f"random_sample_{idx:02d}_rgb.png")
            sample_panels.append((image, f"sample {idx:02d}  cos={cos_value:.3f}"))
            random_records.append({"index": idx, "cos_to_goal": cos_value})
        render_random_overview(
            out_dir,
            f"{traj_dir.name}  {case['candidate_label']}  random HEP samples  sigma={args.sigma:g}",
            refs,
            sample_panels,
        )

    pseudo_records = []
    if args.mode in {"goal_oriented", "both"}:
        goal_oriented_dir = out_dir / "goal_oriented"
        goal_oriented_dir.mkdir(parents=True, exist_ok=True)
        save_image_array(decision_image, goal_oriented_dir / "ref__00_decision_frame.png")
        save_image_array(goal_image, goal_oriented_dir / "ref__01_goal_image_resized.png")
        save_image_array(goal_rgb, goal_oriented_dir / "ref__02_goal_dino_rgb.png")
        save_image_array(mean_rgb, goal_oriented_dir / "ref__03_mean_mu_rgb.png")
        save_image_array(goal_image, goal_oriented_dir / "true_goal__0_goal_image_resized.png")
        save_image_array(goal_rgb, goal_oriented_dir / "true_goal__1_goal_dino_rgb.png")
        save_image_array(goal_by_s_rgb, goal_oriented_dir / "true_goal__2_goal_by_S_rgb.png")
        true_e_heat = save_heatmap(
            e_np, heat_grid, goal_oriented_dir / "true_goal__3_e_total.png", 0.0, e_max, "magma"
        )
        true_epar_heat = save_heatmap(
            epar_np, heat_grid, goal_oriented_dir / "true_goal__4_e_parallel_explained.png", 0.0, e_max, "magma"
        )
        true_eperp_heat = save_heatmap(
            eperp_np, heat_grid, goal_oriented_dir / "true_goal__5_e_perp_unexplained.png", 0.0, e_max, "magma"
        )
        goal_rows = [
            {
                "name": "true goal",
                "target_label": "true goal",
                "target": goal_image,
                "target_dino": goal_rgb,
                "goal_by_s": goal_by_s_rgb,
                "e_heat": true_e_heat,
                "epar_heat": true_epar_heat,
                "eperp_heat": true_eperp_heat,
            }
        ]
        pseudo_paths = collect_pseudo_images(args.pseudo_goal)
        if not pseudo_paths:
            reason = "no --pseudo_goal provided" if not args.pseudo_goal else "no pseudo-goal images found"
            print(f"[visualization] {reason}; writing true-goal goal-oriented visualization only", flush=True)
        for pseudo_path in pseudo_paths:
            pseudo_pil = Image.open(pseudo_path).convert("RGB").resize(case["goal_pil"].size, Image.Resampling.BICUBIC)
            pseudo_feat = encode_pil(core, transform, pseudo_pil, device, amp_dtype)
            pseudo_decomp = decompose_hep(head, hidden, mean, pseudo_feat.unsqueeze(0), ctx_sum)
            pseudo_display = np.asarray(pseudo_pil.resize((256, 256), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            pseudo_dino_rgb = decode_feature(decoder, pseudo_feat, grid, args.gamma)
            pseudo_goal_by_s_rgb = decode_feature(decoder, pseudo_decomp["feat_par"], grid, args.gamma)
            pseudo_e = pseudo_decomp["e"].detach().cpu().numpy().reshape(heat_grid, heat_grid)
            pseudo_epar = pseudo_decomp["epar"].detach().cpu().numpy().reshape(heat_grid, heat_grid)
            pseudo_eperp = pseudo_decomp["eperp"].detach().cpu().numpy().reshape(heat_grid, heat_grid)
            pseudo_e_max = max(float(pseudo_e.max()), 1e-6)
            pseudo_e_heat = save_heatmap(
                pseudo_e, heat_grid, goal_oriented_dir / f"{pseudo_path.stem}__3_e_total.png", 0.0, pseudo_e_max, "magma"
            )
            pseudo_epar_heat = save_heatmap(
                pseudo_epar,
                heat_grid,
                goal_oriented_dir / f"{pseudo_path.stem}__4_e_parallel_explained.png",
                0.0,
                pseudo_e_max,
                "magma",
            )
            pseudo_eperp_heat = save_heatmap(
                pseudo_eperp,
                heat_grid,
                goal_oriented_dir / f"{pseudo_path.stem}__5_e_perp_unexplained.png",
                0.0,
                pseudo_e_max,
                "magma",
            )
            save_image_array(pseudo_display, goal_oriented_dir / f"{pseudo_path.stem}__0_pseudo_goal_resized.png")
            save_image_array(pseudo_dino_rgb, goal_oriented_dir / f"{pseudo_path.stem}__1_pseudo_goal_dino_rgb.png")
            save_image_array(pseudo_goal_by_s_rgb, goal_oriented_dir / f"{pseudo_path.stem}__2_pseudo_goal_by_S_rgb.png")
            record = {
                "name": pseudo_path.name,
                "path": str(pseudo_path),
                "cos_mean": cosine_to_goal(mean[0], pseudo_feat),
                "cos_goalbyS": cosine_to_goal(pseudo_decomp["feat_par"], pseudo_feat),
                "explained_fraction": float(pseudo_decomp["frac"].mean().detach().cpu()),
                "orthogonal_cost": float(pseudo_decomp["cost"].mean().detach().cpu()),
            }
            pseudo_records.append(record)
            goal_rows.append(
                {
                    "name": pseudo_path.name,
                    "target_label": "pseudo goal",
                    "target": pseudo_display,
                    "target_dino": pseudo_dino_rgb,
                    "goal_by_s": pseudo_goal_by_s_rgb,
                    "e_heat": pseudo_e_heat,
                    "epar_heat": pseudo_epar_heat,
                    "eperp_heat": pseudo_eperp_heat,
                }
            )
        render_goal_overview(
            out_dir,
            f"{traj_dir.name}  {case['candidate_label']}  goal-oriented S projections",
            refs,
            goal_rows,
        )
        (goal_oriented_dir / "meta.json").write_text(json.dumps({"pseudo_goals": pseudo_records}, indent=2))

    meta = {
        "traj": traj_dir.name,
        "traj_dir": str(traj_dir),
        "mode": args.mode,
        "candidate_label": case["candidate_label"],
        "decision_idx": int(case["decision_idx"]),
        "goal_idx": int(case["goal_idx"]),
        "spacing": float(case["spacing"]),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "raev2_checkpoint": str(args.raev2_checkpoint),
        "sigma": float(args.sigma),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "gamma": float(args.gamma),
        "cos_mean": cosine_to_goal(mean[0], case["goal_feat"]),
        "cos_goalbyS": cosine_to_goal(decomposition["feat_par"], case["goal_feat"]),
        "explained_fraction": float(decomposition["frac"].mean().detach().cpu()),
        "orthogonal_cost": float(decomposition["cost"].mean().detach().cpu()),
        "random_samples": random_records,
        "pseudo_goals": pseudo_records,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[visualization] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
