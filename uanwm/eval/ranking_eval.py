"""offline_ranking — candidate-selection eval on the airgoal10k test set.

For each case there are **32 candidate trajectories** (each = the context prefix frames 0..3 +
8 future waypoints, goal = frame 11). The world model rolls out each candidate's action sequence
and scores its predicted final features against the GOAL image (GT frame 11); the lowest-cost
candidate among the first N is selected (N = 8/16/32).

  * decision_idx = 3, goal_idx = 11; candidates are the 32 proposals only (GT is the reference).
  * "correct" candidate = the one whose ENDPOINT is closest to the GT endpoint. SR = (selected==correct).
  * ATE / RPE = between the SELECTED candidate's future path and the GT future path.
  * With --dump_deterministic, the SAME rollout is also scored by the mean-predictor cosine, so
    UA-NWM (HEP module) and UA-NWM_deterministic are produced in one pass.

Usage (see scripts/eval_ranking.sh):
  python uanwm/eval/ranking_eval.py --config configs/ua_nwm.yaml \
    --checkpoint pretrained/ua-nwm/ua-nwm_best.pt --root data/airgoal10k/test \
    --score_metric hep --dump_deterministic --dump_raw \
    --output_dir results/airgoal10k/offline_ranking/UA-NWM
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uanwm.datasets.transforms import build_image_transform
from uanwm.models import UncertaintyAwareWorldModel
from uanwm.utils.config import load_config
from uanwm.utils.tensor import get_autocast_dtype
from uanwm.eval.action_norm import load_action_normalizer
from uanwm.eval.nav_utils import candidate_local_actions, list_traj_dirs, pose_traj_xyzyaw, load_traj_pkl
from uanwm.eval.scoring import score_candidates
from uanwm.eval.validate_from_raw import eval_ranking, select_min_score

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "airgoal10k" / "test"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help="test split root containing traj_*")
    p.add_argument("--score_metric", choices=["hep", "dinov3_cosine"], default="dinov3_cosine")
    p.add_argument("--hep_active_scales", default="",
                   help="HEP pyramid ablation: comma-separated grid sizes to keep, e.g. '14' or '2,7,14'. "
                        "Empty uses all trained scales.")
    p.add_argument("--dump_deterministic", action="store_true", default=False,
                   help="ALSO select+dump 'UA-NWM_deterministic' (Stage 1+2 mean predictor, dinov3_cosine) "
                        "from the SAME forward pass, to <output_dir>_deterministic. One rollout, two scorers.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_cases", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="random tie-break seed")
    p.add_argument("--nonstrict_load", action="store_true", default=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dump_raw", action="store_true", default=False,
                   help="deprecated compatibility flag; per-case output always stores the minimal "
                        "raw fields needed to recompute ATE/RPE for any 8/16/32 selection.")
    return p.parse_args()


def main():
    args = parse_args()
    active_scales = [int(x) for x in args.hep_active_scales.split(",") if x.strip()] or None
    if active_scales is not None:
        print(f"[HEP ablation] active_scales={active_scales}")
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ctx_len = int(cfg["training"]["context_len"])
    decision_idx = ctx_len - 1
    spacing = float(cfg["data"].get("metric_waypoint_spacing", 1.0))
    image_size = int(cfg["model"]["image_size"])
    transform = build_image_transform(image_size)
    amp_dtype = get_autocast_dtype(cfg["training"].get("amp", "bf16"))

    core = UncertaintyAwareWorldModel(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = core.load_state_dict(state.get("model", state), strict=not args.nonstrict_load)
    if missing or unexpected:
        print(f"[load] missing={missing[:4]} unexpected={unexpected[:4]}")
    core.eval()
    normalizer = load_action_normalizer(cfg, state, device)

    traj_dirs = list_traj_dirs(args.root)
    if args.max_cases:
        traj_dirs = traj_dirs[: args.max_cases]

    # ONE forward pass per case (score_candidates), scored by up to two metrics off the SAME rollout:
    #   suffix ""              -> primary (args.score_metric, e.g. hep)  -> <output_dir>
    #   suffix "_deterministic"-> Stage 1+2 MEAN predictor cosine (out["dino_cos"]) -> <output_dir>_deterministic
    scorers = [("", "score", args.score_metric)]
    if args.dump_deterministic:
        scorers.append(("_deterministic", "dino_cos", "dinov3_cosine"))
    ACC = {suf: {"n": 0, "per_case": [], "metric_cases": []}
           for suf, _, _ in scorers}
    pbar = tqdm(traj_dirs, desc=f"offline_ranking[{args.score_metric}]", unit="case")
    for td in pbar:
        gt_poses, frame_stems, cand_pose_list = load_traj_pkl(td)   # gt_poses (F,6); candidates (32) from traj_data.pkl
        F = gt_poses.shape[0]
        goal_idx = F - 1
        eval_steps = goal_idx - decision_idx

        actions = np.stack([candidate_local_actions(gt_poses, cp, decision_idx, goal_idx, spacing)
                            for cp in cand_pose_list], axis=0)  # (K, goal_idx, 4)
        K = actions.shape[0]

        ctx_imgs = torch.stack([transform(Image.open(td / f"{frame_stems[i]}.jpg").convert("RGB"))
                                for i in range(ctx_len)], dim=0).to(device)
        goal_img = transform(Image.open(td / f"{frame_stems[goal_idx]}.jpg").convert("RGB")).to(device)
        cand_t = torch.as_tensor(actions, dtype=torch.float32, device=device)

        # ---- ONE forward pass (rollout of the 32 fixed candidates) ----
        out = score_candidates(core, normalizer, ctx_imgs, goal_img, cand_t, eval_steps,
                               amp_dtype, args.score_metric, repeats=1,
                               hep_active_scales=active_scales)

        metric_gt_traj = pose_traj_xyzyaw(gt_poses, decision_idx, goal_idx)
        metric_cand_traj = [pose_traj_xyzyaw(cp, decision_idx, goal_idx) for cp in cand_pose_list]
        # ---- select + record for EACH scorer (cheap; reuses the same rollout) ----
        for suf, key, _ in scorers:
            acc = ACC[suf]
            score = out[key].float().cpu().numpy()              # (K,), lower=better
            sel = select_min_score(score, args.seed, td.name, K)
            acc["n"] += 1
            rec = {
                "traj": td.name,
                "sel": sel,
                "decision_idx": int(decision_idx),
                "goal_idx": int(goal_idx),
                "spacing": float(spacing),
                "gt_traj": metric_gt_traj,
                "cand_traj": metric_cand_traj,
                "scores": [float(x) for x in score],
            }
            acc["per_case"].append(rec)
            acc["metric_cases"].append({
                "traj": td.name,
                "spacing": float(spacing),
                "gt_traj": metric_gt_traj,
                "cand_traj": metric_cand_traj,
                "scores": [float(x) for x in score],
            })

    for suf, key, metric_name in scorers:
        acc = ACC[suf]; n = acc["n"]
        res = eval_ranking(acc["metric_cases"], [8, 16, 32], args.seed)
        out_dir = Path(args.output_dir + suf); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
        (out_dir / "per_case.json").write_text(json.dumps(acc["per_case"]))
        print(f"[offline_ranking:{metric_name}] wrote {out_dir}/metrics.json ({n} cases)")


if __name__ == "__main__":
    main()
