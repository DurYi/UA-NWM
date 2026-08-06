"""Compute ATE / RPE for each strategy by recomputing from the RAW per-case trajectories
dumped by the evaluators (--dump_raw), under ONE metric definition so every strategy is scored
by the identical code path. Handles both settings, auto-detected from the raw schema:

  offline_ranking  (ranking_eval.py):  cand_traj=[K,L,4] + scores=[K]  -> select argmin(scores[:N])
                                        at N in --settings (8/16/32).
  offline_planning (planning_eval.py): pred_traj=[L,4]                 -> one planned trajectory.
  (both also carry: traj, spacing, gt_traj=[L,4] with [x,y,z,yaw]).

Metrics (aligned across strategies): drop the decision/origin frame (future waypoints only),
normalize positions by metric_waypoint_spacing, keep 3D.
  ATE = RMSE||p_pred - p_gt||,  RPE = RMSE||dp_pred - dp_gt||.

Usage:
  python -m uanwm.eval.validate_from_raw --settings 8 16 32 \
    --members UA-NWM=results/airgoal10k/offline_ranking/UA-NWM/per_case.json \
              UA-NWM_deterministic=results/airgoal10k/offline_ranking/UA-NWM_deterministic/per_case.json \
    --baseline UA-NWM_deterministic --out results/airgoal10k/offline_ranking/metrics.json
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def ate_rpe(pred, gt, spacing):
    """pred, gt: [L,4]=[x,y,z,yaw] over decision..goal (frame 0 = shared decision pose).
    Exclude frame 0 (future waypoints only), normalize positions by metric_waypoint_spacing,
    keep 3D. Returns (ATE, RPE)."""
    sp = float(spacing) if spacing else 1.0
    p = np.asarray(pred, dtype=np.float64)[1:, :3] / sp
    g = np.asarray(gt, dtype=np.float64)[1:, :3] / sp
    ate = float(np.sqrt(np.mean(np.sum((p - g) ** 2, axis=1)))) if len(p) else 0.0
    dp, dg = np.diff(p, axis=0), np.diff(g, axis=0)
    rpe = float(np.sqrt(np.mean(np.sum((dp - dg) ** 2, axis=1)))) if len(dp) else 0.0
    return ate, rpe


def _tie_rng(seed, traj, N):
    """stable per-(case,setting) RNG so exact-score ties break reproducibly across runs."""
    h = int(hashlib.md5(f"{seed}|{traj}|{N}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h)


def select_min_score(scores, seed, traj, setting):
    """Select the lowest-score candidate with the shared deterministic tie-break."""
    scores = np.asarray(scores, dtype=np.float64)
    mins = np.flatnonzero(scores == scores.min())
    return int(_tie_rng(seed, traj, setting).choice(mins)) if mins.size > 1 else int(mins[0])


def eval_ranking(recs, settings, seed):
    """Selection: for every case pick the argmin-score candidate among the first N and score it."""
    acc = {N: {"ate": [], "rpe": []} for N in settings}
    for r in recs:
        gt = r["gt_traj"]; cand = r["cand_traj"]
        scores = np.asarray(r["scores"], dtype=np.float64); spacing = r.get("spacing", 1.0)
        for N in settings:
            sel = select_min_score(scores[:N], seed, r.get("traj"), N)
            a, rp = ate_rpe(cand[sel], gt, spacing)
            acc[N]["ate"].append(a); acc[N]["rpe"].append(rp)
    return {str(N): {"ate": float(np.mean(v["ate"])), "rpe": float(np.mean(v["rpe"])),
                     "n": len(v["ate"])} for N, v in acc.items()}


def eval_planning(recs):
    """Single planned trajectory: score pred_traj against gt_traj directly."""
    ate_l, rpe_l = [], []
    for r in recs:
        a, rp = ate_rpe(r["pred_traj"], r["gt_traj"], r.get("spacing", 1.0))
        ate_l.append(a); rpe_l.append(rp)
    return {"all": {"ate": float(np.mean(ate_l)), "rpe": float(np.mean(rpe_l)), "n": len(ate_l)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="name=raw_per_case.json ...")
    ap.add_argument("--baseline", default="UA-NWM_deterministic")
    ap.add_argument("--settings", type=int, nargs="*", default=[8, 16, 32], help="offline_ranking only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    specs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.members]
    per_member, planning = {}, False
    for name, path in specs:
        recs = json.loads(Path(path).read_text())
        if not recs:
            raise SystemExit(f"{name}: empty {path}")
        r0 = recs[0]
        if "gt_traj" not in r0:
            raise SystemExit(f"{name}: {path} lacks raw gt_traj — re-run eval with --dump_raw.")
        if "pred_traj" in r0:      # offline_planning (single planned trajectory)
            planning = True; per_member[name] = eval_planning(recs)
        elif "cand_traj" in r0:    # offline_ranking (candidate selection)
            per_member[name] = eval_ranking(recs, args.settings, args.seed)
        else:
            raise SystemExit(f"{name}: {path} has neither pred_traj nor cand_traj.")

    setting = "offline_planning" if planning else "offline_ranking"
    keys = ["all"] if planning else [str(N) for N in args.settings]
    result = {"setting": setting, "metric": "ATE / RPE (recomputed from raw)", "strategies": per_member}
    for k in keys:
        base = per_member.get(args.baseline, {}).get(k)
        n = next(iter(per_member.values()))[k]["n"]
        head = "offline_planning" if planning else f"offline_ranking  {k}-choose-1"
        print(f"\n===== {head}   n={n} =====")
        print(f"{'strategy':24s} {'ATE':>9} {'RPE':>9}   dATE")
        for name, m in per_member.items():
            d = f"{m[k]['ate']-base['ate']:+.3f}" if base else ""
            print(f"{name:24s} {m[k]['ate']:9.3f} {m[k]['rpe']:9.3f}   {d}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[validate_from_raw] wrote {args.out}")


if __name__ == "__main__":
    main()
