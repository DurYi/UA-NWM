#!/usr/bin/env bash
# =============================================================================
#  UA-NWM  —  offline_planning inference + evaluation on the airgoal10k test set
# =============================================================================
#  CEM planning (MPC): optimize a continuous action sequence so the world model's predicted
#  final features match the goal image, then score the planned trajectory. The "32x3" config =
#  --num_samples 32 --topk 16 --opt_steps 3 --sampler anchor.  Each scorer drives its own CEM
#  optimization, so the two strategies are two separate MPC runs (no shared rollout here):
#     UA-NWM               -> hep    (HEP module)
#     UA-NWM_deterministic -> dinov3_cosine  (Stage 1+2 mean predictor)
#  Then computes ATE / RPE.
#
#  CEM sampling is stochastic, so results are made BIT-REPRODUCIBLE with --deterministic
#  (fixed cudnn/algorithms + fixed RNG) + fixed thread counts.
#
#  Usage:  GPU=0 bash scripts/eval_planning.sh
#  Output: results/airgoal10k/offline_planning/{UA-NWM,UA-NWM_deterministic}/per_case.json + metrics.json
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
GPU="${GPU:-0}"

CFG="${CFG:-configs/ua_nwm.yaml}"
CKPT="${CKPT:-pretrained/ua-nwm/ua-nwm_best.pt}"
ROOT="${ROOT:-data/airgoal10k/test}"
OUT="${OUT:-results/airgoal10k/offline_planning}"
CEM="--sampler anchor --num_samples 32 --topk 16 --opt_steps 3 --deterministic"

run_one () {   # <strategy_name> <score_metric>
  echo "[offline_planning] CEM 32x3 deterministic | $1 ($2)  (GPU $GPU) -> $OUT/$1"
  CUDA_VISIBLE_DEVICES="$GPU" python uanwm/eval/planning_eval.py \
    --config "$CFG" --checkpoint "$CKPT" --root "$ROOT" \
    --score_metric "$2" $CEM --dump_raw --output_dir "$OUT/$1"
}

run_one UA-NWM                hep
run_one UA-NWM_deterministic  dinov3_cosine

bash "$REPO/scripts/compute_metrics.sh" "$OUT"
