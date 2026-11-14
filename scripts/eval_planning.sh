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
#  Usage: GPU=0 bash scripts/eval_planning.sh [--split test|test_unseen]
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
split=test
root_path=${ROOT:-}
root_path=${root_path%/}
root_name=${root_path##*/}
if [[ $root_name == test_unseen ]]; then
  split=test_unseen
fi
if [[ ${1:-} == --help && $# == 1 ]]; then
  echo "Usage: bash scripts/eval_planning.sh [--split test|test_unseen]"
  exit 0
fi
if [[ $# != 0 ]]; then
  if [[ $# != 2 || $1 != --split || (${2:-} != test && ${2:-} != test_unseen) ]]; then
    echo "Usage: bash scripts/eval_planning.sh [--split test|test_unseen]" >&2
    exit 2
  fi
  split=$2
  if [[ ( $root_name == test || $root_name == test_unseen ) && $root_name != "$split" ]]; then
    echo "ROOT points to $root_name but --split selects $split." >&2
    exit 2
  fi
fi

ROOT="${ROOT:-data/airgoal10k/$split}"
OUT="${OUT:-results/airgoal10k/$split/offline_planning}"
if ! compgen -G "$ROOT/traj_*/traj_data.pkl" >/dev/null; then
  echo "No trajectory folders found in $ROOT. Place the split there or set ROOT." >&2
  exit 1
fi
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
