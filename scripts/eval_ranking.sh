#!/usr/bin/env bash
# =============================================================================
#  UA-NWM  —  offline_ranking inference + evaluation on the airgoal10k test set
# =============================================================================
#  Rolls out the pretrained model over the 32 fixed candidates per case and scores candidate
#  selection with BOTH strategies from ONE forward pass (one checkpoint, two scorers):
#     UA-NWM               -> HEP scorer   (hep)
#     UA-NWM_deterministic -> Stage 1+2 mean-predictor cosine     (dinov3_cosine)
#  Then computes ATE / RPE at 8/16/32-choose-1.
#
#  Usage: GPU=0 bash scripts/eval_ranking.sh [--split test|test_unseen]
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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
  echo "Usage: bash scripts/eval_ranking.sh [--split test|test_unseen]"
  exit 0
fi
if [[ $# != 0 ]]; then
  if [[ $# != 2 || $1 != --split || (${2:-} != test && ${2:-} != test_unseen) ]]; then
    echo "Usage: bash scripts/eval_ranking.sh [--split test|test_unseen]" >&2
    exit 2
  fi
  split=$2
  if [[ ( $root_name == test || $root_name == test_unseen ) && $root_name != "$split" ]]; then
    echo "ROOT points to $root_name but --split selects $split." >&2
    exit 2
  fi
fi

ROOT="${ROOT:-data/airgoal10k/$split}"
OUT="${OUT:-results/airgoal10k/$split/offline_ranking}"
if ! compgen -G "$ROOT/traj_*/traj_data.pkl" >/dev/null; then
  echo "No trajectory folders found in $ROOT. Place the split there or set ROOT." >&2
  exit 1
fi

echo "[offline_ranking] one rollout, two scorers  (GPU $GPU)  ->  $OUT/{UA-NWM,UA-NWM_deterministic}"
CUDA_VISIBLE_DEVICES="$GPU" python uanwm/eval/ranking_eval.py \
  --config "$CFG" --checkpoint "$CKPT" --root "$ROOT" \
  --score_metric hep --dump_deterministic --dump_raw \
  --output_dir "$OUT/UA-NWM"
# ^ produces $OUT/UA-NWM (hep) and $OUT/UA-NWM_deterministic (dinov3_cosine)

bash "$REPO/scripts/compute_metrics.sh" "$OUT"
