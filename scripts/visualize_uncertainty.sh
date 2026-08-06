#!/usr/bin/env bash
# UA-NWM uncertainty visualization.
#
# Usage:
#   TRAJ=traj_000000 GPU=0 bash scripts/visualize_uncertainty.sh
#   TRAJ=traj_000000 PSEUDO_GOAL=/path/to/pseudo_or_dir MODE=both bash scripts/visualize_uncertainty.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GPU="${GPU:-0}"
TRAJ="${TRAJ:-traj_000000}"
MODE="${MODE:-random}"
DATA_ROOT="${DATA_ROOT:-data/airgoal10k/test}"
PSEUDO_GOAL="${PSEUDO_GOAL:-}"
OUTDIR="${OUTDIR:-uncertainty}"

ARGS=(
  --traj "$TRAJ"
  --mode "$MODE"
  --data_root "$DATA_ROOT"
  --outdir "$OUTDIR"
)

if [[ -n "$PSEUDO_GOAL" ]]; then
  ARGS+=(--pseudo_goal "$PSEUDO_GOAL")
fi

CUDA_VISIBLE_DEVICES="$GPU" python -m uanwm.visualization.uncertainty "${ARGS[@]}"
