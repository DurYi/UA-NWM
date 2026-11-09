#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ $# -lt 1 || ! -f "$1" ]]; then
  echo "Usage: bash deploy/jetson/run_server.sh /absolute/path/to/goal.jpg [serve_policy options]" >&2
  exit 2
fi
JETSON_GOAL="$(realpath -- "$1")"
shift
export UANWM_CEM_HORIZON="${UANWM_CEM_HORIZON:-8}"
export UANWM_CEM_CANDIDATES="${UANWM_CEM_CANDIDATES:-32}"
export UANWM_CEM_ITERATIONS="${UANWM_CEM_ITERATIONS:-3}"
export UANWM_CEM_SCORE_BATCH_SIZE="${UANWM_CEM_SCORE_BATCH_SIZE:-4}"
export UANWM_CEM_ACTION_MEAN="${UANWM_CEM_ACTION_MEAN:-8.0,0.0,0.0,0.0}"
export UANWM_CEM_ACTION_STD="${UANWM_CEM_ACTION_STD:-1.5,1.5,0.0,0.60}"
export UANWM_CEM_ACTION_LOW="${UANWM_CEM_ACTION_LOW:-0.0,-3.0,0.0,-0.785398}"
export UANWM_CEM_ACTION_HIGH="${UANWM_CEM_ACTION_HIGH:-15.0,3.0,0.0,0.785398}"
export GOAL_REACHED_METHOD="${GOAL_REACHED_METHOD:-lpips}"
export GOAL_REACHED_THRESHOLD="${GOAL_REACHED_THRESHOLD:-0.5}"
export GOAL_REACHED_CONSECUTIVE_HITS="${GOAL_REACHED_CONSECUTIVE_HITS:-1}"
export GOAL_REACHED_MIN_STEPS="${GOAL_REACHED_MIN_STEPS:-2}"
export GOAL_REACHED_LPIPS_DEVICE="${GOAL_REACHED_LPIPS_DEVICE:-cpu}"
cd "$JETSON_REPO_ROOT"
"$JETSON_REPO_ROOT/.venv-jetson/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback"'
# Uses caller's UAV_POLICY_API_KEY without printing or overwriting it.
# Local test endpoint by default; this script never starts a flight client.
exec "$JETSON_REPO_ROOT/.venv-jetson/bin/python" -u deploy/scripts/serve_policy.py \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8001}" \
  --goal-rgb "$JETSON_GOAL" \
  --policy-factory uav_openpi.ua_nwm_policy:create_goal_aware_policy_from_env "$@"
