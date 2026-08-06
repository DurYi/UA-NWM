#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UANWM_REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export UAV_POLICY_API_KEY='Your-API-Key-Here'
PYTORCH_ENABLE_MPS_FALLBACK=1

export PYTHONPATH="$REPO_ROOT/src:$UANWM_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UANWM_ROOT="${UANWM_ROOT:-$UANWM_REPO_ROOT}"
export UANWM_ENCODER_NAME="${UANWM_ENCODER_NAME:-$UANWM_ROOT/pretrained/dinov3-vitb16}"
export UANWM_DEVICE="${UANWM_DEVICE:-cpu}"
export UANWM_INPUT_COLOR_SPACE="${UANWM_INPUT_COLOR_SPACE:-rgb}"
export UANWM_MODEL_FRAME="${UANWM_MODEL_FRAME:-body_frd}"

export UANWM_CEM_HORIZON="${UANWM_CEM_HORIZON:-8}"
export UANWM_CEM_CANDIDATES="${UANWM_CEM_CANDIDATES:-32}"
export UANWM_CEM_ITERATIONS="${UANWM_CEM_ITERATIONS:-3}"
export UANWM_CEM_ELITE_RATIO="${UANWM_CEM_ELITE_RATIO:-0.1}"
export UANWM_CEM_SCORE_BATCH_SIZE="${UANWM_CEM_SCORE_BATCH_SIZE:-4}"
export UANWM_CEM_ACTION_MEAN="${UANWM_CEM_ACTION_MEAN:-8.0,0.0,0.0,0.0}"
export UANWM_CEM_ACTION_STD="${UANWM_CEM_ACTION_STD:-1.5,1.5,0.0,0.60}"
export UANWM_CEM_ACTION_LOW="${UANWM_CEM_ACTION_LOW:-0.0,-3.0,0.0,-0.785398}"
export UANWM_CEM_ACTION_HIGH="${UANWM_CEM_ACTION_HIGH:-15.0,3.0,0.0,0.785398}"
export UANWM_CEM_GOAL_HINT_MODE="${UANWM_CEM_GOAL_HINT_MODE:-none}"

export UANWM_TRANSLATION_SCALE="${UANWM_TRANSLATION_SCALE:-1.0}"
export UANWM_YAW_SCALE="${UANWM_YAW_SCALE:-1.0}"

export GOAL_REACHED_METHOD="${GOAL_REACHED_METHOD:-lpips}"
export GOAL_REACHED_THRESHOLD="${GOAL_REACHED_THRESHOLD:-0.5}"
export GOAL_REACHED_CONSECUTIVE_HITS="${GOAL_REACHED_CONSECUTIVE_HITS:-1}"
export GOAL_REACHED_MIN_STEPS="${GOAL_REACHED_MIN_STEPS:-2}"
export GOAL_REACHED_LPIPS_DEVICE="${GOAL_REACHED_LPIPS_DEVICE:-cpu}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  if [[ "$UANWM_DEVICE" == cuda* ]]; then
    echo "macOS detected; overriding UANWM_DEVICE=$UANWM_DEVICE to cpu" >&2
    export UANWM_DEVICE=cpu
  fi
  if [[ "$GOAL_REACHED_LPIPS_DEVICE" == cuda* ]]; then
    echo "macOS detected; overriding GOAL_REACHED_LPIPS_DEVICE=$GOAL_REACHED_LPIPS_DEVICE to cpu" >&2
    export GOAL_REACHED_LPIPS_DEVICE=cpu
  fi
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GOAL_RGB="${GOAL_RGB:-$REPO_ROOT/goals/8.jpg}"
RECORD_OBSERVATIONS_DIR="${RECORD_OBSERVATIONS_DIR:-$REPO_ROOT/records/real_flight}"
RECORD_IMAGE_FORMAT="${RECORD_IMAGE_FORMAT:-png}"
RECORD_QUEUE_SIZE="${RECORD_QUEUE_SIZE:-64}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/python or fix the uanwm environment." >&2
  exit 1
fi

if [[ ! -d "$UANWM_ROOT" ]]; then
  echo "UANWM_ROOT does not exist: $UANWM_ROOT" >&2
  exit 1
fi

if [[ ! -d "$UANWM_ENCODER_NAME" ]]; then
  echo "UANWM_ENCODER_NAME does not exist: $UANWM_ENCODER_NAME" >&2
  exit 1
fi

if [[ ! -f "$GOAL_RGB" ]]; then
  echo "Goal image does not exist: $GOAL_RGB" >&2
  exit 1
fi

mkdir -p "$RECORD_OBSERVATIONS_DIR"
cd "$REPO_ROOT"

exec "$PYTHON_BIN" scripts/serve_policy.py \
  --host "$HOST" --port "$PORT" \
  --goal-rgb "$GOAL_RGB" \
  --record-observations-dir "$RECORD_OBSERVATIONS_DIR" \
  --record-image-format "$RECORD_IMAGE_FORMAT" \
  --record-queue-size "$RECORD_QUEUE_SIZE" \
  --policy-factory uav_openpi.ua_nwm_policy:create_goal_aware_policy_from_env
