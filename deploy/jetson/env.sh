#!/usr/bin/env bash
# Source this file, or use run_benchmark.sh / run_server.sh.
JETSON_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$JETSON_REPO_ROOT/deploy/src:$JETSON_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$JETSON_REPO_ROOT/deploy/jetson/runtime/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$JETSON_REPO_ROOT/deploy/jetson/runtime/torch"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export UANWM_ROOT="$JETSON_REPO_ROOT"
export UANWM_ENCODER_NAME="$JETSON_REPO_ROOT/pretrained/dinov3-vitb16"
export UANWM_DEVICE=cuda
