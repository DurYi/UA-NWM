#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$JETSON_REPO_ROOT"
exec "$JETSON_REPO_ROOT/.venv-jetson/bin/python" -u deploy/jetson/benchmark.py "$@"
