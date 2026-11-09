#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$JETSON_REPO_ROOT"
[[ "$(uname -m)" == aarch64 ]] || { echo 'This installer targets Jetson ARM64 / JetPack 6.2.' >&2; exit 1; }
JETSON_PYTHON="$JETSON_REPO_ROOT/.venv-jetson/bin/python"
if [[ ! -x "$JETSON_PYTHON" ]]; then
  python3 -m venv --without-pip "$JETSON_REPO_ROOT/.venv-jetson"
fi
if ! "$JETSON_PYTHON" -m pip --version >/dev/null 2>&1; then
  JETSON_PIP_BOOTSTRAP="$(mktemp /tmp/uanwm-get-pip.XXXXXX.py)"
  trap 'rm -f -- "$JETSON_PIP_BOOTSTRAP"' EXIT
  curl -fSL https://bootstrap.pypa.io/get-pip.py -o "$JETSON_PIP_BOOTSTRAP"
  "$JETSON_PYTHON" "$JETSON_PIP_BOOTSTRAP" --no-cache-dir
fi
"$JETSON_PYTHON" -m pip install --no-cache-dir --no-deps \
  torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
# These exact torch versions are already satisfied by the GPU wheels above.
"$JETSON_PYTHON" -m pip install --no-cache-dir -r deploy/jetson/requirements.lock.txt
bash deploy/jetson/install_runtime.sh
"$JETSON_PYTHON" deploy/jetson/prepare_model.py
"$JETSON_PYTHON" - <<'PY'
import torch, numpy as np
assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.from_numpy(np.eye(2, dtype=np.float32)).cuda()
assert torch.allclose(torch.linalg.solve(x, x), x)
torch.cuda.synchronize()
print("Verified:", torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
PY
# Cache the original server's optional goal-check backbone in the project.
"$JETSON_PYTHON" -c 'import lpips; lpips.LPIPS(net="alex").eval()'
"$JETSON_PYTHON" -m pip check
