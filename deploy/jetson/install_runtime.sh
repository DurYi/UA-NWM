#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
# NVIDIA Holoscan FAQ Q29: JPL PyTorch needs cuSolver 11.7.1.2.
# Extract into this project; never apt-install or replace host CUDA libraries.
JETSON_RUNTIME="$JETSON_REPO_ROOT/deploy/jetson/runtime"
mkdir -p "$JETSON_RUNTIME/lib"
if [[ ! -f "$JETSON_RUNTIME/lib/libcusolver.so.11.7.1.2" ]]; then
  JETSON_TMP="$(mktemp -d /tmp/uanwm-cusolver.XXXXXX)"
  trap 'rm -rf -- "$JETSON_TMP"' EXIT
  curl -fSL --retry 3 -o "$JETSON_TMP/libcusolver.deb" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/libcusolver-12-6_11.7.1.2-1_arm64.deb
  echo "d059b90c4b1e4d9ae62bd4346d52aa8cd7dc4b523d4bba809558d965944c3890  $JETSON_TMP/libcusolver.deb" | sha256sum -c -
  sha256sum "$JETSON_TMP/libcusolver.deb" > "$JETSON_RUNTIME/cusolver-package.sha256"
  dpkg-deb -x "$JETSON_TMP/libcusolver.deb" "$JETSON_TMP/extracted"
  cp -a "$JETSON_TMP/extracted/usr/local/cuda-12.6/targets/aarch64-linux/lib/"libcusolver.so* "$JETSON_RUNTIME/lib/"
fi
echo "Private cuSolver runtime: $JETSON_RUNTIME/lib"
