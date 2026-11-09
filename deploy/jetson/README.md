# Deploying UA-NWM on Jetson Orin NX

[简体中文](README.zh-CN.md)

This directory provides an inference server and an offline GPU benchmark for UA-NWM on Jetson Orin NX. Run all commands from the repository root.

## Target environment

| Component | Configuration |
| --- | --- |
| Hardware | Jetson Orin NX 16 GB, ARM64 |
| OS | Ubuntu 22.04, L4T 36.4.3 / JetPack 6.2 |
| CUDA / cuDNN | 12.6 / 9.3 |
| Python | 3.10.12 |
| PyTorch / torchvision | Jetson AI Lab GPU wheels 2.8.0 / 0.23.0, SM 8.7 |
| Transformers / NumPy | 5.3.0 / 1.26.4 |
| Precision | BF16 autocast with FP32 weights |
| cuSolver | Project-local library 11.7.1.2 |

## Installation

Follow the [repository's checkpoint instructions](../../README.md#2-checkpoint-preparation) to prepare `pretrained/dinov3-vitb16/` and `pretrained/ua-nwm/ua-nwm_best.pt`, then run the setup script.

```bash
bash deploy/jetson/setup.sh
```

Setup creates `.venv-jetson`, installs GPU wheels for JetPack, and extracts cuSolver into `deploy/jetson/runtime/lib/` for use by the project's processes. If `action_stats.json` is missing, setup extracts the action normalization statistics from the UA-NWM checkpoint. The first installation requires internet access to download dependencies and the AlexNet weights used by LPIPS. Package versions are listed in [requirements.lock.txt](requirements.lock.txt).

The launch scripts load the environment automatically. Before invoking `.venv-jetson/bin/python` directly, run `source deploy/jetson/env.sh` to set the CUDA library and Python module search paths.

See the [NVIDIA Holoscan FAQ, Q29](https://docs.nvidia.com/holoscan/archive/3.10.0/hsdk_faq.html) for cuSolver compatibility and the [PyTorch JetPack documentation](https://docs.pytorch.org/TensorRT/getting_started/jetpack.html) for GPU wheel installation.

## Offline benchmark

```bash
bash deploy/jetson/run_benchmark.sh
```

The benchmark runs 5 warmup iterations, 30 timed iterations, and 10 separate profiling iterations by default. It measures model setup separately and synchronizes the GPU at the start and end of each timed inference. Reports, samples, environment metadata, and device logs are written under the Git-ignored `deploy/jetson/results/` directory. When using `--output`, you can specify a new subdirectory there.

The default workload uses 32 candidates, 3 CEM iterations, an 8-step prediction horizon, and a scoring batch size of 4. It uses four history frames and one goal frame, resizes inputs to 224 × 224, and generates 640 × 480 synthetic RGB images with a fixed seed by default. You can supply your own images by replacing the example paths below.

```bash
bash deploy/jetson/run_benchmark.sh \
  --current-image ./path/to/current.jpg \
  --goal-image ./path/to/goal.jpg
```

To group more candidates into each scoring batch, use the following command.

```bash
bash deploy/jetson/run_benchmark.sh --score-batch-size 32
```

Changing the scoring batch size keeps the candidate count, iteration count, and prediction horizon fixed. It may affect floating-point results and CEM selection. Navigation quality requires a separate evaluation.

The timed `policy.infer()` call covers preprocessing, visual encoding, state compression, world-model rollout, HEP, CEM, and action conversion. Network transport, camera capture, LPIPS goal detection, ROS, and controller execution are outside the timing scope.

## Inference server

Replace the example path with your goal image.

```bash
# Set UAV_POLICY_API_KEY in the environment if you want authentication.
bash deploy/jetson/run_server.sh ./path/to/goal.jpg
```

The server listens on `127.0.0.1:8001` by default. UA-NWM inference runs on CUDA, and LPIPS goal detection runs on the CPU by default. Press Ctrl+C to stop the inference server.

Use `HOST`, `PORT`, `UAV_POLICY_API_KEY`, and `UANWM_CEM_*` to configure the server. To change the scoring batch size, use the following command.

```bash
UANWM_CEM_SCORE_BATCH_SIZE=32 bash deploy/jetson/run_server.sh ./path/to/goal.jpg
```

Image recording is disabled by default. Use the recording options from `deploy/scripts/serve_policy.py` to enable it.

## Local functional checks

With the server running, send one synthetic request from another terminal.

```bash
source deploy/jetson/env.sh
.venv-jetson/bin/python deploy/examples/simple_client.py \
  --host 127.0.0.1 --port 8001 --steps 1 --timeout 120
```

For an automated check, generate synthetic inputs and run the smoke test. It starts and stops its own loopback server and checks inference and goal detection.

```bash
bash deploy/jetson/run_benchmark.sh \
  --warmup 0 --runs 1 --profile-runs 0 \
  --output deploy/jetson/results/smoke-inputs
source deploy/jetson/env.sh
.venv-jetson/bin/python deploy/jetson/smoke_server.py \
  --inputs deploy/jetson/results/smoke-inputs
```

The goal check can first report arrival on the third request. To repeat the example, choose a new directory for `--output` and pass the same path to `--inputs`.
