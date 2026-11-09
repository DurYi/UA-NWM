# UA-NWM 在 Jetson Orin NX 上部署与测速

[English](README.md)

本目录提供 UA-NWM 在 Jetson Orin NX 上的本机推理服务和离线测速脚本。所有命令均从仓库根目录执行。

## 目标环境

| 项目 | 配置 |
|---|---|
| 硬件 | Jetson Orin NX 16 GB，ARM64 |
| 系统 | Ubuntu 22.04，L4T 36.4.3 / JetPack 6.2 |
| CUDA / cuDNN | 12.6 / 9.3 |
| Python | 3.10.12 |
| PyTorch / torchvision | Jetson AI Lab GPU 版 2.8.0 / 0.23.0，SM 8.7 |
| Transformers / NumPy | 5.3.0 / 1.26.4 |
| 精度 | BF16 autocast，FP32 权重 |
| cuSolver | 项目本地库 11.7.1.2 |

## 安装

按[仓库的权重下载说明](../../README.md#2-checkpoint-preparation)准备 `pretrained/dinov3-vitb16/` 和 `pretrained/ua-nwm/ua-nwm_best.pt`，再运行：

```bash
bash deploy/jetson/setup.sh
```

安装脚本创建独立的 `.venv-jetson`，使用匹配 JetPack 的 GPU wheel，并将 cuSolver 解压到 `deploy/jetson/runtime/lib/` 供项目进程使用。缺少 `action_stats.json` 时，脚本从 UA-NWM 检查点提取动作归一化统计量。首次安装需要联网下载依赖和 LPIPS 使用的 AlexNet 权重。依赖版本见 [requirements.lock.txt](requirements.lock.txt)。

使用本目录的启动脚本即可加载环境。直接调用 `.venv-jetson/bin/python` 时，先运行 `source deploy/jetson/env.sh` 来设置 CUDA 库和 Python 模块的搜索路径。

cuSolver 兼容性说明和 GPU wheel 安装文档见 [NVIDIA Holoscan FAQ Q29](https://docs.nvidia.com/holoscan/archive/3.10.0/hsdk_faq.html) 和 [PyTorch JetPack 文档](https://docs.pytorch.org/TensorRT/getting_started/jetpack.html)。

## 离线测速

```bash
bash deploy/jetson/run_benchmark.sh
```

默认预热 5 次、正式计时 30 次，另做 10 次分段诊断。模型构建与加载单独计时，正式测量首尾同步 GPU。报告、原始样本、环境元数据和设备日志默认写入 Git 忽略的 `deploy/jetson/results/`。自定义 `--output` 时，也可以在该目录下指定新的子目录。

工作量与 `deploy/run_server.sh` 对齐：32 个候选，3 轮 CEM，8 步预测，每批评分 4 个候选，4 帧历史及 1 张目标图，内部 224×224 输入。默认使用种子固定的 640×480 RGB 合成图，重复当前图相当于静止观测。它只用于推理速度测试，不能验证导航成功率。

将示例路径替换为自己的图像路径，即可使用指定图像测速：

```bash
bash deploy/jetson/run_benchmark.sh \
  --current-image ./path/to/current.jpg \
  --goal-image ./path/to/goal.jpg
```

较大评分批次的独立对比：

```bash
bash deploy/jetson/run_benchmark.sh --score-batch-size 32
```

这个参数将候选集中处理，不减少候选数、迭代次数或预测步数。不同批大小可能造成浮点及 CEM 选择差异；速度测试不等同于真实导航质量验证。原服务默认仍为 4。

完整 `policy.infer()` 包含预处理、视觉编码、状态压缩、世界模型 rollout、HEP、CEM 和动作转换。不含网络、相机、LPIPS 目标到达检查、ROS 或控制器。分段数据来自额外诊断轮次，不与未插入分段同步的主统计混合。

## 在 Jetson 启动推理服务

将示例路径替换为自己的目标图路径：

```bash
# 如需鉴权，在当前终端设置 UAV_POLICY_API_KEY；脚本会沿用它。
bash deploy/jetson/run_server.sh ./path/to/goal.jpg
```

默认监听 `127.0.0.1:8001`，使用 CUDA 模型和原部署的 LPIPS CPU 到达判定。目标图必须由调用者指定。脚本不会启动飞行客户端、解锁无人机或发布 ROS 指令。停止服务按 Ctrl+C。

如需较大评分批次：

```bash
UANWM_CEM_SCORE_BATCH_SIZE=32 bash deploy/jetson/run_server.sh ./path/to/goal.jpg
```

`HOST`、`PORT`、`UAV_POLICY_API_KEY` 和 `UANWM_CEM_*` 均可用环境变量设置。图像落盘录制默认关闭，可通过 `serve_policy.py` 的录制参数启用。

只测本机协议和推理的合成客户端示例：

```bash
source deploy/jetson/env.sh
.venv-jetson/bin/python deploy/examples/simple_client.py \
  --host 127.0.0.1 --port 8001 --steps 1 --timeout 120
```

也可以用合成图运行自动功能检查。先生成测试输入，再运行脚本验证推理和目标到达判定。脚本会自行启动和关闭本机服务。

```bash
bash deploy/jetson/run_benchmark.sh \
  --warmup 0 --runs 1 --profile-runs 0 \
  --output deploy/jetson/results/smoke-inputs
source deploy/jetson/env.sh
.venv-jetson/bin/python deploy/jetson/smoke_server.py \
  --inputs deploy/jetson/results/smoke-inputs
```

第 3 次请求开始允许返回到达状态。重复运行时，请为 `--output` 指定一个尚不存在的目录，并将同一路径传给 `--inputs`。
