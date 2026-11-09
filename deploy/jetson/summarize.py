"""Summarize the two formal runs without mixing their samples."""
import argparse
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", type=Path, default=root / "deploy/jetson/results/formal-batch4")
parser.add_argument("--batched", type=Path, default=root / "deploy/jetson/results/formal-batch32")
parser.add_argument("--output", type=Path, default=root / "deploy/jetson/DEPLOYMENT_REPORT.md")
args = parser.parse_args()
folders = [args.baseline.resolve(), args.batched.resolve()]
data = [json.loads((folder / "summary.json").read_text()) for folder in folders]
assert data[0]["metadata"]["checkpoint_sha256"] == data[1]["metadata"]["checkpoint_sha256"]
assert data[0]["metadata"]["encoder_sha256"] == data[1]["metadata"]["encoder_sha256"]
for item in data:
    assert item["summary"]["total_ms"]["n"] == 30, "Formal report expects 30 complete samples per condition"
    c = item["metadata"]["config"]
    assert (c["horizon"], c["cem_candidates"], c["cem_iterations"]) == (8, 32, 3)
speedup = data[0]["summary"]["total_ms"]["mean_ms"] / data[1]["summary"]["total_ms"]["mean_ms"]
smoke_path = folders[0] / "server_smoke.json"
smoke_note = "- 私有 loopback 服务功能验证尚未记录。"
if smoke_path.is_file():
    smoke = json.loads(smoke_path.read_text())
    assert smoke["first_request"]["goal_reached"] is False
    assert smoke["identical_goal_request"]["goal_reached"] is True
    smoke_note = f"- 本机 WebSocket 记录显示，首次请求在最小步数限制内返回未到达，最后一次请求使用与目标相同的图像并返回到达。[检查记录]({smoke_path})。"
lines = ["# UA-NWM Jetson Orin NX 部署与实测报告", "",
         "本报告比较两组 UA-NWM 完整推理测速，每组包含 30 次计时。", "",
         "## 测试条件", "",
         "| 项目 | 本次实测配置 |", "|---|---|",
         "| 设备、软件版本、功耗模式和时钟 | 见两组各自的 `metadata.json` |",
         "| 输入 | 每组重复使用同一对当前图和目标图。默认使用固定种子生成的合成图，也可指定图像 |",
         "| 搜索规模 | 32 个候选、3 轮 CEM、8 步预测。实际评分批大小见各组 `metadata.json` |",
         "| 测量 | 每组正式计时 30 次。预热和分段诊断配置见各组 `metadata.json` |",
         "| 时间边界 | 从调用 policy.infer 到其返回且 GPU 同步完成，计时前先同步 GPU |",
         "| 不计入 | 相机、网络、LPIPS 到达判定、ROS / PX4 控制；模型加载单独记录 |", "",
         "## 本次部署的模块位置", "",
         "| 模块 | 执行位置 | 是否计入主延迟 |", "|---|---|---|",
         "| 图像裁剪、缩放、归一化、历史管理、动作转换 | Jetson CPU | 是 |",
         "| DINOv3、状态压缩、世界模型 rollout、HEP 求解 | Jetson GPU | 是 |",
         "| CEM | Jetson：Python 循环在 CPU，候选采样/评分/筛选张量运算在 GPU | 是 |",
         "| LPIPS 到达判定 | Jetson CPU（服务默认配置） | 否 |",
         "| WebSocket 服务 | Jetson CPU | 否 |",
         "| ROS / PX4 | 测速程序未接入 | 否 |", "",
         "## 完整推理延迟", "",
         "| 输入组 | N | 平均 (ms) | 标准差 (ms) | P50 (ms) | P95 (ms) | 按平均延迟估算的串行推理速率 (次/s) | PyTorch 峰值分配 (MiB) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
for item, label in zip(data, ("基线组", "比较组")):
    s = item["summary"]["total_ms"]
    lines.append(f"| {label} | {s['n']} | {s['mean_ms']:.3f} | {s['std_ms']:.3f} | {s['p50_ms']:.3f} | {s['p95_ms']:.3f} | {1000/s['mean_ms']:.3f} | {item['cuda_peak_allocated_mib']:.1f} |")
lines += ["", f"基线组平均推理延迟除以比较组平均推理延迟为 **{speedup:.2f}**。解释该比值前，请核对两组的输入、软件配置、评分批大小和设备运行条件。", "",
          "## 分段诊断（含额外的 GPU 同步开销）", "",
          "| 阶段 | 基线组平均 (ms) | 比较组平均 (ms) |", "|---|---:|---:|"]
for key, label in (("preprocess_ms", "图像预处理"), ("encode_ms", "DINOv3 编码 + 状态压缩 + 输入搬运"),
                   ("plan_ms", "CEM（世界模型 rollout + HEP 评分及候选更新）"),
                   ("other_ms", "总耗时减去已计时阶段"), ("total_ms", "诊断轮完整推理")):
    values = [d["summary"]["profile_" + key]["mean_ms"] for d in data]
    lines.append(f"| {label} | {values[0]:.3f} | {values[1]:.3f} |")
lines += ["", "分段诊断和主结果使用独立轮次，不能把它们当作同一批样本相加或计算差值。", "",
          "## 加载与资源", "",
          "| 项目 | 基线组 | 比较组 |", "|---|---:|---:|"]
for key, label, scale in (("model_load_ms", "模型构建与加载 (s)", .001),
                          ("cuda_peak_reserved_mib", "PyTorch 峰值保留内存 (MiB)", 1)):
    lines.append(f"| {label} | {data[0][key]*scale:.3f} | {data[1][key]*scale:.3f} |")
for pattern, label in ((r"RAM (\d+)/", "整机 RAM 采样最大值 (MB)"), (r"gpu@([\d.]+)C", "GPU 温度采样最大值 (°C)")):
    values = []
    for folder in folders:
        samples = [float(x) for x in re.findall(pattern, (folder / "tegrastats.log").read_text())]
        values.append(max(samples) if samples else float("nan"))
    lines.append(f"| {label} | {values[0]:.1f} | {values[1]:.1f} |")
lines += ["", "整机 RAM 与 GPU 温度来自约每秒一次的设备采样。PyTorch 的峰值分配和峰值保留统计覆盖模型加载、预热、正式计时和分段诊断。", "",
          "## 部署与功能检查", "",
          "- Jetson 安装脚本使用独立的 `.venv-jetson` 环境。",
          "- 缺少 `action_stats.json` 时，安装脚本从检查点恢复动作归一化统计量。",
          "- 为 Jetson GPU wheel 固定 NumPy 1.26.4；按 [NVIDIA 的 cuSolver 兼容说明](https://docs.nvidia.com/holoscan/archive/3.10.0/hsdk_faq.html) 将 11.7.1.2 解压至项目私有目录，通过进程环境加载，系统 CUDA 库未替换。",
          "- 测速脚本检查返回航点和 HEP 评分是否为有限值。", smoke_note, "",
          "## 结论边界", "",
          "每组来自一次连续运行。P95 描述该次运行的样本，跨次稳定性需通过独立重复运行评估。",
          "合成静止图像用于测量推理性能。评估导航质量需要真实观测数据或飞行试验，尤其是在比较可能改变浮点计算次序的评分批大小时。", "",
          "## 文件与复现", "",
          f"- [基线组报告]({folders[0] / 'report.md'})",
          f"- [比较组报告]({folders[1] / 'report.md'})",
          f"- [部署及服务操作说明]({root / 'deploy/jetson/README.zh-CN.md'})", "",
          "以下命令从仓库根目录运行。测速输出目录应尚未创建。", "",
          "```bash",
          "bash deploy/jetson/run_benchmark.sh --score-batch-size 4 --output deploy/jetson/results/formal-batch4",
          "bash deploy/jetson/run_benchmark.sh --score-batch-size 32 --output deploy/jetson/results/formal-batch32",
          ".venv-jetson/bin/python deploy/jetson/summarize.py", "```", ""]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("\n".join(lines))
print(args.output)
