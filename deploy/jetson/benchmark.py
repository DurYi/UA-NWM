"""Offline GPU benchmark of the real deployment policy. No ROS or flight client.

Primary timings run the unmodified infer method. A separate diagnostic pass
adds synchronization at the encode/plan boundaries for a stage breakdown.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from uav_openpi.ua_nwm_policy import UANWMRealNavConfig, UANWMMpcCemPolicy
from uav_openpi.uav_schema import make_observation

ROOT = Path(__file__).resolve().parents[2]


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(a), "mean_ms": float(a.mean()),
            "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "min_ms": float(a.min()), "max_ms": float(a.max())}


def command_output(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def device_clocks():
    base = Path("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu")
    return {name: (base / name).read_text().strip() for name in
            ("governor", "cur_freq", "min_freq", "max_freq", "available_frequencies")
            if (base / name).is_file()}


def write_report(out, meta, rows, load_ms):
    measured = [r for r in rows if r["phase"] == "measured"]
    profiled = [r for r in rows if r["phase"] == "profile"]
    if not measured:
        return
    summary = {"total_ms": stats([r["total_ms"] for r in measured])}
    for key in ("preprocess_ms", "encode_ms", "plan_ms", "other_ms", "total_ms"):
        if profiled:
            summary["profile_" + key] = stats([r[key] for r in profiled])
    payload = {"metadata": meta, "model_load_ms": load_ms, "summary": summary,
               "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    total = summary["total_ms"]
    lines = ["# UA-NWM Jetson Orin NX 本机 GPU 推理测试", "",
             f"- 时间：{meta['timestamp']}",
             f"- GPU：{meta['gpu']}；功耗模式：{meta['power_mode'].replace(chr(10), ' / ')}",
             f"- GPU 频率上限：{int(meta['gpu_clocks_start'].get('max_freq', '0')) / 1e6:.0f} MHz；保留系统原设置。",
             f"- PyTorch {torch.__version__}，CUDA {torch.version.cuda}，Transformers {meta['versions']['transformers']}；AMP BF16。",
             f"- CEM：候选 {meta['config']['cem_candidates']}，迭代 {meta['config']['cem_iterations']}，预测 {meta['config']['horizon']} 步，评分批大小 {meta['config']['score_batch_size']}。",
             f"- 输入：{meta['input_description']}；4 帧历史 + 1 张目标图，内部缩放到 224×224。",
             f"- 预热 {meta['args']['warmup']} 次（不计入统计），正式 {len(measured)} 次；CPU 线程 {torch.get_num_threads()}。",
             "- 时间边界：已解码 RGB 数组进入 policy.infer() → 返回航点；首尾 CUDA synchronize + monotonic perf_counter_ns。",
             "- 包含图像预处理、DINOv3 编码、状态压缩、完整世界模型 rollout / HEP / CEM 及动作转换。",
             "- 不包含网络、相机采集、LPIPS 到达判定、ROS、控制器及真实飞行；本脚本不会连接飞控。",
             "", "## 主结果：未插入分段计时的完整推理", "",
             "| 指标 | 平均 (ms) | 标准差 (ms) | P50 (ms) | P95 (ms) | 最小 (ms) | 最大 (ms) |",
             "|---|---:|---:|---:|---:|---:|---:|",
             "| 完整 policy.infer | " + " | ".join(f"{total[k]:.3f}" for k in ("mean_ms", "std_ms", "p50_ms", "p95_ms", "min_ms", "max_ms")) + " |",
             "", f"平均延迟 {total['mean_ms']/1000:.3f} s；串行吞吐 {1000/total['mean_ms']:.3f} 次/s。",
             f"模型构建、权重加载和搬入 GPU：{load_ms/1000:.3f} s（不计入稳态推理）。",
             f"PyTorch 峰值分配 {payload['cuda_peak_allocated_mib']:.1f} MiB，峰值保留 {payload['cuda_peak_reserved_mib']:.1f} MiB（不等于整机内存）。", ""]
    if profiled:
        lines += ["## 独立诊断轮次的分段耗时", "",
                  f"以下 {len(profiled)} 次额外测量在分段边界同步 GPU，包含同步开销，不与上表混合统计。", "",
                  "| 阶段 | 平均 (ms) | P95 (ms) |", "|---|---:|---:|"]
        for key, label in (("preprocess_ms", "当前图及目标图预处理"), ("encode_ms", "DINOv3 编码 + 状态压缩 + 输入搬运"),
                           ("plan_ms", "完整 CEM（含世界模型 rollout / HEP）"), ("other_ms", "校验、历史管理、动作转换及其余开销"), ("total_ms", "诊断轮完整推理")):
            s = summary["profile_" + key]
            lines.append(f"| {label} | {s['mean_ms']:.3f} | {s['p95_ms']:.3f} |")
    lines += ["", "## 解释与复现", "", "这是固定工作量的推理速度测试；合成图像不能用于报告导航成功率、终点误差或飞行失败案例。",
              "未改变系统功耗模式或锁定频率；测试运行于当前系统负载下，tegrastats.log 保存资源观测。",
              "逐次原始记录见 samples.jsonl；配置、权重 SHA256、环境与汇总见 metadata.json / summary.json。", "",
              "```bash", f"cd {ROOT}", f"bash deploy/jetson/run_benchmark.sh --warmup {meta['args']['warmup']} --runs {meta['args']['runs']} --profile-runs {meta['args']['profile_runs']} --score-batch-size {meta['args']['score_batch_size']}", "```", ""]
    (out / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--profile-runs", type=int, default=10)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--current-image", type=Path)
    parser.add_argument("--goal-image", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.runs, args.score_batch_size) < 1 or min(args.warmup, args.profile_runs) < 0:
        parser.error("runs / score-batch-size must be positive; warmup / profile-runs must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: refusing a silently substituted CPU benchmark")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    out = args.output or ROOT / "deploy/jetson/results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    def rgb(path):
        return np.asarray(Image.open(path).convert("RGB")) if path else rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    current, goal = rgb(args.current_image), rgb(args.goal_image)
    if args.current_image is None:
        Image.fromarray(current).save(out / "synthetic_current.png")
    if args.goal_image is None:
        Image.fromarray(goal).save(out / "synthetic_goal.png")
    config = UANWMRealNavConfig(
        uanwm_root=ROOT, model_config=ROOT / "configs/ua_nwm.yaml",
        checkpoint=ROOT / "pretrained/ua-nwm/ua-nwm_best.pt",
        action_stats=ROOT / "pretrained/ua-nwm/action_stats.json",
        encoder_name=str(ROOT / "pretrained/dinov3-vitb16"), device="cuda",
        score_batch_size=args.score_batch_size, random_seed=42,
        # Match deploy/run_server.sh, including its horizontal-motion distribution.
        action_mean=(8., 0., 0., 0.), action_std=(1.5, 1.5, 0., .60),
        action_low=(0., -3., 0., -.785398), action_high=(15., 3., 0., .785398))
    meta = {"timestamp": datetime.now().astimezone().isoformat(), "args": vars(args),
            "config": asdict(config), "python": platform.python_version(), "machine": platform.machine(),
            "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "power_mode": command_output(["nvpmodel", "-q"]),
            "power_mode_verbose": command_output(["nvpmodel", "-q", "--verbose"]),
            "gpu_clocks_start": device_clocks(),
            "private_cusolver": str(ROOT / "deploy/jetson/runtime/lib/libcusolver.so.11.7.1.2"),
            "git_commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "l4t": Path("/etc/nv_tegra_release").read_text(),
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "transformers", "numpy", "safetensors")},
            "input_description": f"{'指定图像' if args.current_image else '合成 uint8 RGB 图像（种子 42）'}，current={list(current.shape)}，goal={list(goal.shape)}",
            "checkpoint_sha256": sha256(config.checkpoint),
            "encoder_sha256": sha256(ROOT / "pretrained/dinov3-vitb16/model.safetensors")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"RESULT_DIR={out}", flush=True)
    telemetry_file = (out / "tegrastats.log").open("w")
    telemetry = None
    rows = []
    load_ms = 0.
    try:
        try:
            telemetry = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=telemetry_file, stderr=subprocess.STDOUT)
        except OSError:
            pass
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        policy = UANWMMpcCemPolicy(config)
        torch.cuda.synchronize()
        load_ms = (time.perf_counter_ns() - start) / 1e6
        assert policy._device.type == "cuda"
        meta["parameter_count"] = sum(p.numel() for p in policy._model.parameters())
        meta["encoder_attention"] = str(getattr(policy._model.encoder.model.config, "_attn_implementation", None))
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        print(f"Loaded policy in {load_ms / 1000:.3f} s; device={policy._device}; dtype={policy._amp_dtype}", flush=True)
        stage = {}
        originals = {name: getattr(policy, name) for name in ("_image_tensor", "_encode_context_goal", "_plan")}
        def timed(name, key, sync):
            original = originals[name]
            def call(*a, **kw):
                if sync:
                    torch.cuda.synchronize()
                before = time.perf_counter_ns()
                result = original(*a, **kw)
                if sync:
                    torch.cuda.synchronize()
                stage[key] = stage.get(key, 0.) + (time.perf_counter_ns() - before) / 1e6
                return result
            return call
        with (out / "samples.jsonl").open("w") as samples:
            for phase, count in (("warmup", args.warmup), ("measured", args.runs), ("profile", args.profile_runs)):
                if phase == "profile":
                    for name, key, sync in (("_image_tensor", "preprocess_ms", False), ("_encode_context_goal", "encode_ms", True), ("_plan", "plan_ms", True)):
                        setattr(policy, name, timed(name, key, sync))
                for i in range(count):
                    obs = make_observation(images={"front": current, "goal": goal}, state=np.zeros(12, dtype=np.float32))
                    stage.clear()
                    torch.cuda.synchronize()
                    before = time.perf_counter_ns()
                    result = policy.infer(obs)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter_ns() - before) / 1e6
                    actions = np.asarray(result["actions"])
                    energy = float(result["debug"]["best_energy"])
                    if actions.shape != (1, 4) or not np.isfinite(actions).all() or not np.isfinite(energy):
                        raise RuntimeError("Invalid / non-finite action or HEP energy")
                    row = {"phase": phase, "index": i, "total_ms": elapsed, "actions": actions.tolist(), "best_energy": energy, **stage}
                    if phase == "profile":
                        row["other_ms"] = elapsed - sum(stage.values())
                    rows.append(row)
                    samples.write(json.dumps(row) + "\n")
                    samples.flush()
                    print(f"{phase} {i+1}/{count}: {elapsed:.3f} ms; energy={energy:.6f}", flush=True)
        meta["gpu_clocks_end"] = device_clocks()
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        write_report(out, meta, rows, load_ms)
        print(f"REPORT={out / 'report.md'}", flush=True)
    except BaseException as exc:
        (out / "error.txt").write_text(repr(exc) + "\n")
        write_report(out, meta, rows, load_ms)
        raise
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()
        telemetry_file.close()


if __name__ == "__main__":
    main()
