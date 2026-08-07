<div align="center">

# UA-NWM

### Uncertainty-Aware World Model for Aerial Image-Goal Navigation

[Deyi Zhu](https://github.com/DurYi/)<sup>&ast;</sup>,
[Haoyu Fan](https://github.com/FanG0D)<sup>&ast;</sup>,
[Yinan Zhu](https://github.com/Seattle-233),
[Weichen Zhang](https://scholar.google.com/citations?user=Xsb7crkAAAAJ&hl=zh-CN),
[Shilin Ma](https://github.com/cyp336/),
[Xinlei Chen](https://chen-xinlei.com),
[Yansong Tang](https://andytang15.github.io)<sup>†</sup>

Tsinghua Shenzhen International Graduate School, Tsinghua University

<sub><sup>&ast;</sup> Equal contribution</sub>

<p align="center">
  <a href="https://arxiv.org/abs/2608.05597"><img src="https://img.shields.io/badge/arXiv-2608.05597-b31b1b.svg" alt="arXiv"></a>
  <a href="https://duryi.github.io/UA-NWM-Project-Page/"><img src="https://img.shields.io/badge/Project%20Page-UA--NWM-2b6cb0.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/DurYi/AirGoal-10k"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg" alt="Hugging Face Dataset"></a>
  <a href="https://pan.baidu.com/s/1xzUz1laMjZY1CyDWNec_FQ?pwd=uawm"><img src="https://img.shields.io/badge/%E2%98%81%EF%B8%8F%20Baidu%20Netdisk-Dataset-2932e1.svg" alt="Baidu Netdisk Dataset"></a>
  <a href="https://huggingface.co/DurYi/UA-NWM-Checkpoints/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-yellow.svg" alt="Hugging Face Checkpoints"></a>
</p>

**Official repository** for the [paper](https://arxiv.org/abs/2608.05597)
_"Uncertainty-Aware World Model for Aerial Image-Goal Navigation."_

</div>

---

## 📖 Overview

UA-NWM is an uncertainty-aware navigation world model for aerial image-goal navigation. Given a
candidate navigation trajectory, it predicts future DINO latent features and scores whether the goal
image falls within the predicted future-state distribution. Instead of relying on a single point
prediction, UA-NWM uses Hierarchical Error Projection (HEP) to decompose the prediction-goal
discrepancy into explained and unexplained components, and scores the residual that cannot be
explained by the uncertainty subspace.

This repository provides the training, evaluation, visualization, and deployment code for UA-NWM on
AirGoal-10k.

## ✨ Highlights

- **Uncertainty-aware trajectory scoring:** HEP scores only the goal discrepancy that lies outside
  the predicted uncertainty subspace.
- **One released world model, two inference strategies:** the uncertainty-aware scorer (`UA-NWM`)
  and the deterministic baseline (`UA-NWM_deterministic`) share the same checkpoint.
- **Full AirGoal-10k workflow:** scripts cover staged training, trajectory ranking, CEM/MPC planning,
  metric recomputation, and DINO latent visualization.
- **Real-world deployment adapter:** `deploy/` contains the server-side policy wrapper used for
  UAV deployment experiments.

## 🗺️ Roadmap

- [x] **Done** — Release video demos on the [🌐 project page](https://duryi.github.io/UA-NWM-Project-Page/).
- [x] **Done** — Release pretrained checkpoints on [🤗 Hugging Face](https://huggingface.co/DurYi/UA-NWM-Checkpoints/).
- [x] **Done** — Release the paper on [📄 arXiv](https://arxiv.org/abs/2608.05597).
- [x] **Done** — Release the AirGoal-10k dataset on [🤗 Hugging Face](https://huggingface.co/datasets/DurYi/AirGoal-10k) and [☁️ Baidu Netdisk](https://pan.baidu.com/s/1xzUz1laMjZY1CyDWNec_FQ?pwd=uawm).

## 🚀 Getting Started

Run all commands from the repository root unless noted otherwise.

### 1. Create the conda environment

```bash
conda create -n uanwm python=3.10 -y
conda activate uanwm

# PyTorch + torchvision, CUDA 12.1 build (matches the released model)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies (transformers>=5.3 is required for the DINOv3 encoder)
pip install -r requirements.txt
```

Sanity-check the GPU build:

```bash
python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
# -> 2.5.1+cu121 True 5.3.0
```

> The PyTorch CUDA 12.1 wheels also run on newer CUDA drivers (e.g. 12.x). If your GPU needs a
> different CUDA build, pick the matching index URL from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) —
> the rest of `requirements.txt` is CUDA-agnostic.

### 2. Checkpoint Preparation

We recommend downloading checkpoints in advance and placing them under `pretrained/`. Training and
evaluation use the DINOv3 encoder and UA-NWM checkpoint; DINO latent visualization additionally uses
the optional RAEv2 decoder.

1. **DINOv3 ViT-B/16 encoder.** Download the official gated Meta checkpoint from
   [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
   and place it at:

   ```text
   pretrained/dinov3-vitb16/
   ```

   A typical Hugging Face CLI workflow is:

   ```bash
   hf auth login
   hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
     --local-dir pretrained/dinov3-vitb16
   ```

2. **UA-NWM checkpoint.** Download the released UA-NWM checkpoint from
   [Hugging Face](https://huggingface.co/DurYi/UA-NWM-Checkpoints/) and place it at:

   ```text
   pretrained/ua-nwm/ua-nwm_best.pt
   ```

   Both inference strategies use the same file:
   `hep` scores with the HEP module, while `dinov3_cosine` uses the shared deterministic baseline.

3. [Optional] **RAEv2 decoder for visualization.** Download the fine-tuned
   [RAEv2](https://github.com/nanovisionx/RAEv2) decoder checkpoint from
   [Hugging Face](https://huggingface.co/DurYi/UA-NWM-Checkpoints/) and place it at:

   ```text
   pretrained/raev2-dinov3b-k1-airgoal10k-ft/step_0010000.pt
   ```

   This checkpoint is only required for DINO latent visualization.

### 3. Dataset Preparation

Download AirGoal-10k from [Hugging Face](https://huggingface.co/datasets/DurYi/AirGoal-10k)
and place it under `data/`:

```bash
hf download DurYi/AirGoal-10k \
  --repo-type dataset \
  --local-dir data \
  --include "airgoal10k/**"
```

The dataset is also available from
[Baidu Netdisk](https://pan.baidu.com/s/1xzUz1laMjZY1CyDWNec_FQ?pwd=uawm).

After downloading, the expected local path is:

```text
data/airgoal10k/
```

If you store the dataset or checkpoints elsewhere, update `configs/ua_nwm.yaml`:

- `paths.raw_data` points to the training/validation root.
- `eval.root` points to the test split.
- `model.encoder_name` points to the local DINOv3 encoder path or a Hugging Face model ID.

## 🧱 Layout

### Included in this repository

```
UA-NWM/
├── configs/ua_nwm.yaml          # training/evaluation configuration
├── data_splits/                 # train/val/test trajectory-id split lists
├── deploy/                      # real-world UAV deployment adapter and server wrapper
├── scripts/                     # CLI entry points for training, evaluation, and visualization
└── uanwm/                       # datasets, models, losses, evaluation, and visualization modules
```

### Created during data/checkpoint preparation

```
UA-NWM/
├── pretrained/
│   ├── dinov3-vitb16/                    # frozen DINOv3 ViT-B/16 encoder
│   ├── raev2-dinov3b-k1-airgoal10k-ft/
│   │   └── step_0010000.pt               # fine-tuned RAEv2 decoder for DINO latent visualization
│   └── ua-nwm/
│       └── ua-nwm_best.pt                # released UA-NWM checkpoint: mean predictor + HEP module
└── data/airgoal10k/
    ├── action_stats.json                 # dataset-level action normalizer
    ├── pseudo_goals/                     # example pseudo-goal images for visualization/debugging
    ├── train/traj_000000..008999/        # 9000 training trajectories
    ├── val/traj_000000..000999/          # 1000 validation trajectories
    └── test/traj_000000..000999/         # 1000 evaluation trajectories (+32 candidates each)
```

### Generated after running scripts

```
UA-NWM/
├── outputs/                     # training logs, checkpoints, and loss curves
├── results/                     # offline ranking/planning predictions and metrics
└── visualization/               # DINO latent visualization outputs
```

## 📊 Evaluate the pretrained model

### Offline ranking (8/16/32-choose-1)

```bash
GPU=0 bash scripts/eval_ranking.sh
```

Each test case is rolled out once and scored with both inference strategies. The script reports
ATE / RPE for 8-, 16-, and 32-candidate ranking and writes aggregate metrics to
`results/airgoal10k/offline_ranking/metrics.json`. Raw per-case ranking decisions and trajectories
are written to
`results/airgoal10k/offline_ranking/<strategy>/per_case.json`.

### Offline planning (CEM/MPC, 32x3)

```bash
GPU=0 bash scripts/eval_planning.sh
```

This runs CEM planning (`--num_samples 32 --topk 16 --opt_steps 3`) once per
strategy and reports ATE / RPE.

Aggregate metrics are written to `results/airgoal10k/<setting>/metrics.json`. Raw per-case planning
trajectories are written to `results/airgoal10k/<setting>/<strategy>/per_case.json`.

### Metrics only (no inference)

```bash
bash scripts/compute_metrics.sh results/airgoal10k/offline_ranking     # or .../offline_planning
```

This recomputes ATE / RPE directly from dumped `per_case.json` files with the setting auto-detected.
It is useful for re-scoring existing runs or validating results produced elsewhere with the same raw
schema.

We also provide the raw evaluation dumps used for the paper in
[Google Drive](https://drive.google.com/drive/folders/1j8VcVxT-UD6raGU-tYlwEW3mFCrTVbUN?usp=sharing).
After downloading, place them under `results/airgoal10k/` and run the command above to reproduce
the reported ranking and planning metrics.

## 🎨 DINO latent visualization

The visualization tool decodes predicted DINO latent features through the fine-tuned
[RAEv2](https://github.com/nanovisionx/RAEv2) RGB decoder. It supports three modes:

- `random`: draw random samples from the predicted subspace `S`.
- `goal_oriented`: project the real goal and optional pseudo-goal DINO features onto `S`, then
  render `e`, `e_parallel`, and `e_perp`.
- `both`: run both views in one command.

The default RAEv2 decoder checkpoint is:

```text
pretrained/raev2-dinov3b-k1-airgoal10k-ft/step_0010000.pt
```

Run random sampling with 16 samples:

```bash
TRAJ=traj_000364 MODE=random GPU=0 bash scripts/visualize_uncertainty.sh
```

Run random sampling and goal-oriented pseudo-goal visualization:

```bash
TRAJ=traj_000364 MODE=both GPU=0 \
PSEUDO_GOAL=data/airgoal10k/pseudo_goals/test/traj_000364 \
bash scripts/visualize_uncertainty.sh
```

If `PSEUDO_GOAL` is omitted, the bash wrapper leaves `--pseudo_goal` unset. In this case,
`goal_oriented` still writes the true-goal visualization and skips only the additional pseudo-goal
panels.

Outputs are written under:

```text
visualization/uncertainty/<traj_id>/
├── overview_random.png
├── overview_goal_oriented.png
├── random/          # reference frames, goal-by-S projections, heatmaps, random samples
└── goal_oriented/   # DINO-to-RGB reconstructions, goal-by-S projections, e/e_parallel/e_perp maps
```

## 🏋️ Train from scratch

```bash
TRAIN_GPUS=0,1,2,3 bash scripts/train_all.sh
```

- **Stage 1 + 2** (DDP): tokenizer autoencoder + delta predictor (rollout 8) →
  `outputs/ua_nwm/checkpoints/stage2_best.pt` (the mean-predictor backbone).
- **Stage 3** (single GPU): HEP module on the frozen backbone →
  `outputs/ua_nwm/checkpoints/ua-nwm_best.pt` and `outputs/ua_nwm/checkpoints/ua-nwm_last.pt`.
- Loss curves for all stages are automatically saved to `outputs/ua_nwm/loss_curves.png`.

Effective batch = per-GPU `batch_size` (12) × #GPUs; use 4 GPUs to reproduce the released model
(effective batch 48). Point the eval scripts' checkpoint at your freshly trained
`outputs/ua_nwm/checkpoints/ua-nwm_best.pt` to evaluate it.

## 🙏 Acknowledgment

This repository builds on ideas and implementation patterns from
[Navigation World Models](https://www.amirbar.net/nwm/index.html)
([code](https://github.com/facebookresearch/nwm)) and
[DeltaTok](https://deltatok.github.io/)
([code](https://github.com/amazon-far/deltatok)). AirGoal-10k is built on top of
[AerialVLN](https://github.com/AirVLN/AirVLN) and
[OpenFly](https://huggingface.co/papers/2502.18041). We thank the authors for releasing and
documenting their work.

## 📚 Citation

```bibtex
@misc{zhu2026uanwm,
      title={Uncertainty-Aware World Model for Aerial Image-Goal Navigation},
      author={Deyi Zhu and Haoyu Fan and Yinan Zhu and Weichen Zhang and Shilin Ma and Xinlei Chen and Yansong Tang},
      year={2026},
      eprint={2608.05597},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.05597},
}
```
