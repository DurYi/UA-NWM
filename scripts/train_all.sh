#!/usr/bin/env bash
# =============================================================================
#  UA-NWM  —  full 3-stage training in one script (mean baseline + HEP module)
# =============================================================================
#  Hyperparameters are aligned EXACTLY with the released model. Two stage configs are derived
#  from configs/ua_nwm.yaml at runtime (written into outputs/ua_nwm/ for inspection):
#    outputs/ua_nwm/train_backbone.yaml : model WITHOUT the HEP module, batch_size 12  (Stage 1+2)
#    outputs/ua_nwm/train_head.yaml     : model WITH the HEP module,    batch_size 48  (Stage 3)
#  This mirrors the original two-config split: the backbone is trained head-free (so DDP has no
#  grad-less params), and the HEP module is fresh-initialised in Stage 3 on the frozen backbone.
#
#  Stage 1 + 2  tokenizer autoencoder + delta predictor (rollout=8)   [torchrun DDP, batch 12/GPU]
#               -> outputs/ua_nwm/checkpoints/stage2_best.pt   (= the mean-predictor backbone)
#  Stage 3      HEP module, frozen backbone            [single GPU, batch 48]
#               -> outputs/ua_nwm/checkpoints/ua-nwm_best.pt
#  Then auto-plots the loss curves for all stages -> outputs/ua_nwm/loss_curves.png.
#
#  Effective backbone batch = 12 x #GPUs. The released model used 4 GPUs (=48); use 4 GPUs to align.
#
#  Usage:  TRAIN_GPUS=0,1,2,3 bash scripts/train_all.sh   (comma- or space-separated; DDP over all)
#  Skips a stage whose checkpoint already exists (idempotent / resumable).
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# activate the uanwm conda env if conda is on PATH (else assume it's already active)
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate uanwm
fi
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6

TRAIN_GPUS="${TRAIN_GPUS:-4 5 6 7}"          # DDP GPUs (backbone); comma- OR space-separated
TRAIN_GPUS="${TRAIN_GPUS//,/ }"              # normalize "0,1,2,3" -> "0 1 2 3"
NGPU=$(echo $TRAIN_GPUS | wc -w); DEV=$(echo $TRAIN_GPUS | tr ' ' ',')
GPU0=$(echo $TRAIN_GPUS | awk '{print $1}')  # single GPU for Stage 3
HEAD_GPU="${HEAD_GPU:-$GPU0}"
PORT="${MASTER_PORT:-29541}"
OUT_DIR=outputs/ua_nwm
mkdir -p "$OUT_DIR"
CKPT_DIR="$OUT_DIR/checkpoints"
BACKBONE="$CKPT_DIR/stage2_best.pt"; HEAD="$CKPT_DIR/ua-nwm_best.pt"
HEAD_LAST="$CKPT_DIR/ua-nwm_last.pt"
BACKBONE_CFG="$OUT_DIR/train_backbone.yaml"
HEAD_CFG="$OUT_DIR/train_head.yaml"

echo "===================================================================="
echo " UA-NWM 3-stage training | Stage 1+2 GPUs=[$DEV] (DDPx$NGPU, eff.batch=$((12*NGPU))) | Stage 3 GPU=$HEAD_GPU"
echo "===================================================================="

# ---- derive the two stage configs from the single source of truth ----
python - <<'PY'
import copy, yaml
c = yaml.safe_load(open("configs/ua_nwm.yaml"))
bb = copy.deepcopy(c); bb["model"].pop("hep", None); bb["training"]["batch_size"] = 12
yaml.safe_dump(bb, open("outputs/ua_nwm/train_backbone.yaml", "w"), sort_keys=False)
hd = copy.deepcopy(c); hd["training"]["batch_size"] = 48
yaml.safe_dump(hd, open("outputs/ua_nwm/train_head.yaml", "w"), sort_keys=False)
print("[configs] wrote outputs/ua_nwm/train_backbone.yaml (no HEP, batch 12) + outputs/ua_nwm/train_head.yaml (batch 48)")
PY

# ---- Stage 1 + 2 : delta backbone (DDP, head-free) ----
if [[ -f "$BACKBONE" ]]; then
  echo "[stage1+2] backbone exists -> SKIP ($BACKBONE)"
else
  echo "[stage1+2] training tokenizer + delta predictor (DDP x$NGPU) ..."
  CUDA_VISIBLE_DEVICES="$DEV" torchrun --nproc_per_node="$NGPU" --master_port="$PORT" \
    uanwm/staged_train.py --config "$BACKBONE_CFG" 2>&1 | tee "$OUT_DIR/stage12.log"
  [[ -f "$BACKBONE" ]] || { echo "FATAL: backbone stage2_best.pt not produced"; exit 1; }
fi

# ---- Stage 3 : HEP module (single GPU, frozen backbone, batch 48) ----
if [[ -f "$HEAD" && -f "$HEAD_LAST" ]]; then
  echo "[stage3] final UA-NWM checkpoint exists -> SKIP ($HEAD)"
else
  echo "[stage3] training HEP module on GPU $HEAD_GPU ..."
  CUDA_VISIBLE_DEVICES="$HEAD_GPU" python uanwm/train_hep.py \
    --config "$HEAD_CFG" --init_checkpoint "$BACKBONE" 2>&1 | tee "$OUT_DIR/stage3.log"
  [[ -f "$HEAD" ]] || { echo "FATAL: final ua-nwm_best.pt not produced"; exit 1; }
  [[ -f "$HEAD_LAST" ]] || { echo "FATAL: final ua-nwm_last.pt not produced"; exit 1; }
fi

# ---- loss curves for all stages -> output directory ----
echo "[plot] drawing loss curves ..."
python scripts/plot_loss.py --log "$OUT_DIR/train.log" --out "$OUT_DIR/loss_curves.png"

echo "done. backbone=$BACKBONE  final_best=$HEAD  final_last=$HEAD_LAST  curves=$OUT_DIR/loss_curves.png"
