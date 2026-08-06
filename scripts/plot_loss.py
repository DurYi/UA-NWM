"""Parse a UA-NWM train.log and plot the training-loss curve of every stage.

Both trainers log one line per interval to outputs/ua_nwm/train.log:
  Stage 1 :  ... | INFO | stage=stage1 step=50 loss=0.1234 ...
  Stage 2 :  ... | INFO | stage=stage2 step=50 loss=0.2345 ...
  Stage 3 :  ... | INFO | stage=stage3 step=50 lr=... loss=0.5678 ortho_cost=...
Lines carrying `stage=` are grouped by that tag; the untagged `step=/loss=` lines are the
legacy HEP module. One panel per stage, saved to a single PNG.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP = re.compile(r"\bstep=(\d+)\b")
LOSS = re.compile(r"\bloss=([-+0-9.eE]+)")
STAGE = re.compile(r"\bstage=(\w+)")


def parse(log_path: Path) -> dict[str, tuple[list[int], list[float]]]:
    curves: dict[str, tuple[list[int], list[float]]] = {}
    for line in log_path.read_text().splitlines():
        m_step, m_loss = STEP.search(line), LOSS.search(line)
        if not (m_step and m_loss):
            continue
        m_stage = STAGE.search(line)
        stage = m_stage.group(1) if m_stage else "stage3"
        xs, ys = curves.setdefault(stage, ([], []))
        xs.append(int(m_step.group(1)))
        ys.append(float(m_loss.group(1)))
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    curves = parse(Path(args.log))
    if not curves:
        raise SystemExit(f"no step=/loss= lines found in {args.log}")

    # internal log token -> display label
    LABEL = {"stage1": "Stage 1: representation", "stage2": "Stage 2: prediction",
             "stage3": "Stage 3: HEP"}
    order = [s for s in ("stage1", "stage2", "stage3") if s in curves]
    order += [s for s in curves if s not in order]
    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, stage in zip(axes[0], order):
        xs, ys = curves[stage]
        ax.plot(xs, ys, lw=1.2, color="steelblue")
        ax.set_title(f"{LABEL.get(stage, stage)}  (final loss={ys[-1]:.4f})", fontsize=10)
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.grid(alpha=0.3)
    fig.suptitle("UA-NWM training loss", fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[plot_loss] wrote {args.out}  (stages: {', '.join(order)})")


if __name__ == "__main__":
    main()
