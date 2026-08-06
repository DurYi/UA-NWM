#!/usr/bin/env bash
# =============================================================================
#  UA-NWM  —  compute ATE / RPE from ALREADY-inferred raw results
# =============================================================================
#  No inference / no GPU. Reads the raw per-case trajectories dumped by eval_ranking.sh or
#  eval_planning.sh (<RESULTS_DIR>/<strategy>/per_case.json) and recomputes the metrics. The
#  setting (offline_ranking vs offline_planning) is auto-detected from the raw schema.
#
#  Usage:  bash scripts/compute_metrics.sh [RESULTS_DIR]
#    RESULTS_DIR defaults to results/airgoal10k/offline_ranking
#  Output: <RESULTS_DIR>/metrics.json  (+ printed table)
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

OUT="${1:-results/airgoal10k/offline_ranking}"
UA="$OUT/UA-NWM/per_case.json"
DET="$OUT/UA-NWM_deterministic/per_case.json"

for f in "$UA" "$DET"; do
  [[ -f "$f" ]] || { echo "FATAL: missing $f — run the eval script first (needs --dump_raw output)"; exit 1; }
done

echo "[metrics] ATE / RPE from $OUT ..."
python uanwm/eval/validate_from_raw.py --settings 8 16 32 \
  --members UA-NWM="$UA" UA-NWM_deterministic="$DET" \
  --baseline UA-NWM_deterministic --out "$OUT/metrics.json"
echo "[metrics] done -> $OUT/metrics.json"
