#!/usr/bin/env bash
# Threshold analysis of the IPMD detection score D (paper only): dump per-flow D of the trained
# detect-attribute models, clean (val + test) and under the 220-window attacks, then sweep thresholds.
# Reads runs/<run>/<model>/model.pt; writes ONLY to runs/threshold_sweep/ (new files; existing .npz
# are kept and skipped, so the script can be restarted).
# Usage (A100, from the project folder, venv active):
#   bash scripts/run_threshold_sweep.sh            # all 3 seeds
#   SEEDS="s1" bash scripts/run_threshold_sweep.sh # one seed ("" = seed 42, s1, s2)
set -euo pipefail
OUTROOT=runs/threshold_sweep
MODELS=${MODELS:-"stgids_mono8 mlpadv_mono8 stgids_injaug_mono8 stgids_flowdet stgids_mono8_or"}
SEEDS=${SEEDS:-"s42 s1 s2"}
for s in $SEEDS; do
  if [ "$s" = s42 ]; then RUN=runs/cic2018_full; DATA=runs/cic2018/prepared.pt
  else RUN=runs/cic2018_full_$s; DATA=runs/cic2018_$s/prepared.pt; fi
  [ -f "$DATA" ] || { echo "missing $DATA"; exit 1; }
  for m in $MODELS; do
    [ -f "$RUN/$m/model.pt" ] || { echo "skip $RUN/$m (no model.pt)"; continue; }
    python -m stgids.score_dump --config "$RUN/$m/config.yaml" --data "$DATA" \
      --out "$OUTROOT/$(basename "$RUN")/$m" --max-windows 220 --betas 20 100 --modes hist \
      --feature constrained:0.1 --feature unconstrained:0.1 --feature-betas 20
  done
done
for m in $MODELS; do
  dirs=$(ls -d "$OUTROOT"/cic2018_full*/"$m" 2>/dev/null || true)
  [ -n "$dirs" ] || continue
  csv="$OUTROOT/cic2018_${m}_thresholds.csv"
  [ -f "$csv" ] && { echo "$csv exists, not rewritten"; continue; }
  python scripts/threshold_sweep.py $dirs --out "$csv"
done
