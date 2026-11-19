#!/usr/bin/env bash
# Hard-zero ablation of IPMD (paper only): stgids_mono8_hardzero = stgids_mono8 (8 evidence channels,
# STGIDS attack types), but flows with own score <= 0 add nothing to the host statistics.
# For each CIC seed: train + evaluate + the 19 injection settings of stgids_mono8, with the same
# options (run_variants.sh, plus the beta=100 wrong-type run added later), then the per-flow scores
# for scripts/threshold_sweep.py and scripts/score_rise.py.
# Writes ONLY new folders: runs/<run>/stgids_mono8_hardzero and runs/threshold_sweep/<run>/stgids_mono8_hardzero
# (all steps resume after an interruption). Usage (A100, project folder, venv active):
#   nohup bash scripts/run_hardzero.sh > runs/hardzero.log 2>&1 &
#   SEEDS="s1" bash scripts/run_hardzero.sh     # one seed ("s42", "s1", "s2")
set -euo pipefail
CFG=configs/nf_cse_cic_ids2018_v3.yaml
NAME=stgids_mono8_hardzero
SEEDS=${SEEDS:-"s42 s1 s2"}
STAMP=$(mktemp)
for s in $SEEDS; do
  if [ "$s" = s42 ]; then RUN=runs/cic2018_full; DATA=runs/cic2018/prepared.pt; SET="train.patience=30"
  else RUN=runs/cic2018_full_$s; DATA=runs/cic2018_$s/prepared.pt; SET="seed=${s#s} train.patience=30"; fi
  [ -f "$DATA" ] && [ -f "$RUN/stgids/model.pt" ] || { echo "missing $DATA or $RUN/stgids/model.pt"; exit 1; }
  echo "== $s: $RUN/$NAME (data $DATA, overrides: $SET)"
  DATA="$DATA" SET="$SET" ONLY="$NAME" bash scripts/run_variants.sh "$CFG" "$RUN" 220
  read -r -a EXTRA <<< "$SET"
  python -m stgids.evaluate_inject --config "$CFG" --data "$DATA" --set output_dir="$RUN/$NAME" "${EXTRA[@]}" \
    --max-windows 220 --betas 100 --no-random --objectives untargeted       # wrong type, beta = 100
  python -m stgids.score_dump --config "$RUN/$NAME/config.yaml" --data "$DATA" \
    --out "runs/threshold_sweep/$(basename "$RUN")/$NAME" --max-windows 220 --betas 20 100 --modes hist \
    --feature constrained:0.1 --feature unconstrained:0.1 --feature-betas 20
done
csv=runs/threshold_sweep/cic2018_${NAME}_thresholds.csv
if [ -f "$csv" ]; then echo "$csv exists, not rewritten"
else python scripts/threshold_sweep.py runs/threshold_sweep/cic2018_full*/$NAME --out "$csv"; fi
echo "files changed under runs/ outside the new folders (must be empty):"
find runs -newer "$STAMP" -type f | grep -v -e "/$NAME/" -e "^runs/threshold_sweep" -e "^runs/hardzero.log$" || true
rm -f "$STAMP"
