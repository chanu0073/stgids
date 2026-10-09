#!/usr/bin/env bash
# RQ4: train on dataset A, test on dataset B (binary metrics, since attack classes differ).
# Usage: bash scripts/run_cross_dataset.sh configs/nf_cse_cic_ids2018_v3.yaml runs/unsw
set -euo pipefail
CFG_B=${1:?config of the test dataset}; OUT_A=${2:?run dir of the trained dataset}
python -m stgids.prepare --config "$CFG_B" --reference "$OUT_A/prepared.pt" --out "$OUT_A/cross_test.pt"
for m in stgids stgids_adv egraphsage; do
  python -m stgids.evaluate --config "$OUT_A/$m/config.yaml" --data "$OUT_A/cross_test.pt" --tag cross --no-attacks
done
