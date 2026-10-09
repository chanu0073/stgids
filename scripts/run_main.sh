#!/usr/bin/env bash
# Full pipeline for one dataset: prepare -> train (clean + robust) -> evaluate -> baselines.
# Usage: bash scripts/run_main.sh configs/nf_unsw_nb15_v3.yaml runs/unsw
set -euo pipefail
CFG=${1:?config}; OUT=${2:?output dir}

python -m stgids.prepare   --config "$CFG" --set output_dir="$OUT"

# proposed model, standard training and with constrained adversarial training
python -m stgids.train     --config "$CFG" --set output_dir="$OUT/stgids"          --data "$OUT/prepared.pt"
python -m stgids.evaluate  --config "$CFG" --set output_dir="$OUT/stgids"          --data "$OUT/prepared.pt"
python -m stgids.train     --config "$CFG" --set output_dir="$OUT/stgids_adv" train.adv.mode=constrained --data "$OUT/prepared.pt"
python -m stgids.evaluate  --config "$CFG" --set output_dir="$OUT/stgids_adv"      --data "$OUT/prepared.pt"

# graph baseline
python -m stgids.train     --config "$CFG" --set output_dir="$OUT/egraphsage" model.arch=egraphsage --data "$OUT/prepared.pt"
python -m stgids.evaluate  --config "$CFG" --set output_dir="$OUT/egraphsage"      --data "$OUT/prepared.pt"

# flow-vector baselines (RF, XGBoost, DNN, DNN+adv training, BiGRU, AE-GRU)
python -m stgids.baselines --config "$CFG" --set output_dir="$OUT" --data "$OUT/prepared.pt"
