#!/usr/bin/env bash
# Full pipeline for one dataset: prepare -> train (clean + robust) -> evaluate -> baselines.
# Usage: bash scripts/run_main.sh configs/nf_unsw_nb15_v3.yaml runs/unsw
#   env SET="<extra --set overrides for every step>", e.g. SET="train.patience=30"
set -euo pipefail
CFG=${1:?config}; OUT=${2:?output dir}
read -r -a EXTRA <<< "${SET:-}"

python -m stgids.prepare   --config "$CFG" --set output_dir="$OUT" "${EXTRA[@]}"

# proposed model, standard training and with constrained adversarial training
python -m stgids.train     --config "$CFG" --set output_dir="$OUT/stgids" "${EXTRA[@]}"          --data "$OUT/prepared.pt"
python -m stgids.evaluate  --config "$CFG" --set output_dir="$OUT/stgids" "${EXTRA[@]}"          --data "$OUT/prepared.pt"
python -m stgids.train     --config "$CFG" --set output_dir="$OUT/stgids_adv" train.adv.mode=constrained "${EXTRA[@]}" --data "$OUT/prepared.pt"
python -m stgids.evaluate  --config "$CFG" --set output_dir="$OUT/stgids_adv" "${EXTRA[@]}"      --data "$OUT/prepared.pt"

# graph baseline
python -m stgids.train     --config "$CFG" --set output_dir="$OUT/egraphsage" model.arch=egraphsage "${EXTRA[@]}" --data "$OUT/prepared.pt"
python -m stgids.evaluate  --config "$CFG" --set output_dir="$OUT/egraphsage" "${EXTRA[@]}"      --data "$OUT/prepared.pt"

# flow-vector baselines (RF, XGBoost, DNN, DNN+adv training, BiGRU, AE-GRU)
python -m stgids.baselines --config "$CFG" --set output_dir="$OUT" "${EXTRA[@]}" --data "$OUT/prepared.pt"
