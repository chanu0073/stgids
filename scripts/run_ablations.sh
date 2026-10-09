#!/usr/bin/env bash
# Ablations (paper Section 6). Needs <OUT>/prepared.pt from run_main.sh.
# Usage: bash scripts/run_ablations.sh configs/nf_unsw_nb15_v3.yaml runs/unsw
set -euo pipefail
CFG=${1:?config}; OUT=${2:?output dir}
run () {  # name, overrides...
  local name=$1; shift
  python -m stgids.train    --config "$CFG" --data "$OUT/prepared.pt" --set output_dir="$OUT/ablation_$name" "$@"
  python -m stgids.evaluate --config "$CFG" --data "$OUT/prepared.pt" --set output_dir="$OUT/ablation_$name" "$@"
}
run no_temporal       model.temporal=none
run gru               model.temporal=gru
run loss_ce           train.loss=ce
run loss_weighted_ce  train.loss=weighted_ce
run pretrain          pretrain.enabled=true
run adv_unconstrained train.adv.mode=unconstrained
# window size changes the snapshots, so it needs its own prepare step:
python -m stgids.prepare --config "$CFG" --set output_dir="$OUT/ablation_window_30s" graph.window_seconds=30
python -m stgids.train    --config "$CFG" --set output_dir="$OUT/ablation_window_30s" graph.window_seconds=30
python -m stgids.evaluate --config "$CFG" --set output_dir="$OUT/ablation_window_30s" graph.window_seconds=30
