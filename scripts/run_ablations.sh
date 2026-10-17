#!/usr/bin/env bash
# Ablations (paper Section 6). Needs <OUT>/prepared.pt from run_main.sh.
# Usage: bash scripts/run_ablations.sh configs/nf_unsw_nb15_v3.yaml runs/unsw
#   env SET="<extra --set overrides for every step>", e.g. SET="train.patience=30"
set -euo pipefail
CFG=${1:?config}; OUT=${2:?output dir}
read -r -a EXTRA <<< "${SET:-}"
# ablations are attacked at one budget (eps = 0.1, the paper's headline setting) to keep the
# whole script to about one night on a GTX 1650; the full eps curve is kept for the main models
EVAL_SET=("eval.eps_list=[0.1]")
run () {  # name, overrides...
  local name=$1; shift
  python -m stgids.train    --config "$CFG" --data "$OUT/prepared.pt" --set output_dir="$OUT/ablation_$name" "$@" "${EXTRA[@]}"
  python -m stgids.evaluate --config "$CFG" --data "$OUT/prepared.pt" --set output_dir="$OUT/ablation_$name" "$@" "${EXTRA[@]}" "${EVAL_SET[@]}"
}
run no_temporal       model.temporal=none
run gru               model.temporal=gru
run loss_ce           train.loss=ce
run loss_weighted_ce  train.loss=weighted_ce
run pretrain          pretrain.enabled=true
run adv_unconstrained train.adv.mode=unconstrained
# window size changes the snapshots, so it needs its own prepare step:
python -m stgids.prepare --config "$CFG" --set output_dir="$OUT/ablation_window_30s" graph.window_seconds=30 "${EXTRA[@]}"
python -m stgids.train    --config "$CFG" --set output_dir="$OUT/ablation_window_30s" graph.window_seconds=30 "${EXTRA[@]}"
python -m stgids.evaluate --config "$CFG" --set output_dir="$OUT/ablation_window_30s" graph.window_seconds=30 "${EXTRA[@]}" "${EVAL_SET[@]}"
