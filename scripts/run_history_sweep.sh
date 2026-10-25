#!/usr/bin/env bash
# History-length sweep: does STGIDS need more than the last 6 windows of history?
# For every K in KS, STGIDS is trained with K windows of history (K = 1: the current window only) into
# the NEW folder <OUT>/history_k<K>, tested on clean traffic (by stgids.train) and attacked with flow
# injection into the current window ("now") and into all K windows ("hist", history shaping).
# K = 6 is the main model <OUT>/stgids and is reused unless KS contains 6 (then it goes to history_k6).
# Needs <OUT>/prepared.pt from run_main.sh. Nothing is overwritten: a finished model is never retrained,
# and the injection results resume per setting. Summarise with scripts/aggregate_history.py.
# Usage: bash scripts/run_history_sweep.sh configs/nf_unsw_nb15_v3.yaml runs/unsw_a100 [max test windows]
#   env KS="1 3 12 24" (history lengths), BETAS="20 100" (injected flows per attacker host and window),
#       DATA=<prepared.pt> (default <OUT>/prepared.pt), SET="<extra --set overrides for every step>",
#       EVAL="<--set overrides>" also runs stgids.evaluate (feature attacks), e.g.
#       EVAL="eval.attacks=[pgd,square] eval.eps_list=[0.1]"; empty (default) = clean results only
# A100, same protocol as the main runs (graph models 30 epochs, best validation epoch):
#   SET="train.patience=30" bash scripts/run_history_sweep.sh configs/nf_unsw_nb15_v3.yaml runs/unsw_a100
#   SET="train.patience=30" bash scripts/run_history_sweep.sh configs/unsw_s1.yaml runs/unsw_a100_s1
#   SET="train.patience=30" bash scripts/run_history_sweep.sh configs/unsw_s2.yaml runs/unsw_a100_s2
set -euo pipefail
CFG=${1:?config}; OUT=${2:?output dir}; MAXW=${3:-0}
DATA=${DATA:-$OUT/prepared.pt}
KS=${KS:-1 3 12 24}
read -r -a EXTRA <<< "${SET:-}"
read -r -a EVAL_SET <<< "${EVAL:-}"
read -r -a BETA_LIST <<< "${BETAS:-20 100}"
WIN=(); [ "$MAXW" != 0 ] && WIN=(--max-windows "$MAXW")
[ -f "$DATA" ] || { echo "no $DATA (run run_main.sh first or set DATA)"; exit 1; }
for x in "${EXTRA[@]}" "${EVAL_SET[@]}"; do  # the folder and K are set by this script only
  case $x in output_dir=*|graph.history_k=*) echo "do not put $x in SET or EVAL"; exit 1 ;; esac
done

for K in $KS; do
  D="$OUT/history_k$K"
  # one --set for all overrides: a second --set would silently drop the first one, output_dir included
  OV=(--set output_dir="$D" graph.history_k="$K" "${EXTRA[@]}")
  if [ -f "$D/model.pt" ]; then
    echo "history_k$K: model.pt exists, not retrained"
  else
    python -m stgids.train --config "$CFG" --data "$DATA" "${OV[@]}"
  fi
  grep -qxF "output_dir: $D" "$D/config.yaml" && grep -qxF "  history_k: $K" "$D/config.yaml" \
    || { echo "history_k$K: $D/config.yaml has a different output_dir or history_k, stopping"; exit 1; }
  if [ ${#EVAL_SET[@]} -gt 0 ]; then
    python -m stgids.evaluate --config "$CFG" --data "$DATA" "${OV[@]}" "${EVAL_SET[@]}"
  fi
  MODES=(now hist); [ "$K" = 1 ] && MODES=(now)  # with K = 1 there are no earlier windows to shape
  python -m stgids.evaluate_inject --config "$CFG" --data "$DATA" "${OV[@]}" "${WIN[@]}" \
    --betas "${BETA_LIST[@]}" --modes "${MODES[@]}"
done
