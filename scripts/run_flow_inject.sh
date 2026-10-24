#!/usr/bin/env bash
# Flow-vector baselines in the injection study: wrap each trained neural flow baseline with
# stgids.flow_checkpoint and attack it with exactly the injection and joint attacks of
# run_injection.sh (same windows, same injected flows). Added flows cannot change the predictions
# of a flow model, so the injection-only results must equal its clean result on these windows,
# and a joint attack reduces to its feature attack. Needs the baselines of run_main.sh.
# Usage: bash scripts/run_flow_inject.sh configs/nf_unsw_nb15_v3.yaml runs/unsw [max test windows]
#   env DATA=<prepared.pt> (default <OUT>/prepared.pt), SET="<extra --set overrides>",
#       NETS="<flow baselines>" (default "mlp_adv mlp")
# Results: <OUT>/flow_<net>/results_inject.json (new folders; existing files are never overwritten).
# Example (CIC-2018, seed 1):
#   DATA=runs/cic2018_s1/prepared.pt SET="seed=1 train.patience=30" \
#     bash scripts/run_flow_inject.sh configs/nf_cse_cic_ids2018_v3.yaml runs/cic2018_full_s1 220
set -euo pipefail
CFG=${1:?config}; OUT=${2:?output dir}; MAXW=${3:-0}
DATA=${DATA:-$OUT/prepared.pt}
read -r -a EXTRA <<< "${SET:-}"
read -r -a NET_LIST <<< "${NETS:-mlp_adv mlp}"
WIN=(); [ "$MAXW" != 0 ] && WIN=(--max-windows "$MAXW")

for net in "${NET_LIST[@]}"; do
  M="$OUT/flow_$net"
  [ -f "$M/model.pt" ] || python -m stgids.flow_checkpoint --config "$CFG" --data "$DATA" --run "$OUT" --net "$net" --set "${EXTRA[@]}"
  EI=(python -m stgids.evaluate_inject --config "$CFG" --data "$DATA" --set output_dir="$M" "${EXTRA[@]}")
  "${EI[@]}" "${WIN[@]}" --betas 5 20 100                                   # random + adaptive evasion
  "${EI[@]}" "${WIN[@]}" --betas 20 --no-random --feature constrained:0.1 --feature unconstrained:0.1
  "${EI[@]}" "${WIN[@]}" --betas 20 --no-random --objectives untargeted     # wrong attack type
done
