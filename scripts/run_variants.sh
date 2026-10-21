#!/usr/bin/env bash
# Variants of the detect-attribute model (needs the models of run_main.sh):
#   stgids_mono8     monotone detector with 8 evidence channels + STGIDS attack types
#   mlpadv_mono8     the same detector + attack types from the adversarially trained DNN, which sees
#                    only each flow's own features (injected flows can change neither the detection
#                    nor the attack type of an existing flow)
#   stgids_mono8_or  stgids_mono8, but a flow is also flagged when STGIDS predicts an attack
#   stgids_injaug_mono8  the same detector + attack types from the injection-augmented STGIDS
#                    (stgids_injaug_mono of run_injection.sh with 8 channels; needs stgids_injaug)
# Usage: bash scripts/run_variants.sh configs/nf_unsw_nb15_v3.yaml runs/unsw_a100 [max test windows]
#   env DATA, SET, EVAL as in run_injection.sh; ONLY="<names>" runs only those variants
set -euo pipefail
CFG=${1:?config}; OUT=${2:?output dir}; MAXW=${3:-0}
DATA=${DATA:-$OUT/prepared.pt}
ONLY=${ONLY:-}
read -r -a EXTRA <<< "${SET:-}"
read -r -a EVAL_SET <<< "${EVAL:-eval.attacks=[pgd,square]}"
WIN=(); [ "$MAXW" != 0 ] && WIN=(--max-windows "$MAXW")

run () {  # name, attribution model, detector overrides...
  local name=$1 typer=$2; shift 2
  [[ -z "$ONLY" || " $ONLY " == *" $name "* ]] || return 0
  python -m stgids.train_detector --config "$CFG" --data "$DATA" --typer "$typer" --set output_dir="$OUT/$name" "$@" "${EXTRA[@]}"
  python -m stgids.evaluate       --config "$CFG" --data "$DATA" --set output_dir="$OUT/$name" "${EXTRA[@]}" "${EVAL_SET[@]}"
  local EI=(python -m stgids.evaluate_inject --config "$CFG" --data "$DATA" --set output_dir="$OUT/$name" "${EXTRA[@]}")
  "${EI[@]}" "${WIN[@]}" --betas 20 100                                   # random + adaptive evasion
  "${EI[@]}" "${WIN[@]}" --betas 20 --no-random --feature constrained:0.1 --feature unconstrained:0.1
  "${EI[@]}" "${WIN[@]}" --betas 20 --no-random --objectives untargeted   # wrong attack type
}
run stgids_mono8    "$OUT/stgids/model.pt"               detector.channels=8
run mlpadv_mono8    "$OUT/baselines/model_mlp_adv.pt"    detector.channels=8
run stgids_mono8_or "$OUT/stgids/model.pt"               detector.channels=8 detector.fusion=or
run stgids_injaug_mono8 "$OUT/stgids_injaug/model.pt"    detector.channels=8
