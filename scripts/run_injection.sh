#!/usr/bin/env bash
# Injection study: injection-proof detect-attribute models, the injection-augmentation baseline
# defence, and flow-injection / history-shaping attacks on every graph model of one run.
# Needs the models of run_main.sh (and ablation_no_temporal of run_ablations.sh, if present).
# Usage: bash scripts/run_injection.sh configs/nf_unsw_nb15_v3.yaml runs/unsw [max test windows]
#   env DATA=<prepared.pt> (default <OUT>/prepared.pt), SET="<extra --set overrides>",
#       EVAL="<--set overrides for the feature attacks of the new models>" (default PGD + Square, all eps)
# Example (CIC-2018, seed 1):
#   DATA=runs/cic2018_s1/prepared.pt SET="seed=1 train.patience=30" \
#     bash scripts/run_injection.sh configs/nf_cse_cic_ids2018_v3.yaml runs/cic2018_full_s1 220
set -euo pipefail
CFG=${1:?config}; OUT=${2:?output dir}; MAXW=${3:-0}
DATA=${DATA:-$OUT/prepared.pt}
read -r -a EXTRA <<< "${SET:-}"
WIN=(); [ "$MAXW" != 0 ] && WIN=(--max-windows "$MAXW")
# feature attacks on the new models: PGD and Square (FGSM / DeepFool skipped for time)
read -r -a EVAL_SET <<< "${EVAL:-eval.attacks=[pgd,square]}"

# 1) detect-attribute models: injection-proof detector + trained STGIDS as attribution model
python -m stgids.train_detector --config "$CFG" --data "$DATA" --typer "$OUT/stgids/model.pt" --set output_dir="$OUT/stgids_mono" "${EXTRA[@]}"
python -m stgids.evaluate       --config "$CFG" --data "$DATA" --set output_dir="$OUT/stgids_mono" "${EXTRA[@]}" "${EVAL_SET[@]}"
# ablation: the same detector without host context (flow features only)
python -m stgids.train_detector --config "$CFG" --data "$DATA" --typer "$OUT/stgids/model.pt" --set output_dir="$OUT/stgids_flowdet" detector.context=false "${EXTRA[@]}"
python -m stgids.evaluate       --config "$CFG" --data "$DATA" --set output_dir="$OUT/stgids_flowdet" "${EXTRA[@]}" "${EVAL_SET[@]}"

# 2) baseline defence: injection-augmented training of STGIDS (as in Galli et al., but with history)
python -m stgids.train    --config "$CFG" --data "$DATA" --set output_dir="$OUT/stgids_injaug" "train.inject_aug={beta_max: 20, mode: hist, prob: 0.5}" "${EXTRA[@]}"
python -m stgids.evaluate --config "$CFG" --data "$DATA" --set output_dir="$OUT/stgids_injaug" "${EXTRA[@]}" "${EVAL_SET[@]}"
# combination: monotone detector (certified detection) + injection-augmented STGIDS as attribution model
python -m stgids.train_detector --config "$CFG" --data "$DATA" --typer "$OUT/stgids_injaug/model.pt" --set output_dir="$OUT/stgids_injaug_mono" "${EXTRA[@]}"
python -m stgids.evaluate       --config "$CFG" --data "$DATA" --set output_dir="$OUT/stgids_injaug_mono" "${EXTRA[@]}" "${EVAL_SET[@]}"

# 3) injection attacks (results accumulate in <model>/results_inject.json)
for m in stgids stgids_adv ablation_no_temporal egraphsage stgids_injaug stgids_mono stgids_flowdet stgids_injaug_mono; do
  [ -f "$OUT/$m/model.pt" ] || { echo "skip $m (no model.pt)"; continue; }
  EI=(python -m stgids.evaluate_inject --config "$CFG" --data "$DATA" --set output_dir="$OUT/$m" "${EXTRA[@]}")
  "${EI[@]}" "${WIN[@]}" --betas 5 20 100                                   # random + adaptive evasion
  "${EI[@]}" "${WIN[@]}" --betas 20 --no-random --feature constrained:0.1 --feature unconstrained:0.1
  case $m in  # wrong attack type (untargeted): only where detection itself resists injection
    stgids|stgids_injaug|stgids_mono|stgids_flowdet|stgids_injaug_mono)
      "${EI[@]}" "${WIN[@]}" --betas 20 --no-random --objectives untargeted ;;
  esac
done
