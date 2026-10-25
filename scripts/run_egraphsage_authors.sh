#!/usr/bin/env bash
# E-GraphSAGE with the authors' own code (scripts/egraphsage_authors.py) on our splits, three seeds per dataset,
# into the NEW folders <run>/egraphsage_authors. A finished seed is skipped; nothing is overwritten.
# Needs a separate environment with DGL (DGL has no wheels for torch 2.5), e.g. on the A100:
#   python3 -m venv ~/egs_env && source ~/egs_env/bin/activate
#   pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
#   pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
#   pip install "numpy<2" "pandas<3" scikit-learn pyyaml category_encoders pydantic
#   git clone https://github.com/waimorris/E-GraphSAGE ~/E-GraphSAGE
# Usage: bash scripts/run_egraphsage_authors.sh ~/E-GraphSAGE ~/egs_env/bin/python
#   env CIC_FRACTION=0.1 (share of the CIC training flows; all of them do not fit full-batch training)
set -euo pipefail
REPO=${1:?clone of github.com/waimorris/E-GraphSAGE}; PY=${2:-python}
CIC_FRACTION=${CIC_FRACTION:-0.1}
run () {  # out, args...
  local out=$1; shift
  if [ -f "$out/results_clean_test.json" ]; then echo "skip $out (done)"; return 0; fi
  "$PY" scripts/egraphsage_authors.py --repo "$REPO" --out "$out" "$@"
}
for s in 42 1 2; do
  case $s in
    42) U=runs/unsw_a100;    C=runs/cic2018_full ;;
    1)  U=runs/unsw_a100_s1; C=runs/cic2018_full_s1 ;;
    2)  U=runs/unsw_a100_s2; C=runs/cic2018_full_s2 ;;
  esac
  CHECK=(); [ $s = 42 ] && [ -f runs/unsw_a100/prepared.pt ] && CHECK=(--check runs/unsw_a100/prepared.pt)
  run "$U/egraphsage_authors" --config configs/nf_unsw_nb15_v3.yaml --seed $s "${CHECK[@]}"
  CHECK=(); [ $s = 42 ] && [ -f runs/cic2018/prepared.pt ] && CHECK=(--check runs/cic2018/prepared.pt)
  run "$C/egraphsage_authors" --config configs/nf_cse_cic_ids2018_v3.yaml --seed $s \
    --train-fraction "$CIC_FRACTION" "${CHECK[@]}"
done
