# Cloud ST-GIDS

Code for *Adversarially Robust Spatio-Temporal Graph Learning for Imbalance-Aware
Intrusion Detection in Cloud Networks*.

Pure PyTorch (no PyTorch Geometric needed). Runs on CPU for small tests and on a GPU
for the real datasets.

## What each step of the method maps to

| Step (flowchart) | What it does | File |
|---|---|---|
| 1 Data collection | reads NetFlow-v3 CSVs, checks the required columns exist | `stgids/data.py` |
| 2 Preprocessing | removes duplicates, splits by time (70/15/15), scales and selects features on train only (mutual information + SHAP stability) | `stgids/data.py`, `stgids/features.py`, `stgids/prepare.py` |
| 3 Graph construction | 60 s windows; hosts = nodes; flows = directed edges with flow features | `stgids/prepare.py`, `stgids/graphs.py` |
| 4 Spatial encoder | edge-aware multi-head attention GNN; optional DGI self-supervised pretraining | `stgids/model.py` (`SpatialEncoder`, `DGIHead`) |
| 5 Temporal encoder | Transformer over each host's last K snapshots (GRU / none for ablation) | `stgids/model.py` (`TemporalEncoder`) |
| 6 Classifier | MLP + softmax per flow | `stgids/model.py` (`STGIDS.head`) |
| 7 Robust training | focal loss; constrained PGD adversarial training | `stgids/losses.py`, `stgids/adversarial.py`, `stgids/engine.py` |
| 8 Evaluation | future-traffic test, FGSM/PGD/DeepFool/Square, cross-dataset, baselines, ablations, latency | `stgids/evaluate.py`, `stgids/baselines.py`, `scripts/` |

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python -m pytest tests -q                               # 7 unit tests, a few seconds
```

On Google Colab, `torch`, `pandas` and `scikit-learn` are already installed; run
`pip install xgboost shap`.

## 1. Test the pipeline with synthetic data (about 5 minutes on CPU)

```bash
python -m stgids.synthetic --out data/synthetic_nf_v3.csv
bash scripts/run_main.sh configs/synthetic.yaml runs/syn
python scripts/collect_results.py runs/syn
```

The synthetic data only exercises the code paths. **Never report its numbers.**

## 2. Get the real datasets

Download **NF-UNSW-NB15-v3** and **NF-CSE-CIC-IDS2018-v3** from the University of
Queensland NIDS datasets page and put the CSVs in `data/`. Before the first run:

1. Open the CSV header and check the column names in `configs/*.yaml`
   (`FLOW_START_MILLISECONDS`, `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `Attack`, the benign
   label). If one is wrong, `prepare` stops and suggests the closest column name.
2. Check that `threat_model.controllable` in `stgids/config.py` lists feature names that
   exist in your file. Missing ones are skipped with a warning.
3. NF-CSE-CIC-IDS2018 is large. Test first with `--set data.max_rows=2000000`.

## 3. Full experiments

```bash
# main model, adversarially trained model, E-GraphSAGE, and 6 flow baselines
bash scripts/run_main.sh configs/nf_unsw_nb15_v3.yaml runs/unsw
bash scripts/run_main.sh configs/nf_cse_cic_ids2018_v3.yaml runs/cic2018

# ablations (temporal, loss, pretraining, adversarial-training type, window size)
bash scripts/run_ablations.sh configs/nf_unsw_nb15_v3.yaml runs/unsw

# RQ4: train on UNSW, test on CIC-2018 (and the reverse)
bash scripts/run_cross_dataset.sh configs/nf_cse_cic_ids2018_v3.yaml runs/unsw
bash scripts/run_cross_dataset.sh configs/nf_unsw_nb15_v3.yaml runs/cic2018

# one CSV + LaTeX table per dataset
python scripts/collect_results.py runs/unsw
```

Individual steps can also be run on their own:

```bash
python -m stgids.prepare   --config configs/nf_unsw_nb15_v3.yaml
python -m stgids.train     --config configs/nf_unsw_nb15_v3.yaml --set train.adv.mode=constrained
python -m stgids.evaluate  --config configs/nf_unsw_nb15_v3.yaml
python -m stgids.baselines --config configs/nf_unsw_nb15_v3.yaml --set "baselines.models=[rf,mlp]"
```

Any config value can be changed with `--set section.key=value`. All defaults are
listed, with comments, in `stgids/config.py`.

## If a run stops in the middle

Just run **the same command again**. Every step saves its progress and continues from
where it stopped:

| Step | What is saved | What a rerun does |
|---|---|---|
| `prepare` | `prepared.pt` (+ `prepared.pt.json` with the settings) | skips if the data and settings are unchanged |
| `train` | `checkpoint.pt` after **every epoch** (model, optimiser, scheduler, early-stopping state, random state); `pretrain.pt` after pretraining | continues from the last finished epoch; skips pretraining if done; skips entirely if `model.pt` exists |
| `evaluate` | `results_test.json` after **every attack** | runs only the attacks that are missing |
| `baselines` | `baselines/model_<name>.pt` and `results_<name>.json` after every attack | reuses trained baselines, runs only missing attacks |

So after a crash, power cut or closed terminal, rerunning
`bash scripts/run_main.sh configs/nf_unsw_nb15_v3.yaml runs/unsw` picks up where it stopped.
At most the one epoch (or one attack) that was running is lost.

A resumed run gives the same model as an uninterrupted one on CPU (tested bit-for-bit).
On a GPU, results can differ in the last decimals because some CUDA operations are not
deterministic; that happens between two uninterrupted GPU runs as well.

Safety checks:
* If you rerun into the same folder with **different settings** (e.g. another learning rate)
  or a rebuilt `prepared.pt`, training refuses to mix them and tells you. Use a new
  `output_dir`, or add `--fresh` to start that run from scratch.
* `prepare --force` rebuilds the data; `baselines --fresh` retrains every baseline;
  delete `results_test.json` to redo an evaluation.
* Files are written to a temporary name and renamed, so a crash while saving never leaves a
  broken checkpoint.

## Outputs (per run folder)

| File | Content |
|---|---|
| `prepare.txt` | class counts per split. **Check this first**: a chronological split can leave an attack class out of train or test. |
| `feature_selection.json` | the selected features and each feature's stability score |
| `train_history.json` | loss and validation metrics per epoch |
| `results_clean_test.json` | clean test metrics |
| `results_test.json`, `summary_test.csv` | clean and all attack results (threat × attack × eps) |
| `baselines/summary_baselines.csv` | the same for the flow baselines |
| `model.pt` | final model (best validation epoch) with its config and preprocessing |
| `checkpoint.pt` | last-epoch training state, used to resume |

Metrics are macro-F1, weighted-F1, per-class F1, macro PR-AUC, FPR, detection rate,
binary F1, latency (ms per snapshot) and throughput (flows/s). Classes that appear only
in val or test are marked "unseen" and scored in the binary metrics only.

## Threat model (what the adversarial parts assume)

* **Constrained** (realistic). Only attack flows change, and only the src→dst features
  the attacker controls: bytes, packets, duration, inter-arrival time and longest packet.
  Changes are additive only (padding, extra packets, delays), up to
  `eps × max(value, floor)` and capped at the 1500-byte MTU. Dependent features
  (throughput, bytes/s) are rescaled by the ratio rule. Integers stay integers, and
  `bytes ≥ packets × 40`, `bytes ≥ longest packet`, `duration ≥ duration_in` and
  `IAT_max ≥ IAT_avg` all still hold.
* **Unconstrained** (what most papers report). An L∞ perturbation of size `eps`, in
  standardised units, on every model input of the attack flows, with no validity checks.
* The attacker's goal is **evasion**: get attack flows classified as benign
  (`threat_model.objective: untargeted` is also available).

## Known simplifications (state these in the paper)

* **Dependent features.** Only the ones listed under `threat_model.dependent` are
  recomputed, using a ratio rule. Packet-size histogram bins, `MAX_IP_PKT_LEN` and the
  IAT min/stddev are not updated, so a perturbed flow is not a perfect nProbe record.
  Check the throughput formulas against the nProbe documentation before relying on them.
* **Only the current snapshot is attacked.** Earlier snapshots in the history are left
  unchanged.
* **DeepFool is batched.** Each flow takes a step along its own gradient row, and
  interactions between flows through the graph are ignored when sizing the step.
* **"Square" is a Square-Attack-style score-based random search** adapted to tabular
  features, not the original image-patch algorithm.
* **Budgets are not comparable between the two threat models.** `eps` means a relative
  increase in the constrained model and standard-deviation units in the unconstrained
  one, so compare their trends, not their raw values.
* **Feature selection can drop controllable features.** That makes the model harder to
  attack "for free". `evaluate.py` logs which controllable features were kept; report it.
* **Baselines are re-implemented from each paper's description**, not taken from the
  authors' code: `bigru` (MSBi-GRU-style), `aegru` (FFAE-GRU-style), `mlp` / `mlp_adv`
  (Holla-style DNN and its adversarial training), and `egraphsage`. ICloud has public
  code, so run it on the same split instead of re-implementing it. ST-GNN publishes no
  code; the `model.temporal` ablations cover the ST-GNN design choice.

## Memory and speed

* Each 60 s snapshot is one graph. If a window of CIC-2018 DDoS traffic is too big for
  GPU memory, use `graph.window_seconds: 30`.
* `model.detach_history: true` (the default) runs past snapshots without gradients.
  It is faster and uses less memory.
* Attack evaluation is the slowest step. `eval.max_windows` limits how many test windows
  are attacked; the flow baselines use the same windows.
* Fix `seed` and run at least 3 seeds for the final tables (`--set seed=1`, `seed=2`, …).
