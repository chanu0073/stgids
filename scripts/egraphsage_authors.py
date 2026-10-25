"""E-GraphSAGE run with the authors' own code on our split: the most faithful E-GraphSAGE baseline.

The model classes (SAGELayer, SAGE, MLPPredictor, Model) are executed unchanged from the authors' notebook
in a local clone of https://github.com/waimorris/E-GraphSAGE (Apache-2.0; Lo et al., NOMS 2022), and the
preprocessing, graph construction and training follow the same notebook
(E-GraphSAGE/netflow/ton-iot/Unsw_ton_iot_multiclass_mean_agg.ipynb, cells 3-45). What differs, and why:
  * split: our chronological or blocked split of the same deduplicated flows instead of a random
    stratified 60/40 split - this is the point of the run (check it with --check <prepared.pt>);
  * columns: our dropped columns and the flow timestamps are not used as features (NetFlow v3 has start
    and end times, which identify the attack episode); missing values are set to 0 as in our pipeline;
  * class weights: sklearn "balanced" weights of our training labels (the notebook hard-codes the balanced
    weights of NF-ToN-IoT), and the output layer has our number of classes (the notebook hard-codes 10);
  * graphs are built directly in DGL with every flow in both directions - the same graph as
    nx.MultiGraph(...).to_directed() in the notebook, but in flow order, so every test flow gets exactly one
    prediction (its own direction); the notebook scores both directions of every flow;
  * testing in model.eval() and torch.no_grad() (the notebook leaves dropout on at test time), scored with
    our metrics (macro-F1, detection rate, FPR) on all of our test flows;
  * --train-fraction < 1 trains on a random subset of the training flows, as the authors did for very large
    datasets (10% of BoT-IoT); the default uses all training flows.
Needs dgl and category_encoders (a separate environment is recommended, see docs/CLAUDE.md).

Usage:
    python scripts/egraphsage_authors.py --config configs/nf_unsw_nb15_v3.yaml --repo ~/E-GraphSAGE \
        --out runs/unsw_a100/egraphsage_authors --seed 42 --check runs/unsw_a100/prepared.pt
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stgids import data as D  # noqa: E402
from stgids.config import load_config  # noqa: E402
from stgids.metrics import compute_metrics  # noqa: E402

NOTEBOOK = "E-GraphSAGE/netflow/ton-iot/Unsw_ton_iot_multiclass_mean_agg.ipynb"
MODEL_CELLS = ("class SAGELayer", "class MLPPredictor", "class Model(")


def load_authors_model(repo: Path) -> dict:
    """Execute the model-class cells of the authors' notebook unchanged and return their namespace."""
    nb = json.loads((repo / NOTEBOOK).read_text())
    ns: dict = {}
    exec("import torch as th\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
         "import dgl.function as fn", ns)
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and src.lstrip().startswith(MODEL_CELLS):
            exec(compile(src, str(repo / NOTEBOOK), "exec"), ns)
    missing = {"SAGELayer", "SAGE", "MLPPredictor", "Model"} - set(ns)
    if missing:
        raise SystemExit(f"model classes {sorted(missing)} not found in {repo / NOTEBOOK}")
    return ns


def build_graph(src_keys: pd.Series, dst_keys: pd.Series, x: np.ndarray, y: np.ndarray):
    """Every flow as an edge in both directions (= nx.MultiGraph(...).to_directed()); first half = flow order."""
    import dgl
    codes, uniques = pd.factorize(pd.concat([src_keys, dst_keys], ignore_index=True))
    n = len(src_keys)
    s, d = codes[:n], codes[n:]
    g = dgl.graph((th.from_numpy(np.concatenate([s, d])), th.from_numpy(np.concatenate([d, s]))),
                  num_nodes=len(uniques))
    g.edata["h"] = th.from_numpy(np.concatenate([x, x])).reshape(2 * n, 1, x.shape[1])  # notebook cell 29
    g.edata["label"] = th.from_numpy(np.concatenate([y, y]))
    g.ndata["h"] = th.ones(g.num_nodes(), 1, x.shape[1])  # notebook cell 23 ("Eq1") + 29
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="our dataset config (split, columns, label names)")
    p.add_argument("--csv", help="dataset CSV (default: data.csv of the config)")
    p.add_argument("--repo", required=True, help="local clone of github.com/waimorris/E-GraphSAGE")
    p.add_argument("--out", required=True, help="NEW output folder")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=4999, help="notebook: range(1, 5000) = 4999 steps")
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--check", help="our prepared.pt: verify that the split and labels are identical")
    a = p.parse_args()

    out = Path(a.out)
    if (out / "results_clean_test.json").exists():
        raise SystemExit(f"{out}/results_clean_test.json exists - choose a new --out (nothing is overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "run.log", "a")

    def log(msg: str):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    import category_encoders as ce
    import dgl
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    random.seed(a.seed)
    np.random.seed(a.seed)
    th.manual_seed(a.seed)
    dev = th.device("cuda" if th.cuda.is_available() else "cpu")
    repo = Path(a.repo).expanduser()
    ns = load_authors_model(repo)

    # ---- our flows and split (same functions as stgids.prepare) ----
    cfg = load_config(a.config, [f"data.csv={a.csv}"] if a.csv else None)
    dcfg = cfg["data"]
    df = D.clean(D.load_flows(dcfg), dcfg)
    window = D.assign_windows(df["_t_ms"].to_numpy(), cfg["graph"]["window_seconds"])
    split = D.chronological_split(window, dcfg["split"], dcfg.get("split_block_windows"))
    classes = D.build_label_space(df.loc[split == 0, dcfg["label_col"]], dcfg["benign_label"])
    y_ours, y_bin = D.encode_labels(df[dcfg["label_col"]], classes, dcfg["benign_label"])
    log(f"{len(df):,} flows | train {int((split == 0).sum()):,} | val {int((split == 1).sum()):,} | "
        f"test {int((split == 2).sum()):,} | {len(classes)} classes")
    if a.check:
        b = th.load(a.check, weights_only=False)
        sp = np.repeat(b["snap_split"].numpy(), np.diff(b["snap_ptr"].numpy()))
        if not (len(sp) == len(split) and np.array_equal(sp, split) and np.array_equal(b["y"].numpy(), y_ours)):
            raise SystemExit(f"split or labels differ from {a.check} - stopping")
        log(f"split and labels identical to {a.check}")
        del b, sp

    tr = np.flatnonzero(split == 0)
    if a.train_fraction < 1:
        tr = np.sort(np.random.default_rng(a.seed).choice(tr, size=int(len(tr) * a.train_fraction), replace=False))
    te = np.flatnonzero(split == 2)
    rows = np.concatenate([tr, te])
    d = df.iloc[rows].reset_index(drop=True)
    n_tr = len(tr)
    del df

    # ---- notebook cells 3-6: random source IP, nodes = IP:port ----
    rnd_ip = [socket.inet_ntoa(struct.pack(">I", random.randint(0xac100001, 0xac1f0001))) for _ in range(len(d))]
    src = pd.Series(rnd_ip) + ":" + d[dcfg["src_port_col"]].astype(str)
    dst = d[dcfg["dst_ip_col"]].astype(str) + ":" + d[dcfg["dst_port_col"]].astype(str)

    # ---- cells 8-12: labels (LabelEncoder over the label names, as in the notebook) ----
    le = LabelEncoder().fit(d[dcfg["label_col"]].astype(str))
    lab = le.transform(d[dcfg["label_col"]].astype(str)).astype(np.int64)

    # ---- cells 17-19: target encoding and scaling, fitted on the training flows ----
    exclude = {dcfg["src_ip_col"], dcfg["dst_ip_col"], dcfg["src_port_col"], dcfg["dst_port_col"],
               dcfg["label_col"], dcfg["time_col"], "_t_ms", *dcfg.get("drop_cols", [])}
    feat = sorted(c for c in d.columns if c not in exclude)
    xdf = d[feat].apply(pd.to_numeric, errors="coerce").fillna(0)
    te_cols = [c for c in ("TCP_FLAGS", "L7_PROTO", "PROTOCOL") if c in feat]
    enc = ce.TargetEncoder(cols=te_cols, min_samples_leaf=1, smoothing=1.0)  # old library defaults
    enc.fit(xdf.iloc[:n_tr], lab[:n_tr])
    xdf = enc.transform(xdf)
    scaler = StandardScaler().fit(xdf.iloc[:n_tr])
    x = scaler.transform(xdf).astype(np.float32)
    log(f"{len(feat)} edge features (target-encoded: {te_cols}); training flows used: {n_tr:,}")

    # ---- cells 20-24 / 29: training graph ----
    g = build_graph(src.iloc[:n_tr], dst.iloc[:n_tr], x[:n_tr], lab[:n_tr]).to(dev)
    log(f"training graph: {g.num_nodes():,} nodes, {g.num_edges():,} directed edges")

    # ---- cells 30-36: model, weighted cross-entropy, Adam, full-batch training ----
    n_cls = len(le.classes_)
    w = compute_class_weight("balanced", classes=np.arange(n_cls), y=lab[:n_tr])
    criterion = th.nn.CrossEntropyLoss(weight=th.tensor(w, dtype=th.float32, device=dev))
    nf = x.shape[1]
    model = ns["Model"](nf, 128, nf, th.nn.functional.relu, 0.2)
    if n_cls != 10:  # the notebook's Model hard-codes MLPPredictor(ndim_out, 10)
        model.pred = ns["MLPPredictor"](128, n_cls)
    model = model.to(dev)
    opt = th.optim.Adam(model.parameters())
    node_features, edge_features, edge_label = g.ndata["h"], g.edata["h"], g.edata["label"]
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        pred = model(g, node_features, edge_features)
        loss = criterion(pred, edge_label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == a.epochs:
            acc = (pred.argmax(1) == edge_label).float().mean().item()
            log(f"epoch {epoch} | loss {loss.item():.4f} | training acc {acc:.4f}")
    train_s = time.time() - t0
    del g, node_features, edge_features, edge_label, pred
    if dev.type == "cuda":
        th.cuda.empty_cache()

    # ---- cells 37-50: test graph of the test flows, one prediction per flow ----
    gt = build_graph(src.iloc[n_tr:], dst.iloc[n_tr:], x[n_tr:], lab[n_tr:]).to(dev)
    model.eval()
    with th.no_grad():
        logits = model(gt, gt.ndata["h"], gt.edata["h"])[: len(te)]
    pred_names = le.inverse_transform(logits.argmax(1).cpu().numpy())
    index = {c: i for i, c in enumerate(classes)}
    y_pred = np.array([index.get(str(c), -1) for c in pred_names])
    m = compute_metrics(y_ours[te], y_bin[te], y_pred, None, classes)
    log(f"TEST: macro-F1 {m['macro_f1']:.4f} | accuracy {m['accuracy']:.4f} | "
        f"DR {m['detection_rate']:.4f} | FPR {m['fpr']:.4f}")

    def git_head(path: Path) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None

    m["info"] = {"model": "E-GraphSAGE, authors' code", "repo_commit": git_head(repo), "notebook": NOTEBOOK,
                 "seed": a.seed, "epochs": a.epochs, "train_fraction": a.train_fraction,
                 "train_flows": int(n_tr), "test_flows": int(len(te)), "features": feat,
                 "target_encoded": te_cols, "train_seconds": round(train_s, 1),
                 "versions": {"torch": th.__version__, "dgl": dgl.__version__,
                              "category_encoders": ce.__version__, "python": platform.python_version()}}
    (out / "results_clean_test.json").write_text(json.dumps(m, indent=2))
    log(f"saved {out / 'results_clean_test.json'}")


if __name__ == "__main__":
    main()
