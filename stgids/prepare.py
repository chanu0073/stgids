"""Step 1-3: load flows, clean, split by time, encode, select features, cut into snapshots.

Usage:
    python -m stgids.prepare --config configs/nf_unsw_nb15_v3.yaml
    # cross-dataset: reuse the preprocessing fitted on another dataset, all data becomes test
    python -m stgids.prepare --config configs/nf_cse_cic_ids2018_v3.yaml \
        --reference runs/unsw/prepared.pt --out runs/unsw/cross_cic2018.pt
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D
from .config import load_config, save_config
from .features import select_features
from .utils import LOG, atomic_save, base_parser, file_id, fingerprint, save_json, set_seed, setup_logging


def node_keys(df: pd.DataFrame, dcfg: dict, gcfg: dict) -> tuple[np.ndarray, np.ndarray, int]:
    src = df[dcfg["src_ip_col"]].astype(str)
    dst = df[dcfg["dst_ip_col"]].astype(str)
    if gcfg["node_key"] == "ip_port":
        src = src + ":" + df[dcfg["src_port_col"]].astype(str)
        dst = dst + ":" + df[dcfg["dst_port_col"]].astype(str)
    codes, uniques = pd.factorize(pd.concat([src, dst], ignore_index=True))
    n = len(df)
    return codes[:n].astype(np.int64), codes[n:].astype(np.int64), len(uniques)


def _t(a: np.ndarray) -> torch.Tensor:
    a = np.ascontiguousarray(a)
    return torch.from_numpy(a if a.flags.writeable else a.copy())


def snapshot_pointers(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    change = np.flatnonzero(np.diff(window)) + 1
    ptr = np.concatenate([[0], change, [len(window)]]).astype(np.int64)
    return ptr, window[ptr[:-1]]


def prepare(cfg: dict, reference: str | None = None, out: str | None = None) -> Path:
    dcfg, gcfg = cfg["data"], cfg["graph"]
    out_dir = Path(cfg["output_dir"])
    out_path = Path(out) if out else out_dir / "prepared.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = D.clean(D.load_flows(dcfg), dcfg)
    window = D.assign_windows(df["_t_ms"].to_numpy(), gcfg["window_seconds"])

    if reference is None:
        split = D.chronological_split(window, dcfg["split"])
        tr = split == 0
        num_names, tcp_cols, cat_cols = D.column_groups(df, dcfg)
        enc = D.CategoricalEncoder(tcp_cols, cat_cols, dcfg["cat_top_k"]).fit(df[tr])
        classes = D.build_label_space(df.loc[tr, dcfg["label_col"]], dcfg["benign_label"])
    else:
        ref = torch.load(reference, weights_only=False)["meta"]
        LOG.info(f"reusing preprocessing from {reference} (whole dataset used as test)")
        split = np.full(len(df), 2, dtype=np.int64)
        num_names = ref["num_names"]
        missing = [c for c in num_names if c not in df.columns]
        if missing:
            LOG.warning(f"numeric columns missing in this dataset, filled with 0: {missing}")
        enc = D.CategoricalEncoder.from_state(ref["cat_encoder"])
        for c in enc.tcp_cols + enc.cat_cols:
            if c not in df.columns:
                df[c] = 0
        classes = ref["classes"]
        if dcfg["benign_label"] != ref["benign"]:
            lab = df[dcfg["label_col"]].astype(str)
            df[dcfg["label_col"]] = lab.where(lab != dcfg["benign_label"], ref["benign"])
            dcfg = {**dcfg, "benign_label": ref["benign"]}

    labels = df[dcfg["label_col"]]
    class_counts = D.report_class_counts(labels, split, classes)
    y, y_bin = D.encode_labels(labels, classes, dcfg["benign_label"])
    num = D.numeric_matrix(df, num_names)
    cat = enc.transform(df)
    input_names = num_names + enc.names

    if reference is None:
        tr_idx = np.flatnonzero(split == 0)
        rng = np.random.default_rng(cfg["seed"])
        log_mask = (num[tr_idx].min(axis=0) >= 0) if cfg["features"]["log_transform"] \
            else np.zeros(num.shape[1], bool)
        # scaling statistics from a random sample of training rows (keeps RAM low)
        stat_idx = np.sort(rng.choice(tr_idx, size=min(len(tr_idx), 500_000), replace=False))
        stats = D.fit_transform_stats(num[stat_idx], cat[stat_idx], log_mask)
        scfg = cfg["features"]["selection"]
        if scfg["enabled"]:
            sel_idx = np.sort(rng.choice(tr_idx, size=min(len(tr_idx), 200_000), replace=False))
            x_tr = D.apply_transform_np(num[sel_idx], cat[sel_idx], stats["log_mask"], stats["mu"], stats["sigma"])
            sel = select_features(x_tr, y[sel_idx], input_names, scfg, seed=cfg["seed"])
            del x_tr
        else:
            sel = {"selected": list(range(len(input_names))), "stability": {}}
        transform = {**{k: np.asarray(v) for k, v in stats.items()}, "selected": np.asarray(sel["selected"])}
        save_json({"selected": [input_names[i] for i in sel["selected"]],
                   "stability": sel["stability"]}, out_dir / "feature_selection.json")
    else:
        transform = ref["transform"]

    src, dst, n_hosts = node_keys(df, dcfg, gcfg)
    del df, labels
    gc.collect()
    ptr, snap_window = snapshot_pointers(window)
    snap_split = split[ptr[:-1]]
    sizes = np.diff(ptr)
    LOG.info(f"{len(sizes):,} snapshots of {gcfg['window_seconds']} s | flows per snapshot: "
             f"median {int(np.median(sizes))}, max {int(sizes.max())} | hosts: {n_hosts:,}")
    for s, name in enumerate(D.SPLIT_NAMES):
        LOG.info(f"  {name}: {(snap_split == s).sum():,} snapshots, {(split == s).sum():,} flows")

    meta = {
        "dataset": cfg["name"], "num_names": num_names, "cat_encoder": enc.state(),
        "input_names": input_names, "classes": classes, "benign": dcfg["benign_label"],
        "transform": transform, "class_counts": class_counts, "n_hosts": n_hosts,
        "window_seconds": gcfg["window_seconds"], "node_key": gcfg["node_key"],
    }
    atomic_save({
        "num_raw": _t(num), "cat": _t(cat), "y": _t(y), "y_bin": _t(y_bin),
        "src": _t(src), "dst": _t(dst), "snap_ptr": _t(ptr), "snap_window": _t(snap_window),
        "snap_split": _t(snap_split), "meta": meta,
    }, out_path)
    LOG.info(f"saved {out_path}")
    return out_path


def main():
    p = base_parser("prepare NetFlow data into temporal graph snapshots")
    p.add_argument("--reference", help="prepared.pt whose preprocessing should be reused")
    p.add_argument("--out", help="output path (default: <output_dir>/prepared.pt)")
    p.add_argument("--force", action="store_true", help="rebuild even if prepared data already exists")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    setup_logging(cfg["output_dir"], "prepare")
    set_seed(cfg["seed"])
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out) if a.out else Path(cfg["output_dir"]) / "prepared.pt"
    csvs = cfg["data"]["csv"] if isinstance(cfg["data"]["csv"], list) else [cfg["data"]["csv"]]
    fp = "|".join([fingerprint(cfg, ("seed", "data", "graph", "features"))]
                  + [file_id(c) for c in csvs if c and Path(c).exists()]
                  + ([file_id(a.reference)] if a.reference else []))
    stamp = out_path.with_suffix(".pt.json")
    if not a.force and out_path.exists() and stamp.exists() and json.loads(stamp.read_text()).get("fingerprint") == fp:
        LOG.info(f"{out_path} already prepared with these settings - skipping (use --force to rebuild)")
        return
    if a.reference is None:
        save_config(cfg, Path(cfg["output_dir"]) / "config.yaml")
    prepare(cfg, a.reference, a.out)
    stamp.write_text(json.dumps({"fingerprint": fp}))


if __name__ == "__main__":
    main()
