"""Configuration loading.

A run is described by one YAML file. Anything not set there falls back to
DEFAULTS below. Single values can be overridden from the command line with
``--set section.key=value`` (the value is parsed as YAML, so numbers, lists
and booleans work: ``--set train.loss=ce model.temporal=gru``).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "run",
    "output_dir": "runs/default",
    "seed": 42,
    "device": "auto",  # auto | cpu | cuda
    "data": {
        "csv": None,  # path or list of paths
        "max_rows": None,  # read only the first N rows (debugging / memory)
        "time_col": "FLOW_START_MILLISECONDS",
        "time_unit": "ms",  # ms | s | datetime
        "src_ip_col": "IPV4_SRC_ADDR",
        "dst_ip_col": "IPV4_DST_ADDR",
        "src_port_col": "L4_SRC_PORT",
        "dst_port_col": "L4_DST_PORT",
        "label_col": "Attack",
        "benign_label": "Benign",
        # dropped before modelling (identifiers, label leakage, random IDs)
        "drop_cols": ["FLOW_END_MILLISECONDS", "Label", "DNS_QUERY_ID"],
        # expanded to 8 bit-indicator columns each
        "tcp_flag_cols": ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"],
        # one-hot of the top-k most frequent training values (+ "other")
        "categorical_cols": ["PROTOCOL", "L7_PROTO", "L4_DST_PORT", "ICMP_TYPE",
                             "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE"],
        "cat_top_k": 16,
        "split": [0.70, 0.15, 0.15],  # chronological train / val / test by flow count
        "dedup": True,
    },
    "features": {
        "log_transform": True,
        "selection": {
            "enabled": True,
            "top_k": 30,  # size of the top set in each bootstrap run
            "n_bootstrap": 5,
            "sample_size": 30000,
            "min_freq": 0.6,  # keep features in the top set in >= this share of runs
            "min_features": 15,
            "use_shap": True,  # falls back to tree importance if shap is missing
        },
    },
    "graph": {
        "window_seconds": 60,
        "node_key": "ip",  # ip | ip_port
        "history_k": 6,  # number of snapshots the temporal encoder sees (incl. current)
    },
    "model": {
        "arch": "stgids",  # stgids | egraphsage
        "hidden": 64,
        "heads": 4,
        "gnn_layers": 2,
        "dropout": 0.1,
        "temporal": "transformer",  # transformer | gru | none
        "temporal_layers": 2,
        "detach_history": True,  # no gradient through past snapshots (faster, less memory)
    },
    "pretrain": {
        "enabled": False,
        "epochs": 5,
        "lr": 1e-3,
    },
    "train": {
        "epochs": 30,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_windows": 4,  # windows accumulated per optimiser step
        "patience": 6,
        "loss": "focal",  # focal | ce | weighted_ce
        "focal_gamma": 2.0,
        "focal_alpha": "inv_sqrt",  # inv_sqrt | none
        "grad_clip": 1.0,
        "adv": {
            "mode": "none",  # none | constrained | unconstrained
            "eps": 0.1,
            "steps": 5,
            "weight": 0.5,  # loss = (1-w) * clean + w * adversarial
            "start_epoch": 2,
        },
    },
    "threat_model": {
        # Attacker-controllable features (src->dst direction only), changed additively.
        "controllable": ["IN_BYTES", "IN_PKTS", "FLOW_DURATION_MILLISECONDS", "DURATION_IN",
                         "LONGEST_FLOW_PKT", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_MAX"],
        "integer": ["IN_BYTES", "IN_PKTS", "FLOW_DURATION_MILLISECONDS", "DURATION_IN",
                    "LONGEST_FLOW_PKT", "SRC_TO_DST_IAT_MAX"],
        # minimum absolute budget so zero-valued features can still move
        "abs_floor": {"IN_BYTES": 64, "IN_PKTS": 1, "FLOW_DURATION_MILLISECONDS": 10,
                      "DURATION_IN": 10, "LONGEST_FLOW_PKT": 16,
                      "SRC_TO_DST_IAT_AVG": 5, "SRC_TO_DST_IAT_MAX": 5},
        "upper_bound": {"LONGEST_FLOW_PKT": 1500},
        # dependent = old * prod((new_c + 1) / (old_c + 1)) ** power  (ratio rule, unit-free)
        "dependent": {
            "SRC_TO_DST_SECOND_BYTES": {"IN_BYTES": 1, "FLOW_DURATION_MILLISECONDS": -1},
            "SRC_TO_DST_AVG_THROUGHPUT": {"IN_BYTES": 1, "FLOW_DURATION_MILLISECONDS": -1},
        },
        # [a, b, c] means a >= c * b must hold after perturbation
        "ge_constraints": [["IN_BYTES", "IN_PKTS", 40],
                           ["IN_BYTES", "LONGEST_FLOW_PKT", 1],
                           ["FLOW_DURATION_MILLISECONDS", "DURATION_IN", 1],
                           ["SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", 1]],
        "objective": "evasion",  # evasion (push attack flows to benign) | untargeted
    },
    "eval": {
        "attacks": ["fgsm", "pgd", "deepfool", "square"],
        "threats": ["constrained", "unconstrained"],
        "eps_list": [0.05, 0.1, 0.2],
        "pgd_steps": 20,
        "deepfool_iters": 20,
        "square_iters": 100,
        "max_windows": None,  # limit attacked test windows (speed); None = all
    },
    "baselines": {
        "models": ["rf", "xgb", "mlp", "mlp_adv", "bigru", "aegru"],
        "epochs": 20,
        "batch_size": 1024,
        "lr": 1e-3,
    },
}


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path | None, overrides: Iterable[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        _deep_update(cfg, user)
    apply_overrides(cfg, overrides)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
