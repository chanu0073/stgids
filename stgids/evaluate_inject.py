"""Robustness to flow injection (see inject.py): random benign-template injection and adaptive
(PGD-optimised) injection, into the target window only ("now") or into every window of the
temporal context ("hist", history shaping), optionally combined with a feature attack on the
attack flows themselves.

Usage:
    python -m stgids.evaluate_inject --config runs/unsw/stgids/config.yaml
    python -m stgids.evaluate_inject --config runs/cic2018_full/stgids/config.yaml --max-windows 220 \
        --betas 5 20 100 --feature unconstrained:0.1

Results go to <output_dir>/results_inject.json (one entry per setting, resumable like evaluate.py).
Models without temporal context (E-GraphSAGE, temporal=none) are only attacked with mode "now",
because injections into earlier windows cannot change their predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .adversarial import ThreatModel
from .config import load_config
from .graphs import SnapshotStore
from .inject import attack_window, benign_input_range, inject_window
from .metrics import compute_metrics, short
from .model import build_model
from .utils import LOG, atomic_save_json, base_parser, file_id, get_device, set_seed, setup_logging


def uses_history(mcfg: dict) -> bool:
    return mcfg["model"]["arch"] != "egraphsage" and mcfg["model"].get("temporal", "transformer") != "none"


def run_setting(model, store, targets, device, beta, mode, adaptive, lo, hi, seed, steps, feature=None,
                objective="evasion") -> dict:
    model.eval()
    ys, ybs, preds = [], [], []
    n_inj = n_inj_flagged = 0
    for t in targets:
        gen = torch.Generator(device=device).manual_seed(seed * 100_003 + t)  # same injection on resume
        window, counts = inject_window(store.window(t, device), beta, mode, gen)
        tgt = window[-1]
        e = len(tgt.y) - counts[-1]
        if adaptive or feature is not None:
            xs = attack_window(model, window, counts, lo, hi, steps=steps, feature=feature, objective=objective)
            with torch.no_grad():
                pred = model.forward_inputs(window, xs).argmax(1)
        else:
            with torch.no_grad():
                pred = model(window).argmax(1)
        ys.append(tgt.y[:e].cpu()); ybs.append(tgt.y_bin[:e].cpu()); preds.append(pred[:e].cpu())
        n_inj += counts[-1]
        n_inj_flagged += int((pred[e:] != 0).sum())
    m = compute_metrics(torch.cat(ys).numpy(), torch.cat(ybs).numpy(), torch.cat(preds).numpy(), None,
                        store.meta["classes"])
    m["injected_flows_target"] = n_inj
    m["injected_flagged_rate"] = n_inj_flagged / n_inj if n_inj else None
    return m


def main():
    p = base_parser("evaluate a trained graph model under flow injection")
    p.add_argument("--checkpoint", help="model.pt (default: <output_dir>/model.pt)")
    p.add_argument("--data", help="prepared data (default: <output_dir>/prepared.pt)")
    p.add_argument("--betas", type=int, nargs="+", default=[5, 20, 100],
                   help="injected flows per attacker host and window")
    p.add_argument("--modes", nargs="+", default=["now", "hist"])
    p.add_argument("--steps", type=int, default=20, help="PGD steps of the adaptive attack")
    p.add_argument("--max-windows", type=int, help="attack an evenly spaced subset of the test windows")
    p.add_argument("--feature", action="append", default=[],
                   help="also run adaptive injection jointly with a feature attack, e.g. constrained:0.1")
    p.add_argument("--no-random", action="store_true", help="skip the random (non-adaptive) injection")
    p.add_argument("--objectives", nargs="+", default=["evasion"], choices=["evasion", "untargeted"],
                   help="attacker goals of the adaptive attacks (untargeted = any wrong class)")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    out = Path(cfg["output_dir"])
    setup_logging(out, "evaluate_inject")
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    ckpt_path = Path(a.checkpoint or out / "model.pt")
    data_path = Path(a.data or out / "prepared.pt")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    mcfg = ckpt["config"]
    k = mcfg["graph"]["history_k"] if mcfg["model"]["arch"] != "egraphsage" else 1
    store = SnapshotStore(str(data_path), history_k=k)
    model = build_model(ckpt["meta"], mcfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    targets = store.targets("test")
    if a.max_windows and a.max_windows < len(targets):
        stride = len(targets) / a.max_windows
        targets = [targets[int(i * stride)] for i in range(a.max_windows)]
    modes = a.modes if uses_history(mcfg) else ["now"]
    wanted = []  # (kind, mode, beta, feature attack, objective)
    for mode in modes:
        for beta in a.betas:
            if not a.no_random:
                wanted.append(("random", mode, beta, None, "evasion"))
            for obj in a.objectives:
                wanted.append(("adaptive", mode, beta, None, obj))
                for f in a.feature:
                    wanted.append(("adaptive", mode, beta, f, obj))
    for obj in a.objectives:
        for f in a.feature:
            wanted.append(("feature", "now", 0, f, obj))  # the feature attack alone, same subset

    def key_of(kind, mode, beta, feat, obj):
        return f"{kind}/{mode}/beta={beta}" + (f"/{feat}" if feat else "") + ("" if obj == "evasion" else f"/{obj}")

    res_path = out / "results_inject.json"
    source = "|".join([file_id(ckpt_path), file_id(data_path), f"steps={a.steps}",
                       f"windows={len(targets)}", f"seed={cfg['seed']}"])
    results = {}
    if res_path.exists():
        prev = json.loads(res_path.read_text())
        if prev.get("_source") == source:
            results = prev
            LOG.info(f"RESUMING: {sum(1 for k in prev if not k.startswith('_'))} settings already saved")
        else:
            LOG.info("saved injection results belong to another model, data or setting - starting over")
    results["_source"] = source
    if "clean" not in results:
        results["clean"] = run_setting(model, store, targets, device, 0, "now", False, None, None, cfg["seed"], a.steps)
        atomic_save_json(results, res_path)
    LOG.info(f"clean ({len(targets)} test windows): {short(results['clean'])}")

    lo, hi = benign_input_range(model, store)
    tm = ThreatModel(cfg["threat_model"], store.meta["num_names"])
    for kind, mode, beta, feat, obj in wanted:
        key = key_of(kind, mode, beta, feat, obj)
        if key in results:
            LOG.info(f"{key}: already done, skipped")
            continue
        feature = None
        if feat:
            threat, eps = feat.split(":")
            feature = (threat, tm, float(eps))
        results[key] = run_setting(model, store, targets, device, beta, mode, kind != "random", lo, hi,
                                   cfg["seed"], a.steps, feature, obj)
        atomic_save_json(results, res_path)
        m = results[key]
        flagged = m["injected_flagged_rate"]
        LOG.info(f"{key}: {short(m)} | injected flagged "
                 f"{'n/a' if flagged is None else f'{flagged:.3f}'}")
    results["_complete"] = all(key_of(*w) in results for w in wanted)
    atomic_save_json(results, res_path)
    LOG.info(f"saved {res_path}")


if __name__ == "__main__":
    main()
