"""Step 8: evaluation on future traffic, under attack, and across datasets.

Usage:
    # clean + robustness on the chronological test split
    python -m stgids.evaluate --config runs/unsw/config.yaml
    # cross-dataset (data prepared with --reference, see prepare.py)
    python -m stgids.evaluate --config runs/unsw/config.yaml \
        --data runs/unsw/cross_cic2018.pt --tag cross_cic2018 --no-attacks
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .adversarial import ThreatModel, make_space, run_attack
from .config import load_config
from .engine import evaluate_split, graph_forward
from .graphs import SnapshotStore
from .metrics import compute_metrics, short
from .model import build_model
from .utils import (LOG, atomic_save_json, base_parser, file_id, fingerprint, get_device, set_seed,
                    setup_logging)


def attack_split(model, store, device, threat, attack, eps, tm, ecfg, seed) -> dict:
    model.eval()
    ys, ybs, preds, probs = [], [], [], []
    targets = store.targets("test")
    if ecfg.get("max_windows"):
        targets = targets[: ecfg["max_windows"]]
    for j, t in enumerate(targets):
        window = store.window(t, device)
        tgt = window[-1]
        mask = tgt.y_bin == 1
        fwd = graph_forward(model, window)
        if mask.any():
            space = make_space(threat, tm, model.tf, tgt.num_raw, tgt.cat, mask, eps)
            with torch.enable_grad():
                adv = run_attack(attack, fwd, space, tgt.y, mask, ecfg, tm.objective, seed + j)
            with torch.no_grad():
                logits = fwd(space.kind, adv)
        else:
            with torch.no_grad():
                logits = model(window)
        p = logits.softmax(1)
        ys.append(tgt.y.cpu()); ybs.append(tgt.y_bin.cpu())
        preds.append(p.argmax(1).cpu()); probs.append(p.cpu())
    cat = lambda xs: torch.cat(xs).numpy()  # noqa: E731
    return compute_metrics(cat(ys), cat(ybs), cat(preds), cat(probs), store.meta["classes"])


def write_summary(rows: list[dict], path: Path) -> None:
    keys = ["setting", "threat", "attack", "eps", "macro_f1", "weighted_f1", "macro_pr_auc",
            "fpr", "detection_rate", "binary_f1"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    p = base_parser("evaluate a trained model")
    p.add_argument("--checkpoint", help="model.pt (default: <output_dir>/model.pt)")
    p.add_argument("--data", help="prepared data to test on (default: <output_dir>/prepared.pt)")
    p.add_argument("--tag", default="test", help="name used for the result files")
    p.add_argument("--no-attacks", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    out = Path(cfg["output_dir"])
    setup_logging(out, f"evaluate_{a.tag}")
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    ckpt_path = Path(a.checkpoint or out / "model.pt")
    data_path = Path(a.data or out / "prepared.pt")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    mcfg = ckpt["config"]
    k = mcfg["graph"]["history_k"] if mcfg["model"]["arch"] == "stgids" else 1
    store = SnapshotStore(str(data_path), history_k=k)
    model = build_model(ckpt["meta"], mcfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    res_path = out / f"results_{a.tag}.json"
    ecfg = cfg["eval"]
    source = "|".join([file_id(ckpt_path), file_id(data_path), fingerprint(cfg, ("threat_model",)),
                       f"max_windows={ecfg.get('max_windows')}"])
    wanted = ["clean"] + ([] if a.no_attacks else [
        f"{th}/{at}/eps={e}" for th in ecfg["threats"] for at in ecfg["attacks"] for e in ecfg["eps_list"]])
    results = {}
    if res_path.exists():
        prev = json.loads(res_path.read_text())
        if prev.get("_source") == source:
            results = prev
            done = [k for k in wanted if k in prev]
            if len(done) == len(wanted):
                LOG.info(f"{res_path}: all {len(wanted)} requested results already saved - nothing to do "
                         f"(delete the file to re-evaluate)")
                write_rows(results, out / f"summary_{a.tag}.csv")
                return
            LOG.info(f"RESUMING evaluation: {len(done)}/{len(wanted)} results already saved")
        else:
            LOG.info("saved results were made with a different model, data or threat model - starting over")
    results["_source"] = source

    if "clean" not in results:
        results["clean"] = evaluate_split(model, store, "test", device)
        atomic_save_json(results, res_path)
    LOG.info(f"clean: {short(results['clean'])}")
    LOG.info(f"latency {results['clean']['latency_ms_per_snapshot']:.1f} ms/snapshot, "
             f"{results['clean']['flows_per_second']:.0f} flows/s")

    if not a.no_attacks:
        tm = ThreatModel(cfg["threat_model"], store.meta["num_names"])
        LOG.info(f"controllable features: {tm.names}")
        sel_names = [store.meta["input_names"][i] for i in np.asarray(store.meta["transform"]["selected"])]
        kept = [n for n in tm.names if n in sel_names]
        LOG.info(f"controllable features kept by feature selection: {kept}")
        for threat in ecfg["threats"]:
            for attack in ecfg["attacks"]:
                for eps in ecfg["eps_list"]:
                    key = f"{threat}/{attack}/eps={eps}"
                    if key in results:
                        LOG.info(f"{key}: already done, skipped")
                        continue
                    results[key] = attack_split(model, store, device, threat, attack, eps, tm, ecfg, cfg["seed"])
                    atomic_save_json(results, res_path)  # progress survives a crash
                    LOG.info(f"{key}: {short(results[key])}")

    results["_complete"] = all(k in results for k in wanted)
    atomic_save_json(results, res_path)
    write_rows(results, out / f"summary_{a.tag}.csv")
    LOG.info(f"saved {res_path} and summary_{a.tag}.csv")


def write_rows(results: dict, path: Path) -> None:
    rows = []
    for key, m in results.items():
        if key.startswith("_"):
            continue
        if key == "clean":
            rows.append({"setting": "clean", "threat": "-", "attack": "-", "eps": 0, **m})
        else:
            threat, attack, e = key.split("/")
            rows.append({"setting": "attack", "threat": threat, "attack": attack,
                         "eps": float(e.split("=")[1]), **m})
    write_summary(rows, path)


if __name__ == "__main__":
    main()
