"""Steps 4-7: build the model, optional self-supervised pretraining, train, test on future traffic.

Interrupted runs resume automatically: just run the same command again.
  * a checkpoint is saved after every epoch (<output_dir>/checkpoint.pt);
  * self-supervised pretraining is saved once it finishes (<output_dir>/pretrain.pt);
  * if <output_dir>/model.pt already exists for the same settings, training is skipped.
Use --fresh to throw the saved state away and start again.

Usage:
    python -m stgids.train --config configs/nf_unsw_nb15_v3.yaml
    python -m stgids.train --config configs/nf_unsw_nb15_v3.yaml \
        --set train.adv.mode=constrained output_dir=runs/unsw_adv
"""
from __future__ import annotations

from pathlib import Path

import torch

from .config import load_config, save_config
from .engine import evaluate_split, pretrain_spatial, train_model
from .graphs import SnapshotStore
from .metrics import short
from .model import build_model
from .utils import (LOG, atomic_save, atomic_save_json, base_parser, file_id, fingerprint,
                    get_device, save_json, set_seed, setup_logging)


def run_fingerprint(cfg: dict, data_path: Path) -> str:
    keys = ["seed", "model", "graph", "train", "pretrain"]
    if cfg["train"]["adv"]["mode"] != "none":
        keys.append("threat_model")  # only matters when it shapes training
    return fingerprint(cfg, tuple(keys)) + "-" + file_id(data_path)


def main():
    p = base_parser("train the spatio-temporal graph IDS")
    p.add_argument("--data", help="prepared.pt (default: <output_dir>/prepared.pt)")
    p.add_argument("--fresh", action="store_true", help="ignore saved progress and train from scratch")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    out = Path(cfg["output_dir"])
    setup_logging(out, "train")
    device = get_device(cfg["device"])
    data_path = Path(a.data or out / "prepared.pt")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found - run `python -m stgids.prepare` first "
                                f"or pass --data")
    fp = run_fingerprint(cfg, data_path)
    ckpt_path, pre_path, model_path = out / "checkpoint.pt", out / "pretrain.pt", out / "model.pt"

    if a.fresh:
        for f in (ckpt_path, pre_path, model_path):
            f.unlink(missing_ok=True)
        LOG.info("--fresh: removed saved progress")

    if model_path.exists():
        done = torch.load(model_path, weights_only=False, map_location="cpu")
        if done.get("fingerprint") == fp:
            LOG.info(f"{model_path} already trained with these settings - skipping "
                     f"(use --fresh to retrain)")
            return
        raise SystemExit(f"{model_path} exists but was trained with different settings or data.\n"
                         f"Use a new output_dir, or add --fresh to overwrite it.")

    resume = None
    if ckpt_path.exists():
        resume = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        if resume.get("fingerprint") != fp:
            raise SystemExit(f"{ckpt_path} belongs to a run with different settings or data.\n"
                             f"Use a new output_dir, or add --fresh to start over.")

    set_seed(cfg["seed"])
    save_config(cfg, out / "config.yaml")
    k = cfg["graph"]["history_k"] if cfg["model"]["arch"] == "stgids" else 1
    store = SnapshotStore(str(data_path), history_k=k)
    model = build_model(store.meta, cfg).to(device)
    LOG.info(f"model {cfg['model']['arch']} | "
             f"temporal={cfg['model']['temporal'] if cfg['model']['arch'] == 'stgids' else '-'} | "
             f"{sum(p.numel() for p in model.parameters()):,} parameters | device {device}")

    pre_hist = None
    if cfg["pretrain"]["enabled"] and cfg["model"]["arch"] == "stgids" and resume is None:
        saved = torch.load(pre_path, weights_only=False, map_location=device) if pre_path.exists() else None
        if saved is not None and saved.get("fingerprint") == fp:
            model.load_state_dict(saved["model"])
            pre_hist = saved["history"]
            LOG.info("pretraining already done - loaded saved pretrained weights")
        else:
            pre_hist = pretrain_spatial(model, store, cfg, device)
            atomic_save({"model": model.state_dict(), "history": pre_hist, "fingerprint": fp}, pre_path)
    elif resume is not None and pre_path.exists():
        pre_hist = torch.load(pre_path, weights_only=False, map_location="cpu").get("history")

    model, history = train_model(model, store, cfg, device, ckpt_path=ckpt_path, fp=fp, resume=resume)

    test = evaluate_split(model, store, "test", device)
    LOG.info(f"TEST (clean, future traffic): {short(test)}")
    save_json({"history": history, "pretrain_history": pre_hist}, out / "train_history.json")
    atomic_save_json(test, out / "results_clean_test.json")
    atomic_save({"state_dict": model.state_dict(), "config": cfg, "meta": store.meta,
                 "fingerprint": fp}, model_path)
    LOG.info(f"saved {model_path}")


if __name__ == "__main__":
    main()
