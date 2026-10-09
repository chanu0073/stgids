"""Training / inference loops for the graph models (STGIDS and E-GraphSAGE)."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch

from .adversarial import ThreatModel, make_space, pgd
from .graphs import SnapshotStore
from .losses import build_loss
from .metrics import compute_metrics, short
from .model import DGIHead, STGIDS
from .utils import LOG, Timer, atomic_save, rng_state, set_rng_state


def graph_forward(model, window):
    def fwd(kind, t):
        return model(window, num_override=t) if kind == "num" else model(window, x_override=t)
    return fwd


@torch.no_grad()
def predict(model, store: SnapshotStore, split: str, device, max_windows: int | None = None) -> dict:
    model.eval()
    ys, ybs, preds, probs, times = [], [], [], [], []
    targets = store.targets(split)[:max_windows] if max_windows else store.targets(split)
    for t in targets:
        window = store.window(t, device)
        with Timer() as tm:
            logits = model(window)
        times.append(tm.elapsed)
        p = logits.softmax(1)
        ys.append(window[-1].y.cpu()); ybs.append(window[-1].y_bin.cpu())
        preds.append(p.argmax(1).cpu()); probs.append(p.cpu())
    return {"y": torch.cat(ys).numpy(), "y_bin": torch.cat(ybs).numpy(),
            "pred": torch.cat(preds).numpy(), "prob": torch.cat(probs).numpy(),
            "latency_ms_per_snapshot": 1000 * float(np.mean(times)),
            "flows_per_second": float(sum(len(y) for y in ys) / max(sum(times), 1e-9))}


def evaluate_split(model, store, split, device, max_windows=None) -> dict:
    out = predict(model, store, split, device, max_windows)
    m = compute_metrics(out["y"], out["y_bin"], out["pred"], out["prob"], store.meta["classes"])
    m["latency_ms_per_snapshot"] = out["latency_ms_per_snapshot"]
    m["flows_per_second"] = out["flows_per_second"]
    return m


def pretrain_spatial(model: STGIDS, store: SnapshotStore, cfg: dict, device) -> list[float]:
    """Self-supervised DGI pretraining of the spatial encoder on training snapshots (no labels)."""
    pcfg = cfg["pretrain"]
    head = DGIHead(cfg["model"]["hidden"]).to(device)
    params = list(model.spatial.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=pcfg["lr"])
    targets = store.targets("train")
    history = []
    for ep in range(pcfg["epochs"]):
        random.shuffle(targets)
        losses = []
        model.train()
        for t in targets:
            s = store.snapshot(t).to(device)
            loss = head.loss(model.spatial, model.inputs(s), s.src, s.dst, s.n_nodes)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
        LOG.info(f"pretrain epoch {ep + 1}/{pcfg['epochs']} | DGI loss {history[-1]:.4f}")
    return history


def train_model(model, store: SnapshotStore, cfg: dict, device, ckpt_path=None, fp: str | None = None,
                resume: dict | None = None) -> tuple[torch.nn.Module, list[dict]]:
    """Train with early stopping. After every epoch the full training state is written to
    ckpt_path, so an interrupted run continues from the last finished epoch when resumed."""
    tcfg, acfg = cfg["train"], cfg["train"]["adv"]
    counts = store.class_counts("train")
    LOG.info("training class counts: " + ", ".join(
        f"{c}={int(n)}" for c, n in zip(store.meta["classes"], counts.tolist())))
    criterion = build_loss(tcfg, counts).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])
    tm = ThreatModel(cfg["threat_model"], store.meta["num_names"]) if acfg["mode"] != "none" else None
    targets = store.targets("train")
    best_f1, best_state, bad, history, start, finished = -1.0, None, 0, [], 0, False

    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"])
        best_f1, best_state, bad = resume["best_f1"], resume["best_state"], resume["bad"]
        history, start, finished = resume["history"], resume["epoch"], resume["finished"]
        set_rng_state(resume["rng"])
        LOG.info(f"RESUMING from checkpoint: {start} epoch(s) already done, best val macro-F1 {best_f1:.4f}"
                 + (" (training had already finished)" if finished else ""))

    def save_checkpoint(epoch: int, done: bool) -> None:
        if ckpt_path is None:
            return
        atomic_save({"epoch": epoch, "finished": done, "model": model.state_dict(),
                     "opt": opt.state_dict(), "sched": sched.state_dict(), "best_f1": best_f1,
                     "best_state": best_state, "bad": bad, "history": history, "rng": rng_state(),
                     "fingerprint": fp}, ckpt_path)

    for ep in range(start, tcfg["epochs"]):
        if finished:
            break
        model.train()
        order = list(targets)  # shuffle a fresh copy so the order depends only on the saved RNG state
        random.shuffle(order)
        use_adv = tm is not None and ep >= acfg["start_epoch"]
        losses = []
        opt.zero_grad()
        for i, t in enumerate(order):
            window = store.window(t, device)
            target = window[-1]
            loss = criterion(model(window), target.y)
            mask = target.y_bin == 1
            if use_adv and mask.any():
                model.eval()  # craft the attack without dropout noise
                space = make_space(acfg["mode"], tm, model.tf, target.num_raw, target.cat, mask, acfg["eps"])
                adv = pgd(graph_forward(model, window), space, target.y, mask, acfg["steps"], tm.objective)
                model.train()
                adv_logits = graph_forward(model, window)(space.kind, adv)
                loss = (1 - acfg["weight"]) * loss + acfg["weight"] * criterion(adv_logits, target.y)
            (loss / tcfg["batch_windows"]).backward()
            losses.append(loss.item())
            if (i + 1) % tcfg["batch_windows"] == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                opt.step()
                opt.zero_grad()
        sched.step()
        val = evaluate_split(model, store, "val", device)
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), "adv": use_adv,
               **{f"val_{k}": v for k, v in val.items() if isinstance(v, (int, float)) or v is None}}
        history.append(rec)
        LOG.info(f"epoch {ep + 1:3d} | loss {rec['train_loss']:.4f} | val {short(val)}")
        score = val.get("macro_f1") or 0.0
        if score > best_f1:
            best_f1, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                LOG.info(f"early stopping (no val macro-F1 gain for {bad} epochs)")
                finished = True
        if ep + 1 == tcfg["epochs"]:
            finished = True
        save_checkpoint(ep + 1, finished)
        LOG.info(f"checkpoint saved (epoch {ep + 1})")
    if best_state is not None:
        model.load_state_dict(best_state)
    LOG.info(f"best val macro-F1 {best_f1:.4f}")
    return model, history
