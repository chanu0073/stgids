"""Train the injection-proof detector of a detect-attribute model (see detector.py).

The attribution model (a trained STGIDS or E-GraphSAGE model.pt) is loaded and frozen; only the
detector is trained, with a binary focal loss (benign vs attack) and optionally adversarial
training on the attack flows' own features. The epoch with the best validation macro-F1 of the
whole model is kept. Writes model.pt (usable by evaluate.py and evaluate_inject.py),
train_history.json and results_clean_test.json; a checkpoint after every epoch makes it resumable.

Usage:
    python -m stgids.train_detector --config configs/nf_unsw_nb15_v3.yaml --data runs/unsw/prepared.pt \
        --typer runs/unsw/stgids/model.pt --set output_dir=runs/unsw/stgids_mono
    # ablation: detector without host context (flow features only)
    ... --set output_dir=runs/unsw/stgids_flowdet detector.context=false
    # attack types from a neural flow baseline (types cannot be changed by injected flows either)
    ... --typer runs/unsw/baselines/model_mlp_adv.pt --set output_dir=runs/unsw/mlpadv_mono

Detector settings live in a separate `detector` config section (not in config.DEFAULTS, so the
fingerprints of existing runs do not change); unset keys take the values of DET_DEFAULTS. Optional:
detector.channels (evidence channels, default 1) and detector.fusion ("detector" | "or", default
"detector"); they are recorded only when set, so earlier models keep their fingerprints.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import torch

from .adversarial import ThreatModel, make_space, pgd
from .config import _deep_update, load_config, save_config
from .engine import evaluate_split
from .graphs import SnapshotStore
from .metrics import compute_metrics, short
from .model import build_model
from .utils import (LOG, atomic_save, atomic_save_json, base_parser, file_id, fingerprint, get_device,
                    rng_state, save_json, set_rng_state, set_seed, setup_logging)

DET_DEFAULTS = {
    "context": True,  # false = flow-only detector (ablation)
    "hidden": 64,
    "context_dim": 16,
    "epochs": 10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_windows": 4,
    "grad_clip": 1.0,
    "focal_gamma": 2.0,
    "focal_alpha": 0.8,  # weight of the attack class in the binary focal loss
    "adv": {"mode": "none", "eps": 0.1, "steps": 5, "weight": 0.5},  # mode: none | constrained | unconstrained
}


def focal_bce(logit: torch.Tensor, y: torch.Tensor, gamma: float, alpha: float) -> torch.Tensor:
    p = torch.sigmoid(logit)
    pt = torch.where(y > 0, p, 1 - p)
    a = torch.where(y > 0, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
    return (-a * (1 - pt) ** gamma * torch.log(pt.clamp(min=1e-8))).mean()


def inputs(model, window):
    return [model.tf(s.num_raw, s.cat) for s in window]


@torch.no_grad()
def typer_cache(model, store, split, device) -> list:
    """Most likely attack class of the frozen attribution model for every flow of the split, and its
    attack-vs-benign margin (> 0 iff it predicts an attack; used by detector.fusion = "or")."""
    out = []
    for t in store.targets(split):
        w = store.window(t, device)
        lg = model.typer.forward_inputs(w, inputs(model, w))
        margin = lg[:, 1:].max(1).values - lg[:, 0]
        out.append((t, (lg[:, 1:].argmax(1) + 1).cpu(), margin.cpu(), w[-1].y.cpu(), w[-1].y_bin.cpu()))
    return out


@torch.no_grad()
def system_metrics(model, store, cache, device) -> dict:
    preds, ys, ybs = [], [], []
    for t, typ, margin, y, yb in cache:
        w = store.window(t, device)
        d = model.detector.forward_inputs(w, inputs(model, w)).cpu()
        if model.fusion == "or":
            d = torch.maximum(d, margin)
        preds.append(torch.where(d > 0, typ, torch.zeros_like(typ)))
        ys.append(y); ybs.append(yb)
    return compute_metrics(torch.cat(ys).numpy(), torch.cat(ybs).numpy(), torch.cat(preds).numpy(), None,
                           store.meta["classes"])


def main():
    p = base_parser("train the injection-proof detector of a detect-attribute model")
    p.add_argument("--typer", required=True, help="model.pt of the trained attribution model")
    p.add_argument("--data", help="prepared.pt (default: <output_dir>/prepared.pt)")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    det = _deep_update(copy.deepcopy(DET_DEFAULTS), cfg.get("detector") or {})
    out = Path(cfg["output_dir"])
    setup_logging(out, "train_detector")
    device = get_device(cfg["device"])
    data_path = Path(a.data or out / "prepared.pt")
    typer_path = Path(a.typer)

    tck = torch.load(typer_path, weights_only=False, map_location="cpu")
    if "config" in tck:  # trained graph model (STGIDS / E-GraphSAGE)
        base, typer_model = tck["config"], tck["config"]["model"]
        if base["seed"] != cfg["seed"]:
            LOG.warning(f"seed {cfg['seed']} differs from the attribution model's seed {base['seed']}")
    else:  # neural flow baseline of baselines.py: <run>/baselines/model_<name>.pt
        name = typer_path.stem.removeprefix("model_")
        if "model" not in tck or name in ("rf", "xgb"):
            raise SystemExit(f"{typer_path} is neither a graph model nor a neural flow baseline")
        base, typer_model = cfg, {"arch": "flow", "net": name}
    det_model = {k: det[k] for k in ("context", "hidden", "context_dim")}
    det_model.update({k: det[k] for k in ("channels", "fusion") if k in det})
    sys_cfg = {**copy.deepcopy(base), "seed": cfg["seed"], "output_dir": str(out), "detector": det,
               "model": {"arch": "detect_attribute", "typer": typer_model, "detector": det_model}}
    fp = "-".join([fingerprint(sys_cfg, ("seed", "model", "graph", "detector")), file_id(data_path),
                   file_id(typer_path)])
    model_path, ckpt_path = out / "model.pt", out / "checkpoint.pt"
    if model_path.exists():
        done = torch.load(model_path, weights_only=False, map_location="cpu")
        if done.get("fingerprint") == fp:
            LOG.info(f"{model_path} already trained with these settings - skipping")
            return
        raise SystemExit(f"{model_path} exists but was trained with different settings or data.\n"
                         f"Use a new output_dir or delete it.")

    set_seed(cfg["seed"])
    out.mkdir(parents=True, exist_ok=True)
    save_config(sys_cfg, out / "config.yaml")
    store = SnapshotStore(str(data_path), history_k=sys_cfg["graph"]["history_k"])
    model = build_model(store.meta, sys_cfg).to(device)
    if "config" in tck:
        model.typer.load_state_dict(tck["state_dict"])
    else:
        model.typer.net.load_state_dict(tck["model"])
    model.typer.eval()
    for prm in model.typer.parameters():
        prm.requires_grad_(False)
    params = list(model.detector.parameters())
    LOG.info(f"detect-attribute model | detector context={det['context']} channels={det.get('channels', 1)} "
             f"fusion={det.get('fusion', 'detector')} | {sum(q.numel() for q in params):,} trainable parameters | "
             f"attribution model {typer_path}")
    opt = torch.optim.AdamW(params, lr=det["lr"], weight_decay=det["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=det["epochs"])
    adv = det["adv"]
    tm = ThreatModel(cfg["threat_model"], store.meta["num_names"]) if adv["mode"] != "none" else None
    best_f1, best_state, history, start = -1.0, None, [], 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        if ck.get("fingerprint") == fp:
            model.detector.load_state_dict(ck["detector"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            best_f1, best_state, history, start = ck["best_f1"], ck["best_state"], ck["history"], ck["epoch"]
            set_rng_state(ck["rng"])
            LOG.info(f"RESUMING from epoch {start}, best val macro-F1 {best_f1:.4f}")
        else:
            LOG.info("checkpoint belongs to other settings - starting over")

    LOG.info("caching the attribution model's predictions on the validation split")
    val_cache = typer_cache(model, store, "val", device)
    targets = store.targets("train")
    for ep in range(start, det["epochs"]):
        order = list(targets)
        random.shuffle(order)
        model.detector.train()
        losses = []
        opt.zero_grad()
        for i, t in enumerate(order):
            w = store.window(t, device)
            tgt = w[-1]
            xs = inputs(model, w)
            y = tgt.y_bin.float()
            loss = focal_bce(model.detector.forward_inputs(w, xs), y, det["focal_gamma"], det["focal_alpha"])
            mask = tgt.y_bin == 1
            if tm is not None and mask.any():
                space = make_space(adv["mode"], tm, model.tf, tgt.num_raw, tgt.cat, mask, adv["eps"])

                def fwd(kind, v, w=w, xs=xs, tgt=tgt):  # two-class logits [benign, attack] for pgd
                    x = model.tf(v, tgt.cat) if kind == "num" else v
                    d = model.detector.forward_inputs(w, xs[:-1] + [x])
                    return torch.stack([torch.zeros_like(d), d], 1)
                x_adv = pgd(fwd, space, tgt.y_bin, mask, adv["steps"], "evasion")
                adv_loss = focal_bce(fwd(space.kind, x_adv)[:, 1], y, det["focal_gamma"], det["focal_alpha"])
                loss = (1 - adv["weight"]) * loss + adv["weight"] * adv_loss
            (loss / det["batch_windows"]).backward()
            losses.append(loss.item())
            if (i + 1) % det["batch_windows"] == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(params, det["grad_clip"])
                opt.step()
                opt.zero_grad()
        sched.step()
        model.detector.eval()
        val = system_metrics(model, store, val_cache, device)
        rec = {"epoch": ep + 1, "train_loss": float(sum(losses) / max(len(losses), 1)),
               **{f"val_{k}": v for k, v in val.items() if isinstance(v, (int, float)) or v is None}}
        history.append(rec)
        LOG.info(f"epoch {ep + 1:3d} | loss {rec['train_loss']:.4f} | val {short(val)}")
        if val["macro_f1"] > best_f1:
            best_f1, best_state = val["macro_f1"], copy.deepcopy(model.detector.state_dict())
        atomic_save({"epoch": ep + 1, "detector": model.detector.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "best_f1": best_f1, "best_state": best_state,
                     "history": history, "rng": rng_state(), "fingerprint": fp}, ckpt_path)

    LOG.info(f"best val macro-F1 {best_f1:.4f}")
    model.detector.load_state_dict(best_state)
    model.eval()
    test = evaluate_split(model, store, "test", device)
    LOG.info(f"TEST (clean, future traffic): {short(test)}")
    save_json({"history": history, "pretrain_history": None}, out / "train_history.json")
    atomic_save_json(test, out / "results_clean_test.json")
    atomic_save({"state_dict": model.state_dict(), "config": sys_cfg, "meta": store.meta, "fingerprint": fp},
                model_path)
    LOG.info(f"saved {model_path}")


if __name__ == "__main__":
    main()
