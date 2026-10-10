"""Flow-vector baselines, retrained on exactly the same chronological splits and features.

    rf       Random Forest
    xgb      XGBoost
    mlp      feed-forward DNN (the model family attacked in Holla et al. 2025)
    mlp_adv  the same DNN with PGD adversarial training in feature space (Holla-style defence)
    bigru    stacked bidirectional GRU over the feature vector (MSBi-GRU-style)
    aegru    feed-forward autoencoder + GRU classifier (FFAE-GRU-style)

These are re-implementations from the papers' descriptions, not the authors'
code; say so in the paper. The graph baseline E-GraphSAGE is trained with
train.py (--set model.arch=egraphsage). ICloud has public code: run it on the
same split files instead of re-implementing it.

Usage:
    python -m stgids.baselines --config runs/unsw/config.yaml
    python -m stgids.baselines --config runs/unsw/config.yaml --set "baselines.models=[rf,mlp]"
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adversarial import GRADIENT_FREE, ThreatModel, UnconstrainedSpace, make_space, pgd, run_attack
from .config import load_config
from .evaluate import write_summary
from .graphs import SnapshotStore
from .losses import build_loss
from .metrics import compute_metrics, short
from .model import FeatureTransform
from .utils import (LOG, Timer, atomic_save, atomic_save_json, base_parser, file_id, fingerprint,
                    get_device, set_seed, setup_logging)


# --------------------------------------------------------------------------- networks
class MLPNet(nn.Module):
    def __init__(self, d, c, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(p), nn.Linear(256, 128),
                                 nn.ReLU(), nn.Dropout(p), nn.Linear(128, c))

    def forward(self, x):
        return self.net(x)


class BiGRUNet(nn.Module):
    def __init__(self, d, c, hidden=64, layers=2, p=0.2):
        super().__init__()
        self.gru = nn.GRU(1, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=p)
        self.fc = nn.Sequential(nn.Dropout(p), nn.Linear(2 * hidden, c))

    def forward(self, x):
        out, _ = self.gru(x[..., None])
        return self.fc(out[:, -1])


class AEGRUNet(nn.Module):
    def __init__(self, d, c, latent=32, hidden=64, p=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, d))
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(p), nn.Linear(hidden, c))

    def forward(self, x):
        out, _ = self.gru(self.enc(x)[..., None])
        return self.fc(out[:, -1])


NETS = {"mlp": MLPNet, "mlp_adv": MLPNet, "bigru": BiGRUNet, "aegru": AEGRUNet}


# --------------------------------------------------------------------------- helpers
def test_flows(store: SnapshotStore, max_windows: int | None) -> dict:
    fl = store.flows("test")
    if max_windows:
        t = store.targets("test")[:max_windows]  # flows(test) lists the test snapshots in this order
        n = int(sum(int(store.ptr[i + 1] - store.ptr[i]) for i in t))
        fl = {k: v[:n] for k, v in fl.items()}
    return fl


@torch.no_grad()
def transform_chunked(tf, num, cat, device, chunk=100_000) -> torch.Tensor:
    """Apply the feature transform in chunks so a whole split never sits on a small GPU at once."""
    out = [tf(num[i:i + chunk].to(device), cat[i:i + chunk].to(device)) for i in range(0, len(num), chunk)]
    return torch.cat(out)


def batches(n, bs, shuffle):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


class TreeWrapper:
    def __init__(self, model, n_classes, classes_seen):
        self.model, self.c, self.seen = model, n_classes, classes_seen

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        p = np.full((len(x), self.c), 1e-9)
        p[:, self.seen] = self.model.predict_proba(x.detach().cpu().numpy())
        return torch.log(torch.as_tensor(p, dtype=torch.float32, device=x.device) + 1e-9)


def train_tree(name, x, y, seed, n_classes):
    keep = y >= 0
    x, y = x[keep], y[keep]
    seen = np.unique(y)
    y_enc = np.searchsorted(seen, y)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed, min_samples_leaf=2)
    else:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
                          tree_method="hist", random_state=seed, n_jobs=-1, verbosity=0)
    m.fit(x, y_enc)
    return TreeWrapper(m, n_classes, seen)


def train_net(name, tf, tr, va, cfg, device, counts, tm):
    bcfg, tcfg = cfg["baselines"], cfg["train"]
    d, c = tf.out_dim, len(counts)
    net = NETS[name](d, c).to(device)
    crit = build_loss(tcfg, counts).to(device)
    xtr = transform_chunked(tf, tr["num_raw"], tr["cat"], device)
    ytr = tr["y"].to(device)
    ybin_tr = tr["y_bin"].to(device)
    if name == "aegru":  # unsupervised autoencoder warm-up, as in FFAE-GRU
        opt = torch.optim.Adam(list(net.enc.parameters()) + list(net.dec.parameters()), lr=bcfg["lr"])
        for _ in range(5):
            for b in batches(len(xtr), bcfg["batch_size"], True):
                loss = F.mse_loss(net.dec(net.enc(xtr[b])), xtr[b])
                opt.zero_grad(); loss.backward(); opt.step()
    opt = torch.optim.AdamW(net.parameters(), lr=bcfg["lr"], weight_decay=tcfg["weight_decay"])
    best, best_state, bad = -1, None, 0
    xva = transform_chunked(tf, va["num_raw"], va["cat"], device)
    for ep in range(bcfg["epochs"]):
        net.train()
        for b in batches(len(xtr), bcfg["batch_size"], True):
            xb, yb = xtr[b], ytr[b]
            loss = crit(net(xb), yb)
            if name == "mlp_adv":
                mask = ybin_tr[b] == 1
                if mask.any():
                    net.eval()
                    space = UnconstrainedSpace(xb, mask, tcfg["adv"]["eps"])
                    adv = pgd(lambda k, t: net(t), space, yb, mask, tcfg["adv"]["steps"], "untargeted")
                    net.train()
                    loss = 0.5 * loss + 0.5 * crit(net(adv), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(xva[b]).argmax(1) for b in batches(len(xva), 8192, False)]).cpu().numpy()
        f1 = compute_metrics(va["y"].numpy(), va["y_bin"].numpy(), pred, None, cfg["_classes"])["macro_f1"]
        LOG.info(f"  {name} epoch {ep + 1} | val macro-F1 {f1:.4f}")
        if f1 > best:
            best, best_state, bad = f1, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def flow_forward(name, model, tf, cat):
    if name in ("rf", "xgb"):
        return lambda kind, t: model.logits(tf(t, cat) if kind == "num" else t)
    return lambda kind, t: model(tf(t, cat) if kind == "num" else t)


def evaluate_flow_model(name, model, tf, te, cfg, device, tm, attacks: bool, res: dict, save) -> dict:
    """Clean + attack evaluation. Results already in `res` are skipped; `save(res)` runs after each."""
    classes = cfg["_classes"]
    num, cat = te["num_raw"].to(device), te["cat"].to(device)
    y, yb = te["y"].to(device), te["y_bin"].to(device)
    bs = 4096

    def predict(adv_fn=None):
        preds, probs = [], []
        for b in batches(len(num), bs, False):
            fwd = flow_forward(name, model, tf, cat[b])
            if adv_fn is None:
                with torch.no_grad():
                    logits = fwd("num", num[b])
            else:
                logits = adv_fn(fwd, b)
            p = logits.softmax(1)
            preds.append(p.argmax(1).cpu()); probs.append(p.cpu())
        pr = torch.cat(probs).numpy()
        return compute_metrics(te["y"].numpy(), te["y_bin"].numpy(), torch.cat(preds).numpy(), pr, classes)

    if "clean" not in res:
        with Timer() as t:
            res["clean"] = predict()
        res["clean"]["flows_per_second"] = len(num) / max(t.elapsed, 1e-9)
        save(res)
    LOG.info(f"  {name} clean: {short(res['clean'])}")
    if not attacks:
        return res
    ecfg = cfg["eval"]
    for threat in ecfg["threats"]:
        for attack in ecfg["attacks"]:
            if name in ("rf", "xgb") and attack not in GRADIENT_FREE:
                continue  # trees are not differentiable: black-box attack only
            for eps in ecfg["eps_list"]:
                key = f"{threat}/{attack}/eps={eps}"
                if key in res:
                    LOG.info(f"  {name} {key}: already done, skipped")
                    continue

                def adv_fn(fwd, b, threat=threat, attack=attack, eps=eps):
                    mask = yb[b] == 1
                    if not mask.any():
                        with torch.no_grad():
                            return fwd("num", num[b])
                    space = make_space(threat, tm, tf, num[b], cat[b], mask, eps)
                    with torch.enable_grad():
                        adv = run_attack(attack, fwd, space, y[b], mask, ecfg, tm.objective, cfg["seed"])
                    with torch.no_grad():
                        return fwd(space.kind, adv)
                res[key] = predict(adv_fn)
                save(res)
                LOG.info(f"  {name} {key}: {short(res[key])}")
    return res


def main():
    p = base_parser("train and evaluate flow-vector baselines")
    p.add_argument("--data", help="prepared.pt (default: <output_dir>/prepared.pt)")
    p.add_argument("--no-attacks", action="store_true")
    p.add_argument("--fresh", action="store_true", help="retrain and re-evaluate every baseline")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    out = Path(cfg["output_dir"]) / "baselines"
    setup_logging(out, "baselines")
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    data_path = Path(a.data or Path(cfg["output_dir"]) / "prepared.pt")
    store = SnapshotStore(str(data_path))
    cfg["_classes"] = store.meta["classes"]
    tf = FeatureTransform(store.meta["transform"]).to(device)
    tm = ThreatModel(cfg["threat_model"], store.meta["num_names"])
    tr, va = store.flows("train"), store.flows("val")
    te = test_flows(store, cfg["eval"].get("max_windows"))
    counts = store.class_counts("train")
    model_src = file_id(data_path) + "|" + fingerprint(cfg, ("seed", "baselines", "train"))
    res_src = model_src + "|" + fingerprint(cfg, ("threat_model",)) + f"|mw={cfg['eval'].get('max_windows')}"
    rows = []
    for name in cfg["baselines"]["models"]:
        LOG.info(f"=== baseline {name} ===")
        if name not in ("rf", "xgb") and name not in NETS:
            LOG.warning(f"unknown baseline {name!r}, skipped")
            continue
        model_path, res_path = out / f"model_{name}.pt", out / f"results_{name}.json"
        if a.fresh:
            model_path.unlink(missing_ok=True)
            res_path.unlink(missing_ok=True)
        set_seed(cfg["seed"])

        saved = torch.load(model_path, weights_only=False, map_location=device) if model_path.exists() else None
        if saved is not None and saved.get("source") == model_src:
            if name in NETS:
                model = NETS[name](tf.out_dim, store.n_classes).to(device)
                model.load_state_dict(saved["model"])
                model.eval()
            else:
                model = saved["model"]
            LOG.info(f"  {name}: loaded already-trained model")
        else:
            if name in ("rf", "xgb"):
                xtr = transform_chunked(tf, tr["num_raw"], tr["cat"], device).cpu().numpy()
                model = train_tree(name, xtr, tr["y"].numpy(), cfg["seed"], store.n_classes)
                del xtr
                atomic_save({"model": model, "source": model_src}, model_path)
            else:
                model = train_net(name, tf, tr, va, cfg, device, counts, tm)
                atomic_save({"model": model.state_dict(), "source": model_src}, model_path)

        res = {}
        if res_path.exists():
            prev = json.loads(res_path.read_text())
            if prev.get("_source") == res_src:
                res = prev
        res["_source"] = res_src
        res = evaluate_flow_model(name, model, tf, te, cfg, device, tm, not a.no_attacks, res,
                                  lambda r, pth=res_path: atomic_save_json(r, pth))
        atomic_save_json(res, res_path)
        for key, m in res.items():
            if key.startswith("_"):
                continue
            if key == "clean":
                threat, attack, eps = "-", "-", 0.0
            else:
                threat, attack, e = key.split("/")
                eps = float(e.split("=")[1])
            rows.append({"setting": name, "threat": threat, "attack": attack, "eps": eps, **m})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_summary(rows, out / "summary_baselines.csv")
    LOG.info(f"saved {out / 'summary_baselines.csv'}")


if __name__ == "__main__":
    main()
