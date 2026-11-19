"""Save the detection score D of every flow, clean and under attack, for threshold analysis.

A detect-attribute model flags a flow when D > 0 (logits column 0 is 0, the best attack column is D).
This script stores D, the attribution model's attack type, and the labels, so that
scripts/threshold_sweep.py can evaluate any threshold tau on CPU without rerunning the model.
It only reads the run directory; everything goes to --out (a new directory), one .npz per setting.
An existing .npz is kept and its setting skipped (resume); nothing is ever overwritten.

Settings (the same as evaluate_inject.py, with the same seeds, so injections are identical):
  clean/val, clean/test           all validation and test windows
  clean/sub                       the --max-windows test subset used by the attacks
  random/<mode>/beta=B            benign-template injection
  adaptive/<mode>/beta=B[/feat]   PGD-optimised injection, optionally with a feature attack
  feature/now/beta=0/<feat>       the feature attack alone
The PGD evasion objective maximises log P(benign) = -softplus(D), which keeps lowering D however
low it already is, so one attack run is valid for every threshold.

Usage:
    python -m stgids.score_dump --config runs/cic2018_full/stgids_mono8/config.yaml \
        --data runs/cic2018_s1/prepared.pt --out runs/threshold_sweep/cic2018_full/stgids_mono8 \
        --max-windows 220
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .adversarial import ThreatModel
from .config import load_config
from .evaluate_inject import uses_history
from .graphs import SnapshotStore
from .inject import attack_window, benign_input_range, inject_window
from .model import build_model
from .utils import LOG, base_parser, file_id, get_device, set_seed, setup_logging


def scores(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(D, attack type) from detect-attribute logits [0, D + typ_1, ..., D + typ_C] with max typ = 0."""
    att = logits[:, 1:]
    d, typ = att.max(1)
    return d, typ + 1


def run_setting(model, store, targets, device, beta, mode, adaptive, lo, hi, seed, steps,
                feature=None, objective="evasion") -> dict[str, np.ndarray]:
    """D and type of the original flows (and of the injected flows of the target window)."""
    model.eval()
    out = {k: [] for k in ("d", "typ", "y", "y_bin", "win", "d_inj")}
    for t in targets:
        gen = torch.Generator(device=device).manual_seed(seed * 100_003 + t)  # as evaluate_inject.py
        window, counts = inject_window(store.window(t, device), beta, mode, gen)
        tgt = window[-1]
        e = len(tgt.y) - counts[-1]
        if adaptive or feature is not None:
            xs = attack_window(model, window, counts, lo, hi, steps=steps, feature=feature, objective=objective)
            with torch.no_grad():
                lg = model.forward_inputs(window, xs)
        else:
            with torch.no_grad():
                lg = model(window)
        d, typ = scores(lg)
        out["d"].append(d[:e].cpu()); out["typ"].append(typ[:e].cpu())
        out["y"].append(tgt.y[:e].cpu()); out["y_bin"].append(tgt.y_bin[:e].cpu())
        out["win"].append(torch.full((e,), t, dtype=torch.int32))
        out["d_inj"].append(d[e:].cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    p = base_parser("save per-flow detection scores of a detect-attribute model (for threshold analysis)")
    p.add_argument("--checkpoint", help="model.pt (default: <output_dir>/model.pt)")
    p.add_argument("--data", help="prepared data (default: <output_dir>/prepared.pt)")
    p.add_argument("--out", required=True, help="new directory for the .npz files (never the run directory)")
    p.add_argument("--betas", type=int, nargs="+", default=[20, 100])
    p.add_argument("--modes", nargs="+", default=["hist"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--max-windows", type=int, help="attack an evenly spaced subset of the test windows")
    p.add_argument("--feature", action="append", default=[], help="e.g. unconstrained:0.1 (repeatable)")
    p.add_argument("--feature-betas", type=int, nargs="+", default=[20],
                   help="betas at which injection is combined with the feature attacks")
    p.add_argument("--clean-only", action="store_true", help="only clean/val, clean/test, clean/sub")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    run_dir = Path(cfg["output_dir"]).resolve()
    out = Path(a.out).resolve()
    if out == run_dir or out in run_dir.parents:
        raise SystemExit("--out must be a new directory, not the run directory")
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out, "score_dump")
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    ckpt_path = Path(a.checkpoint or run_dir / "model.pt")
    data_path = Path(a.data or run_dir / "prepared.pt")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    mcfg = ckpt["config"]
    if mcfg["model"]["arch"] != "detect_attribute":
        raise SystemExit(f"{ckpt_path}: arch {mcfg['model']['arch']!r} has no detection score D")
    store = SnapshotStore(str(data_path), history_k=mcfg["graph"]["history_k"])
    model = build_model(ckpt["meta"], mcfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    source = f"{file_id(ckpt_path)}|{file_id(data_path)}|steps={a.steps}|seed={cfg['seed']}"
    src_file = out / "source.txt"
    if src_file.exists() and src_file.read_text().strip() != source:
        raise SystemExit(f"{out} holds scores of another model or data ({src_file}); use a new --out")
    src_file.write_text(source + "\n")

    test = store.targets("test")
    sub = test
    if a.max_windows and a.max_windows < len(test):
        stride = len(test) / a.max_windows
        sub = [test[int(i * stride)] for i in range(a.max_windows)]
    wanted = [("clean/val", store.targets("val"), 0, "now", False, None),
              ("clean/test", test, 0, "now", False, None),
              ("clean/sub", sub, 0, "now", False, None)]
    if not a.clean_only:
        modes = a.modes if uses_history(mcfg) else ["now"]
        for mode in modes:
            for beta in a.betas:
                wanted.append((f"random/{mode}/beta={beta}", sub, beta, mode, False, None))
                wanted.append((f"adaptive/{mode}/beta={beta}", sub, beta, mode, True, None))
                for f in (a.feature if beta in a.feature_betas else []):
                    wanted.append((f"adaptive/{mode}/beta={beta}/{f}", sub, beta, mode, True, f))
        for f in a.feature:
            wanted.append((f"feature/now/beta=0/{f}", sub, 0, "now", False, f))

    lo = hi = tm = None
    for key, targets, beta, mode, adaptive, feat in wanted:
        path = out / (key.replace("/", "__").replace(":", "_") + ".npz")
        if path.exists():
            LOG.info(f"{key}: {path.name} exists, skipped")
            continue
        if (adaptive or feat) and lo is None:
            lo, hi = benign_input_range(model, store)
            tm = ThreatModel(cfg["threat_model"], store.meta["num_names"])
        feature = None
        if feat:
            threat, eps = feat.split(":")
            feature = (threat, tm, float(eps))
        r = run_setting(model, store, targets, device, beta, mode, adaptive, lo, hi, cfg["seed"], a.steps, feature)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **r, key=np.array(key), classes=np.array(store.meta["classes"]))
        tmp.replace(path)
        att = r["y_bin"] == 1
        LOG.info(f"{key}: {len(r['d'])} flows ({int(att.sum())} attack), detection at 0 "
                 f"{(r['d'][att] > 0).mean() if att.any() else float('nan'):.4f}, "
                 f"FPR at 0 {(r['d'][~att] > 0).mean():.5f}, injected {len(r['d_inj'])}")
    LOG.info(f"done: {out}")


if __name__ == "__main__":
    main()
