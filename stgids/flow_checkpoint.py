"""Wrap a trained neural flow baseline as a model.pt that evaluate_inject.py can attack.

The flow-vector baselines (baselines.py) are saved as bare network weights. This tool wraps one of
them in a FlowTyper (the feature transform of the prepared data + the network), which scores every
flow from its own features only, and saves it in the checkpoint format of the graph models. The
injection study can then attack the flow models with exactly the same injection and joint attacks,
windows and injected flows as the graph models; added flows cannot change their predictions.

Before saving, the wrapped model is evaluated on the clean test split and must reproduce the
macro-F1 that baselines.py saved for this network, so a wrong --data file is caught.

Usage:
    python -m stgids.flow_checkpoint --config configs/nf_cse_cic_ids2018_v3.yaml \
        --data runs/cic2018/prepared.pt --run runs/cic2018_full --net mlp_adv
Writes <run>/flow_<net>/model.pt (or --out); an existing file is never overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .baselines import NETS, test_flows, transform_chunked
from .config import load_config
from .detector import FlowTyper
from .graphs import SnapshotStore
from .metrics import compute_metrics
from .utils import LOG, atomic_save, base_parser, get_device, setup_logging


def main():
    p = base_parser("wrap a neural flow baseline for evaluate_inject.py")
    p.add_argument("--data", required=True, help="prepared.pt that the baseline was trained on")
    p.add_argument("--run", required=True, help="run folder with baselines/model_<net>.pt")
    p.add_argument("--net", required=True, choices=sorted(NETS))
    p.add_argument("--out", help="output path (default: <run>/flow_<net>/model.pt)")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    run = Path(a.run)
    out = Path(a.out or run / f"flow_{a.net}" / "model.pt")
    if out.exists():
        raise SystemExit(f"{out} already exists - not overwritten")
    setup_logging(out.parent, "flow_checkpoint")
    weights = run / "baselines" / f"model_{a.net}.pt"
    results = run / "baselines" / f"results_{a.net}.json"
    device = get_device(cfg["device"])

    store = SnapshotStore(a.data)
    model = FlowTyper(store.meta["transform"], store.n_classes, a.net).to(device)
    model.net.load_state_dict(torch.load(weights, weights_only=False, map_location=device)["model"])
    model.eval()

    # the wrapped model must reproduce the saved clean test result of the baseline
    te = test_flows(store, None)
    with torch.no_grad():
        x = transform_chunked(model.tf, te["num_raw"], te["cat"], device)
        probs = torch.cat([model.net(x[i:i + 4096]).softmax(1).cpu() for i in range(0, len(x), 4096)])
    m = compute_metrics(te["y"].numpy(), te["y_bin"].numpy(), probs.argmax(1).numpy(), probs.numpy(),
                        store.meta["classes"])
    saved = json.loads(results.read_text())["clean"]["macro_f1"]
    LOG.info(f"{a.net}: clean test macro-F1 {m['macro_f1']:.4f}, saved by baselines.py {saved:.4f}")
    if abs(m["macro_f1"] - saved) > 1e-4:
        raise SystemExit(f"clean macro-F1 does not match {results}: is {a.data} the data it was trained on?")

    atomic_save({"state_dict": model.state_dict(), "config": {**cfg, "model": {"arch": "flow", "net": a.net}},
                 "meta": store.meta}, out)
    LOG.info(f"saved {out}")


if __name__ == "__main__":
    main()
