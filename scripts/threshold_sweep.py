"""Threshold analysis of the detection score D (CPU only, reads the .npz files of stgids.score_dump).

Thresholds:
  tau = 0               the untuned default (sigmoid(D) = 0.5 in the training loss)
  val FPR targets       tau chosen on the CLEAN VALIDATION benign flows only, so that their false
                        alarm rate is at most the target; then fixed and applied to test and attacks
  fixed grid            a few fixed values, for the trade-off curve
For every threshold and setting: detection rate, FPR, macro-F1 (flagged flows take the attribution
model's type), the rate at which injected flows are flagged, and, for injection-only settings, a
check that no attack flow's D fell below its clean D (the guarantee; per window, same flow order).

Usage:
    python scripts/threshold_sweep.py runs/threshold_sweep/cic2018_full*/stgids_mono8 \
        --out runs/threshold_sweep/cic2018_stgids_mono8.csv
Several directories (seeds) give one row per directory plus mean and std over directories.
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

FPR_TARGETS = [0.0005, 0.001, 0.002, 0.005, 0.01]
GRID = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]


def load(d: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(d.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        out[str(z["key"])] = {k: z[k] for k in z.files}
    return out


def tau_for_fpr(d_benign: np.ndarray, target: float) -> float:
    """Smallest tau with mean(d_benign > tau) <= target."""
    s = np.sort(d_benign)
    k = int(np.floor(target * len(s)))  # at most k benign flows may lie above tau
    return float(s[len(s) - k - 1]) if k < len(s) else float(s[0]) - 1e-6


def metrics(r: dict, tau: float) -> dict:
    flag = r["d"] > tau
    pred = np.where(flag, r["typ"], 0)
    y, att = r["y"], r["y_bin"] == 1
    seen = y >= 0
    present = sorted(set(y[seen].tolist()))
    return {
        "detection": float(flag[att].mean()) if att.any() else np.nan,
        "fpr": float(flag[~att].mean()) if (~att).any() else np.nan,
        "macro_f1": float(f1_score(y[seen], pred[seen], labels=present, average="macro", zero_division=0)),
        "injected_flagged": float((r["d_inj"] > tau).mean()) if len(r["d_inj"]) else np.nan,
    }


def guarantee_check(clean: dict, att: dict) -> tuple[int, int, float]:
    """(attack flows compared, flows whose D fell by more than 1e-4, largest fall)."""
    if not (np.array_equal(clean["win"], att["win"]) and np.array_equal(clean["y"], att["y"])):
        return 0, -1, np.nan  # flow lists differ: cannot compare flow by flow
    a = clean["y_bin"] == 1
    drop = clean["d"][a] - att["d"][a]
    return int(a.sum()), int((drop > 1e-4).sum()), float(drop.max()) if a.any() else 0.0


def analyse(d: Path) -> list[dict]:
    res = load(d)
    if "clean/val" not in res:
        sys.exit(f"{d}: no clean/val scores")
    val = res["clean/val"]
    taus = [("zero", 0.0)]
    taus += [(f"valfpr={t:g}", tau_for_fpr(val["d"][val["y_bin"] == 0], t)) for t in FPR_TARGETS]
    taus += [(f"fixed={t:g}", t) for t in GRID]
    rows = []
    for key, r in res.items():
        injection_only = key.startswith(("random/", "adaptive/")) and key.count("/") == 2
        n, bad, worst = guarantee_check(res["clean/sub"], r) if injection_only and "clean/sub" in res \
            else (0, 0, np.nan)
        for name, tau in taus:
            rows.append({"run": str(d), "setting": key, "threshold": name, "tau": tau, **metrics(r, tau),
                         "guarantee_flows": n, "guarantee_violations": bad, "largest_drop": worst})
    return rows


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # mean/std of one run, all-nan columns
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True, help="new CSV file (refused if it exists)")
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; choose a new name")
    rows = [r for d in a.dirs for r in analyse(d)]
    # mean / std over runs (seeds) per setting and threshold
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["setting"], r["threshold"]), []).append(r)
    cols = ["tau", "detection", "fpr", "macro_f1", "injected_flagged"]
    summary = []
    for (setting, th), rs in groups.items():
        s = {"run": f"mean of {len(rs)}", "setting": setting, "threshold": th,
             "guarantee_flows": sum(r["guarantee_flows"] for r in rs),
             "guarantee_violations": sum(r["guarantee_violations"] for r in rs),
             "largest_drop": np.nanmax([r["largest_drop"] for r in rs]) if any(
                 not np.isnan(r["largest_drop"]) for r in rs) else np.nan}
        for c in cols:
            v = np.array([r[c] for r in rs], dtype=float)
            s[c], s[c + "_std"] = float(np.nanmean(v)), float(np.nanstd(v))
        summary.append(s)
    fields = ["run", "setting", "threshold"] + [x for c in cols for x in (c, c + "_std")] + \
             ["guarantee_flows", "guarantee_violations", "largest_drop"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows + summary)
    print(f"wrote {a.out}: {len(rows)} rows, {len(summary)} summary rows")
    # short console view: mean over runs, clean test and the strongest attacks
    show = [s for s in summary if s["setting"] in ("clean/test",) or "unconstrained" in s["setting"]
            or s["setting"].startswith("adaptive/hist/beta=100")]
    for s in sorted(show, key=lambda s: (s["setting"], s["tau"])):
        print(f"{s['setting']:42s} {s['threshold']:14s} tau={s['tau']:+.3f} det={s['detection']:.4f} "
              f"fpr={s['fpr']:.5f} mF1={s['macro_f1']:.3f} inj={s['injected_flagged']:.4f} "
              f"viol={s['guarantee_violations']}")


if __name__ == "__main__":
    main()
