"""How much do added flows raise the detection score D of the flows already there? (CPU only)

Compares, flow by flow, the scores that stgids.score_dump saved for clean/sub (the attack subset
without injection) with each injection-only setting (random or adaptive injection, no feature
attack): the same flows in the same order, so D_after - D_before is the change caused by the added
flows alone. GPU scores of identical inputs differ by up to ~5e-7 between runs (atomic additions),
so a change of at most --tol (default 1e-5) counts as "no change".

Per run, model, setting and flow group (benign / attack): share of flows with no change, with a
rise, with a fall (must be 0 up to noise: the guarantee), quantiles of the change, new alerts at
threshold 0 (D_before <= 0 < D_after; for benign flows these are new false alarms) and lost alerts
(D_before > 0 >= D_after; must be 0).

Usage:
    python scripts/score_rise.py runs/threshold_sweep/cic2018_full*/stgids_mono8 \
        runs/threshold_sweep/cic2018_full*/stgids_mono8_hardzero --out runs/threshold_sweep/cic2018_score_rise.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

QUANTILES = [0.5, 0.9, 0.99, 0.999]


def load(d: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(d.glob("*.npz")):
        if not f.name.endswith(".tmp.npz"):
            z = np.load(f)
            out[str(z["key"])] = {k: z[k] for k in ("d", "y", "y_bin", "win")}
    return out


def rise_rows(d: Path, tol: float) -> list[dict]:
    res = load(d)
    if "clean/sub" not in res:
        sys.exit(f"{d}: no clean/sub scores")
    base = res["clean/sub"]
    rows = []
    for key, r in sorted(res.items()):
        if not (key.startswith(("random/", "adaptive/")) and key.count("/") == 2):
            continue  # clean settings, and feature attacks (they change the attack flows themselves)
        if not (np.array_equal(base["win"], r["win"]) and np.array_equal(base["y"], r["y"])):
            sys.exit(f"{d}: {key} does not hold the same flows as clean/sub")
        delta = r["d"].astype(np.float64) - base["d"].astype(np.float64)
        for group, sel in (("benign", base["y_bin"] == 0), ("attack", base["y_bin"] == 1)):
            x, b0, b1 = delta[sel], base["d"][sel], r["d"][sel]
            if not len(x):
                continue
            q = np.quantile(x, QUANTILES)
            rows.append({
                "run": d.parent.name, "model": d.name, "setting": key, "group": group, "flows": int(len(x)),
                "no_change": float((np.abs(x) <= tol).mean()), "rose": float((x > tol).mean()),
                "fell": float((x < -tol).mean()), "mean": float(x.mean()),
                **{f"p{100 * p:g}": float(v) for p, v in zip(QUANTILES, q)},
                "max": float(x.max()), "min": float(x.min()),
                "new_alerts": int(((b0 <= 0) & (b1 > 0)).sum()), "lost_alerts": int(((b0 > 0) & (b1 <= 0)).sum()),
            })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dirs", nargs="+", type=Path, help="score_dump output folders (<run>/<model>)")
    p.add_argument("--out", type=Path, required=True, help="new CSV file (refused if it exists)")
    p.add_argument("--tol", type=float, default=1e-5)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; choose a new name")
    rows = [r for d in a.dirs for r in rise_rows(d, a.tol)]
    if not rows:
        sys.exit("no injection settings found")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} rows (tol {a.tol:g})")
    # console view: per model, setting and group, summed / averaged over runs (seeds)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["setting"], r["group"]), []).append(r)
    print(f"{'model':24s} {'setting':24s} {'group':6s} runs  no-change   rose   fell   p99 rise  max rise"
          f"  new alerts  lost")
    for (model, setting, group), rs in sorted(groups.items()):
        n = sum(r["flows"] for r in rs)
        share = lambda k: sum(r[k] * r["flows"] for r in rs) / n  # noqa: E731
        print(f"{model:24s} {setting:24s} {group:6s} {len(rs):4d}  {share('no_change'):8.4f}  {share('rose'):6.4f}"
              f" {share('fell'):6.4f}  {np.mean([r['p99'] for r in rs]):9.4f} {max(r['max'] for r in rs):9.4f}"
              f"  {sum(r['new_alerts'] for r in rs):10d}  {sum(r['lost_alerts'] for r in rs):4d}")


if __name__ == "__main__":
    main()
