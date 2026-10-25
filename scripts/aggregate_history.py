"""Combine the history-length sweep (scripts/run_history_sweep.sh) of several seeds into tables.

Usage:
    python scripts/aggregate_history.py runs/unsw_a100 runs/unsw_a100_s1 runs/unsw_a100_s2 --out runs/unsw_history

Each run folder holds history_k<K>/ for the swept K and stgids/ (the main model, K = 6), which is used
for K = 6 unless history_k6/ exists. Only the injection settings that the sweep runs for every K are
compared (beta 20 and 100, random and adaptive evasion, modes now and hist); with K = 1 there is no
history to shape, so "hist" equals "now". Writes <out>.csv (one row per K and seed) and <out>.md
(mean ± std over seeds, per-class F1); existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

SETTINGS = [f"{kind}/{mode}/beta={b}" for mode in ("now", "hist") for b in (20, 100)
            for kind in ("random", "adaptive")]


def mean_std(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def seconds_per_epoch(train_log: Path) -> float | None:
    """Median time between consecutive epochs in train.txt (robust to a resumed run)."""
    if not train_log.exists():
        return None
    times = {}
    for line in train_log.read_text().splitlines():
        m = re.match(r"(\d\d):(\d\d):(\d\d) \| epoch\s+(\d+) ", line)
        if m:
            h, mi, s, e = map(int, m.groups())
            times[e] = 3600 * h + 60 * mi + s
    gaps = [(times[e + 1] - times[e]) % 86400 for e in times if e + 1 in times]
    return statistics.median(gaps) if gaps else None


def load_run(run: Path, k: int) -> dict | None:
    folder = run / f"history_k{k}"
    if k == 6 and not folder.exists():
        folder = run / "stgids"
    clean_f = folder / "results_clean_test.json"
    if not clean_f.exists():
        print(f"missing {clean_f}")
        return None
    clean = json.loads(clean_f.read_text())
    hist = json.loads((folder / "train_history.json").read_text())["history"]
    best = max(hist, key=lambda e: e["val_macro_f1"])
    row = {"run": str(run), "K": k, "folder": folder.name, "best_epoch": best["epoch"],
           "val_macro_f1": best["val_macro_f1"], "macro_f1": clean["macro_f1"],
           "dr": clean["detection_rate"], "fpr": clean["fpr"],
           "latency_ms": clean.get("latency_ms_per_snapshot"),
           "s_per_epoch": seconds_per_epoch(folder / "train.txt")}
    for name, f1 in clean["per_class_f1"].items():
        row[f"f1/{name}"] = f1
    inj_f = folder / "results_inject.json"
    inj = json.loads(inj_f.read_text()) if inj_f.exists() else {}
    for s in SETTINGS:
        r = inj.get(s if not (k == 1 and "/hist/" in s) else s.replace("/hist/", "/now/"))
        if r is not None:
            row[f"{s}/macro_f1"], row[f"{s}/dr"] = r["macro_f1"], r["detection_rate"]
    return row


def fmt(xs: list[float], pct: bool = False, digits: int = 1) -> str:
    if not xs:
        return "–"
    m, s = mean_std([100 * x for x in xs] if pct else xs)
    return f"{m:.{digits}f} ± {s:.{digits}f}" if pct else f"{m:.3f} ± {s:.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="one run folder per seed")
    p.add_argument("--ks", default="1 3 6 12 24", help="history lengths to collect")
    p.add_argument("--out", required=True, help="output prefix, e.g. runs/unsw_history")
    a = p.parse_args()
    out_csv, out_md = Path(a.out + ".csv"), Path(a.out + ".md")
    for f in (out_csv, out_md):
        if f.exists():
            raise SystemExit(f"{f} exists - choose a new --out (existing results are never overwritten)")

    rows = [r for run in a.runs for k in map(int, a.ks.split()) if (r := load_run(Path(run), k))]
    if not rows:
        raise SystemExit("no results found")
    cols = list(dict.fromkeys(c for r in rows for c in r))

    ks = sorted({r["K"] for r in rows})
    by_k = {k: [r for r in rows if r["K"] == k] for k in ks}
    lines = ["Macro-F1 as 0–1, detection rate (DR) and FPR in %; mean ± std over seeds. "
             "Worst case = the injection setting with the lowest seed-mean macro-F1 "
             "(beta 20/100, random/adaptive, now/hist).", "",
             "| K (min) | seeds | val macro-F1 | test macro-F1 | DR (%) | FPR (%) | adaptive now β=20 | "
             "adaptive hist β=20 | worst case | worst-case DR (%) | ms/snapshot | s/epoch |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in ks:
        rs = by_k[k]
        col = lambda key: [r[key] for r in rs if r.get(key) is not None]
        worst = None
        for s in SETTINGS:
            xs = col(f"{s}/macro_f1")
            if len(xs) == len(rs) and (worst is None or statistics.mean(xs) < worst[1]):
                worst = (s, statistics.mean(xs))
        w_f1 = fmt(col(f"{worst[0]}/macro_f1")) + f" ({worst[0]})" if worst else "–"
        w_dr = fmt(col(f"{worst[0]}/dr"), pct=True) if worst else "–"
        lat, spe = col("latency_ms"), col("s_per_epoch")
        lat_s = f"{statistics.mean(lat):.1f}" if lat else "–"
        spe_s = f"{statistics.median(spe):.0f}" if spe else "–"
        lines.append(
            f"| {k} | {len(rs)} | {fmt(col('val_macro_f1'))} | {fmt(col('macro_f1'))} | "
            f"{fmt(col('dr'), True)} | {fmt(col('fpr'), True, 2)} | "
            f"{fmt(col('adaptive/now/beta=20/macro_f1'))} ({fmt(col('adaptive/now/beta=20/dr'), True)}) | "
            f"{fmt(col('adaptive/hist/beta=20/macro_f1'))} ({fmt(col('adaptive/hist/beta=20/dr'), True)}) | "
            f"{w_f1} | {w_dr} | {lat_s} | {spe_s} |")
    classes = [c[3:] for c in cols if c.startswith("f1/")]
    lines += ["", "Per-class test F1 (mean over seeds):", "",
              "| K | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for k in ks:
        vals = [[r[f"f1/{c}"] for r in by_k[k] if r.get(f"f1/{c}") is not None] for c in classes]
        lines.append(f"| {k} | " + " | ".join(f"{statistics.mean(v):.3f}" if v else "–" for v in vals) + " |")
    with open(out_csv, "w", newline="") as f:  # files are written only after everything was computed
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
