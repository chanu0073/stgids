"""Combine the ablation runs of several seeds into one table (mean ± std).

Usage:
    python scripts/aggregate_ablations.py runs/unsw runs/unsw_s1 runs/unsw_s2 --out runs/unsw_ablations

Each run folder is the output_dir given to scripts/run_ablations.sh (it holds stgids/, stgids_adv/
and ablation_<name>/). Writes <out>.csv (every number) and <out>.tex (the paper table), and prints
the table with the difference of every ablation to its reference model.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

# (folder, table name, reference folder); the reference is the model the ablation changes one thing in
VARIANTS = [
    ("stgids", "Proposed (60 s, Transformer, focal)", None),
    ("ablation_no_temporal", "w/o temporal encoder", "stgids"),
    ("ablation_gru", "GRU temporal encoder", "stgids"),
    ("ablation_loss_ce", "Cross-entropy loss", "stgids"),
    ("ablation_loss_weighted_ce", "Weighted cross-entropy loss", "stgids"),
    ("ablation_pretrain", "+ DGI pretraining", "stgids"),
    ("ablation_window_30s", "30 s windows", "stgids"),
    ("stgids_adv", "Proposed + adv. training (constrained PGD)", None),
    ("ablation_adv_unconstrained", "Adv. training with unconstrained PGD", "stgids_adv"),
]

# (column label, result key); every column is macro-F1
COLUMNS = [
    ("Clean", "clean"),
    ("Constr. Square", "constrained/square/eps=0.1"),
    ("Constr. PGD", "constrained/pgd/eps=0.1"),
    ("Unconstr. Square", "unconstrained/square/eps=0.1"),
    ("Unconstr. PGD", "unconstrained/pgd/eps=0.1"),
    ("Unconstr. FGSM", "unconstrained/fgsm/eps=0.1"),
]


def mean_std(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="one run folder per seed")
    p.add_argument("--out", default="runs/ablations_summary")
    a = p.parse_args()
    runs = [Path(r) for r in a.runs]

    table, per_class = {}, {}
    for folder, _, _ in VARIANTS:
        loaded = []
        for run in runs:
            f = run / folder / "results_test.json"
            if f.exists():
                loaded.append(json.loads(f.read_text()))
            else:
                print(f"warning: missing {f}")
        if not loaded:
            continue
        row = {"seeds": len(loaded)}
        for label, key in COLUMNS:
            vals = [r[key]["macro_f1"] for r in loaded if key in r]
            row[label] = mean_std(vals) if len(vals) == len(loaded) else None  # only if every seed has it
        table[folder] = row
        classes = loaded[0]["clean"].get("per_class_f1", {})
        per_class[folder] = {c: statistics.mean([r["clean"]["per_class_f1"][c] or 0.0 for r in loaded])
                             for c in classes}

    def delta(folder, ref, label):
        if ref not in table or table[folder][label] is None or table[ref][label] is None:
            return None
        return table[folder][label][0] - table[ref][label][0]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "reference", "seeds"]
                   + [f"{lab} {s}" for lab, _ in COLUMNS for s in ("mean", "std", "delta")])
        for folder, _, ref in VARIANTS:
            if folder not in table:
                continue
            r = table[folder]
            cells = []
            for lab, _ in COLUMNS:
                d = delta(folder, ref, lab)
                cells += ([f"{r[lab][0]:.4f}", f"{r[lab][1]:.4f}"] if r[lab] else ["", ""]) \
                    + [f"{d:+.4f}" if d is not None else ""]
            w.writerow([folder, ref or "", r["seeds"]] + cells)

    def cell(v):
        return "--" if v is None else f"{v[0]:.3f} $\\pm$ {v[1]:.3f}"
    n_seeds = max(r["seeds"] for r in table.values())
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Ablation study on NF-UNSW-NB15-v3: macro-F1 (mean $\pm$ std over {n_seeds} seeds, "
             r"$\epsilon=0.1$). Each ablation changes one component of the model above it.}",
             r"\label{tab:ablation}",
             r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}", r"\toprule",
             "Variant & " + " & ".join(lab for lab, _ in COLUMNS) + r" \\", r"\midrule"]
    first = True
    for folder, name, ref in VARIANTS:
        if folder not in table:
            continue
        if ref is None and not first:  # separate the adversarially trained group
            lines.append(r"\midrule")
        first = False
        prefix = r"\quad " if ref else ""
        lines.append(prefix + name + " & " + " & ".join(cell(table[folder][lab]) for lab, _ in COLUMNS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out.with_suffix(".tex").write_text("\n".join(lines) + "\n")

    # console view: mean ± std, and the change against the reference model in brackets
    w0 = max(len(name) for _, name, _ in VARIANTS)
    print(f"{'variant':<{w0}}  seeds  " + "  ".join(f"{lab:>24}" for lab, _ in COLUMNS))
    for folder, name, ref in VARIANTS:
        if folder not in table:
            continue
        cells = []
        for lab, _ in COLUMNS:
            v, d = table[folder][lab], delta(folder, ref, lab)
            txt = "--" if v is None else f"{v[0]:.3f} ± {v[1]:.3f}" + (f" ({d:+.3f})" if d is not None else "")
            cells.append(f"{txt:>24}")
        print(f"{name:<{w0}}  {table[folder]['seeds']:>5}  " + "  ".join(cells))
    print("\nper-class F1 (clean, mean over seeds):")
    for folder, name, _ in VARIANTS:
        if folder in per_class:
            print(f"  {name}: " + ", ".join(f"{c} {v:.3f}" for c, v in per_class[folder].items()))
    print(f"\nwrote {out.with_suffix('.csv')} and {out.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
