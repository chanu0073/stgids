"""Combine several seed runs into one results table (mean ± std).

Usage:
    python scripts/aggregate_seeds.py runs/unsw runs/unsw_s1 runs/unsw_s2 --out runs/unsw_seeds

Each run folder is the output_dir given to scripts/run_main.sh. Writes <out>.csv (every number)
and <out>.tex (the paper table), and prints the table.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

GRAPH_MODELS = ["stgids", "stgids_adv", "egraphsage", "stgids_injaug", "stgids_flowdet", "stgids_mono"]
FLOW_MODELS = ["rf", "xgb", "mlp", "mlp_adv", "bigru", "aegru"]
NAMES = {"stgids": "Proposed", "stgids_adv": "Proposed + adv. training", "egraphsage": "E-GraphSAGE",
         "stgids_injaug": "STGIDS + injection augmentation", "stgids_flowdet": "Flow detector + STGIDS types",
         "stgids_mono": "Monotone detector + STGIDS types",
         "rf": "Random Forest", "xgb": "XGBoost", "mlp": "DNN", "mlp_adv": "DNN + adv. training",
         "bigru": "BiGRU", "aegru": "AE-GRU"}

# (column label, result key, metric)
COLUMNS = [
    ("Clean accuracy", "clean", "accuracy"),
    ("Clean macro-P", "clean", "macro_precision"),
    ("Clean macro-R", "clean", "macro_recall"),
    ("Clean macro-F1", "clean", "macro_f1"),
    ("Clean FPR", "clean", "fpr"),
    ("Constr. Square macro-F1", "constrained/square/eps=0.1", "macro_f1"),
    ("Unconstr. Square macro-F1", "unconstrained/square/eps=0.1", "macro_f1"),
    ("Unconstr. Square DR", "unconstrained/square/eps=0.1", "detection_rate"),
    ("Constr. PGD macro-F1", "constrained/pgd/eps=0.1", "macro_f1"),
    ("Unconstr. PGD macro-F1", "unconstrained/pgd/eps=0.1", "macro_f1"),
]


def result_file(run: Path, model: str) -> Path:
    if model in GRAPH_MODELS:
        return run / model / "results_test.json"
    return run / "baselines" / f"results_{model}.json"


def mean_std(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="one run folder per seed")
    p.add_argument("--out", default="runs/seeds_summary")
    a = p.parse_args()
    runs = [Path(r) for r in a.runs]

    table, per_class = [], {}
    for model in GRAPH_MODELS + FLOW_MODELS:
        loaded = []
        for run in runs:
            f = result_file(run, model)
            if f.exists():
                loaded.append(json.loads(f.read_text()))
            else:
                print(f"warning: missing {f}")
        if not loaded:
            continue
        row = {"model": model, "seeds": len(loaded)}
        for label, key, metric in COLUMNS:
            vals = [r[key][metric] for r in loaded if key in r and r[key].get(metric) is not None]
            row[label] = mean_std(vals) if len(vals) == len(loaded) else None  # only if every seed has it
        table.append(row)
        if model in GRAPH_MODELS:
            classes = loaded[0]["clean"].get("per_class_f1", {})
            per_class[model] = {c: mean_std([r["clean"]["per_class_f1"][c] for r in loaded
                                             if r["clean"].get("per_class_f1", {}).get(c) is not None] or [0.0])
                                for c in classes}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "seeds"] + [f"{lab} {s}" for lab, _, _ in COLUMNS for s in ("mean", "std")])
        for r in table:
            w.writerow([r["model"], r["seeds"]] + [x for lab, _, _ in COLUMNS
                        for x in ((f"{r[lab][0]:.4f}", f"{r[lab][1]:.4f}") if r[lab] else ("", ""))])

    def cell(v):
        return "--" if v is None else f"{v[0]:.3f} $\\pm$ {v[1]:.3f}"
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Results on NF-UNSW-NB15-v3 (mean $\pm$ std over {len(runs)} seeds, "
             r"$\epsilon=0.1$). DR: detection rate. PGD needs gradients, so it is not applicable (--) "
             r"to tree models.}",
             r"\label{tab:results}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}", r"\toprule",
             "Model & " + " & ".join(lab for lab, _, _ in COLUMNS) + r" \\", r"\midrule"]
    for r in table:
        lines.append(NAMES.get(r["model"], r["model"]) + " & " +
                     " & ".join(cell(r[lab]) for lab, _, _ in COLUMNS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    out.with_suffix(".tex").write_text("\n".join(lines) + "\n")

    # console view
    w0 = max(len(NAMES.get(r["model"], r["model"])) for r in table)
    print(f"{'model':<{w0}}  seeds  " + "  ".join(f"{lab:>25}" for lab, _, _ in COLUMNS))
    for r in table:
        vals = "  ".join(f"{('--' if r[lab] is None else f'{r[lab][0]:.3f} ± {r[lab][1]:.3f}'):>25}"
                         for lab, _, _ in COLUMNS)
        print(f"{NAMES.get(r['model'], r['model']):<{w0}}  {r['seeds']:>5}  {vals}")
    print("\nper-class F1 (clean, mean ± std):")
    for m, d in per_class.items():
        print(f"  {NAMES[m]}: " + ", ".join(f"{c} {v[0]:.3f}±{v[1]:.3f}" for c, v in d.items()))
    print(f"\nwrote {out.with_suffix('.csv')} and {out.with_suffix('.tex')}")


if __name__ == "__main__":
    main()