"""Combine the injection results (results_inject.json) of several seed runs into one table.

Usage:
    python scripts/aggregate_inject.py runs/unsw runs/unsw_s1 runs/unsw_s2 --out runs/unsw_inject

Each run folder holds one sub-folder per model (stgids, egraphsage, stgids_mono, ...). Writes
<out>.csv (macro-F1, detection rate and injected-flow alarm rate of every setting, mean and std)
and <out>.tex (macro-F1 with the detection rate in brackets), and prints the table. Models without
temporal context are only attacked in the target window; history injection cannot change their
predictions, so their "hist" columns repeat the target-window result.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

MODELS = ["stgids", "stgids_adv", "ablation_no_temporal", "egraphsage", "stgids_injaug", "stgids_flowdet",
          "stgids_mono", "stgids_injaug_mono", "stgids_injaug_mono8", "stgids_mono8", "stgids_mono8_or",
          "mlpadv_mono8", "flow_mlp", "flow_mlp_adv", "flow_bigru", "flow_aegru"]
NAMES = {"stgids": "Proposed (STGIDS)", "stgids_adv": "STGIDS + adv. training",
         "ablation_no_temporal": "STGIDS w/o temporal encoder", "egraphsage": "E-GraphSAGE",
         "stgids_injaug": "STGIDS + injection augmentation", "stgids_flowdet": "Flow detector + STGIDS types",
         "stgids_mono": "Monotone detector + STGIDS types",
         "stgids_injaug_mono": "Monotone detector + InjAug STGIDS types",
         "stgids_injaug_mono8": "Monotone detector (8 ch.) + InjAug STGIDS types",
         "stgids_mono8": "Monotone detector (8 ch.) + STGIDS types",
         "stgids_mono8_or": "Monotone detector (8 ch.) OR STGIDS",
         "mlpadv_mono8": "Monotone detector (8 ch.) + flow types (DNN adv.)",
         "flow_mlp": "DNN (flow)", "flow_mlp_adv": "DNN + adv. training (flow)",
         "flow_bigru": "BiGRU (flow)", "flow_aegru": "AE-GRU (flow)"}
SETTINGS = [  # (column label, result key)
    ("Clean", "clean"),
    ("Random hist b=100", "random/hist/beta=100"),
    ("Adaptive now b=20", "adaptive/now/beta=20"),
    ("Adaptive hist b=20", "adaptive/hist/beta=20"),
    ("Adaptive hist b=100", "adaptive/hist/beta=100"),
    ("Adaptive hist b=20 + constr. eps=0.1", "adaptive/hist/beta=20/constrained:0.1"),
    ("Adaptive hist b=20 + unconstr. eps=0.1", "adaptive/hist/beta=20/unconstrained:0.1"),
    ("Wrong type hist b=20", "adaptive/hist/beta=20/untargeted"),
]
METRICS = [("macro_f1", "F1"), ("detection_rate", "DR"), ("injected_flagged_rate", "inj. flagged")]


def lookup(res: dict, key: str):
    if key in res:
        return res[key]
    return res.get(key.replace("/hist/", "/now/"))  # model without history


def mean_std(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="one run folder per seed")
    p.add_argument("--out", default="runs/inject_summary")
    a = p.parse_args()
    rows = []
    for m in MODELS:
        loaded = [json.loads(f.read_text()) for r in a.runs if (f := Path(r) / m / "results_inject.json").exists()]
        if not loaded:
            continue
        row = {"model": m, "seeds": len(loaded)}
        for label, key in SETTINGS:
            for met, _ in METRICS:
                vals = [v[met] for res in loaded if (v := lookup(res, key)) and v.get(met) is not None]
                row[(label, met)] = mean_std(vals) if len(vals) == len(loaded) else None
        rows.append(row)
    if not rows:
        raise SystemExit("no results_inject.json found")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "seeds"] + [f"{lab} {name} {s}" for lab, _ in SETTINGS for _, name in METRICS
                                         for s in ("mean", "std")])
        for r in rows:
            w.writerow([r["model"], r["seeds"]] + [x for lab, _ in SETTINGS for met, _ in METRICS
                                                   for x in ((f"{r[(lab, met)][0]:.4f}", f"{r[(lab, met)][1]:.4f}")
                                                             if r[(lab, met)] else ("", ""))])

    def cell(r, lab):
        f1, dr = r[(lab, "macro_f1")], r[(lab, "detection_rate")]
        return "--" if f1 is None else f"{f1[0]:.3f} ({dr[0]:.3f})"
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Macro-F1 (detection rate in brackets) under flow injection, mean over {len(a.runs)} seeds. "
             r"$b$: injected flows per attacker host and window; now: target window only; hist: also the "
             r"history windows. Models without temporal context are attacked in the target window only.}",
             r"\label{tab:inject}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(SETTINGS) + "}", r"\toprule",
             "Model & " + " & ".join(lab for lab, _ in SETTINGS) + r" \\", r"\midrule"]
    lines += [NAMES.get(r["model"], r["model"]) + " & " + " & ".join(cell(r, lab) for lab, _ in SETTINGS) + r" \\"
              for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    out.with_suffix(".tex").write_text("\n".join(lines) + "\n")

    w0 = max(len(NAMES.get(r["model"], r["model"])) for r in rows)
    print(f"{'model':<{w0}}  seeds  " + "  ".join(f"{lab:>22}" for lab, _ in SETTINGS))
    for r in rows:
        print(f"{NAMES.get(r['model'], r['model']):<{w0}}  {r['seeds']:>5}  "
              + "  ".join(f"{cell(r, lab):>22}" for lab, _ in SETTINGS))
    print(f"\nwrote {out.with_suffix('.csv')} and {out.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
