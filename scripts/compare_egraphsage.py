"""Clean test results of E-GraphSAGE (authors' code), our E-GraphSAGE and STGIDS, mean ± std over seeds.

Usage:
    python scripts/compare_egraphsage.py --unsw runs/unsw_a100 runs/unsw_a100_s1 runs/unsw_a100_s2 \
        --cic runs/cic2018_full runs/cic2018_full_s1 runs/cic2018_full_s2
Reads <run>/<model>/results_clean_test.json; prints a table, writes nothing.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MODELS = [("stgids", "STGIDS"), ("egraphsage", "E-GraphSAGE (our re-implementation)"),
          ("egraphsage_authors", "E-GraphSAGE (authors' code)")]


def cell(runs: list[str], folder: str, key: str, pct: bool) -> str:
    xs = []
    for r in runs:
        f = Path(r) / folder / "results_clean_test.json"
        if f.exists():
            xs.append(json.loads(f.read_text())[key] * (100 if pct else 1))
    if not xs:
        return "–"
    m, s = statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)
    return (f"{m:.2f} ± {s:.2f}" if pct else f"{m:.3f} ± {s:.3f}") + f" (n={len(xs)})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unsw", nargs="+", default=[])
    p.add_argument("--cic", nargs="+", default=[])
    a = p.parse_args()
    print("| Dataset | Model | macro-F1 | DR (%) | FPR (%) |\n|---|---|---|---|---|")
    for name, runs in (("NF-UNSW-NB15-v3", a.unsw), ("NF-CSE-CIC-IDS2018-v3", a.cic)):
        for folder, label in MODELS:
            if runs:
                print(f"| {name} | {label} | {cell(runs, folder, 'macro_f1', False)} | "
                      f"{cell(runs, folder, 'detection_rate', True)} | {cell(runs, folder, 'fpr', True)} |")


if __name__ == "__main__":
    main()
