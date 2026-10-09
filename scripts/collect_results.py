"""Collect result JSON files from run folders into one CSV and a LaTeX table.

Usage:
    python scripts/collect_results.py runs/unsw --attack constrained/pgd/eps=0.1 \
        --attack2 unconstrained/pgd/eps=0.1 --out runs/unsw/table
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def fmt(v):
    return "--" if v is None else f"{v:.3f}"


def rows_from(path: Path, name: str, a1: str, a2: str) -> dict:
    r = json.loads(path.read_text())
    clean = r["clean"]
    return {"model": name,
            "macro_f1": clean.get("macro_f1"), "weighted_f1": clean.get("weighted_f1"),
            "pr_auc": clean.get("macro_pr_auc"), "fpr": clean.get("fpr"),
            "robust_constrained": (r.get(a1) or {}).get("macro_f1"),
            "robust_unconstrained": (r.get(a2) or {}).get("macro_f1")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--attack", default="constrained/pgd/eps=0.1")
    p.add_argument("--attack2", default="unconstrained/pgd/eps=0.1")
    p.add_argument("--tag", default="test")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    root = Path(a.root)
    rows = []
    for f in sorted(root.glob(f"*/results_{a.tag}.json")):
        rows.append(rows_from(f, f.parent.name, a.attack, a.attack2))
    for f in sorted((root / "baselines").glob("results_*.json")):
        rows.append(rows_from(f, f.stem.replace("results_", ""), a.attack, a.attack2))
    if not rows:
        raise SystemExit(f"no results_{a.tag}.json found under {root}")
    out = Path(a.out or root / "table")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Clean and robust results (robust: {a.attack} / {a.attack2}).}}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Model & Macro-F1 & W-F1 & PR-AUC & FPR & Rob. (constr.) & Rob. (unconstr.) \\", r"\midrule"]
    for r in rows:
        name = r["model"].replace("_", "\\_")
        lines.append(f"{name} & {fmt(r['macro_f1'])} & {fmt(r['weighted_f1'])} & "
                     f"{fmt(r['pr_auc'])} & {fmt(r['fpr'])} & {fmt(r['robust_constrained'])} & "
                     f"{fmt(r['robust_unconstrained'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.with_suffix(".tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out.with_suffix('.csv')} and {out.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
