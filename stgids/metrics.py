"""Metrics. Macro-F1 and per-class F1 are the headline numbers (accuracy hides minority classes)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score)


def compute_metrics(y: np.ndarray, y_bin: np.ndarray, pred: np.ndarray, prob: np.ndarray | None,
                    classes: list[str], benign_idx: int = 0) -> dict:
    """y: class index (-1 = class unseen in training); pred: predicted class index."""
    out: dict = {"n_flows": int(len(y)), "n_unseen_class_flows": int((y < 0).sum())}
    seen = y >= 0
    labels = list(range(len(classes)))
    present = sorted(set(y[seen].tolist()))
    if seen.any():
        ys, ps = y[seen], pred[seen]
        out["accuracy"] = float(accuracy_score(ys, ps))
        # macro over classes that occur in this split's ground truth
        out["macro_f1"] = float(f1_score(ys, ps, labels=present, average="macro", zero_division=0))
        out["weighted_f1"] = float(f1_score(ys, ps, labels=present, average="weighted", zero_division=0))
        out["macro_precision"] = float(precision_score(ys, ps, labels=present, average="macro", zero_division=0))
        out["macro_recall"] = float(recall_score(ys, ps, labels=present, average="macro", zero_division=0))
        f1s = f1_score(ys, ps, labels=labels, average=None, zero_division=0)
        out["per_class_f1"] = {classes[c]: (float(f1s[c]) if c in present else None) for c in labels}
        out["support"] = {classes[c]: int((ys == c).sum()) for c in labels}
        if prob is not None:
            aps = []
            for c in present:
                yc = (ys == c).astype(int)
                if 0 < yc.sum() < len(yc):
                    aps.append(average_precision_score(yc, prob[seen][:, c]))
            out["macro_pr_auc"] = float(np.mean(aps)) if aps else None
    # binary view: any attack class counts as "attack" (also covers unseen classes)
    pb = (pred != benign_idx).astype(int)
    benign = y_bin == 0
    out["fpr"] = float(pb[benign].mean()) if benign.any() else None
    out["detection_rate"] = float(pb[~benign].mean()) if (~benign).any() else None
    out["binary_f1"] = float(f1_score(y_bin, pb, zero_division=0))
    if prob is not None and 0 < y_bin.sum() < len(y_bin):
        out["binary_pr_auc"] = float(average_precision_score(y_bin, 1 - prob[:, benign_idx]))
    return out


def short(m: dict) -> str:
    f = lambda k: "n/a" if m.get(k) is None else f"{m[k]:.4f}"  # noqa: E731
    return (f"macroF1 {f('macro_f1')} | wF1 {f('weighted_f1')} | PR-AUC {f('macro_pr_auc')} | "
            f"FPR {f('fpr')} | DR {f('detection_rate')}")
