"""Stability-based feature selection: mutual information + SHAP over bootstrap resamples.

For each bootstrap sample of the *training* split we rank features by
(a) mutual information with the label and (b) mean |SHAP| of a gradient-boosted
tree model. A feature's stability score is the share of (run, method) pairs in
which it lands in the top-k set. Features with score >= min_freq are kept.
Unlike PCA this keeps the original, interpretable features.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from .utils import LOG


def _tree_importance(x: np.ndarray, y: np.ndarray, seed: int, use_shap: bool) -> np.ndarray:
    try:
        from xgboost import XGBClassifier
        classes = np.unique(y)
        y_enc = np.searchsorted(classes, y)
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.2, subsample=0.8,
                              tree_method="hist", random_state=seed, n_jobs=-1, verbosity=0)
        model.fit(x, y_enc)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed)
        model.fit(x, y)
        use_shap = False  # TreeExplainer on RF is slow; fall back to impurity importance
        LOG.info("xgboost not installed: using random-forest importance instead of SHAP")
    if use_shap:
        try:
            import shap
            sub = x[: min(len(x), 3000)]
            sv = shap.TreeExplainer(model).shap_values(sub)
            sv = np.abs(np.asarray(sv))
            # shapes differ across shap versions: (n, d), (n, d, C) or (C, n, d)
            if sv.ndim == 3 and sv.shape[0] != sub.shape[0]:
                sv = sv.transpose(1, 2, 0)
            imp = sv.mean(axis=0)
            return imp.mean(axis=-1) if imp.ndim == 2 else imp
        except ImportError:
            LOG.info("shap not installed: using tree gain importance instead")
    return np.asarray(model.feature_importances_, dtype=np.float64)


def select_features(x: np.ndarray, y: np.ndarray, names: list[str], scfg: dict, seed: int = 0) -> dict:
    d = x.shape[1]
    top_k = min(scfg["top_k"], d)
    rng = np.random.default_rng(seed)
    hits = np.zeros(d)
    runs = 0
    keep_rows = y >= 0
    x, y = x[keep_rows], y[keep_rows]
    for b in range(scfg["n_bootstrap"]):
        idx = rng.choice(len(x), size=min(scfg["sample_size"], len(x)), replace=True)
        xb, yb = x[idx], y[idx]
        mi = mutual_info_classif(xb, yb, random_state=seed + b)
        hits[np.argsort(-mi)[:top_k]] += 1
        runs += 1
        imp = _tree_importance(xb, yb, seed + b, scfg.get("use_shap", True))
        hits[np.argsort(-imp)[:top_k]] += 1
        runs += 1
        LOG.info(f"feature selection bootstrap {b + 1}/{scfg['n_bootstrap']} done")
    score = hits / runs
    selected = np.where(score >= scfg["min_freq"])[0]
    if len(selected) < scfg["min_features"]:
        selected = np.argsort(-score)[: scfg["min_features"]]
    selected = np.sort(selected)
    LOG.info(f"selected {len(selected)}/{d} features")
    return {"selected": selected.tolist(),
            "stability": {n: float(s) for n, s in zip(names, score)}}
