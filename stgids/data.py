"""Loading, cleaning, chronological splitting and encoding of NetFlow records.

Design rules (see paper, Section 3):
* exact duplicate rows are removed before anything else;
* the split is chronological and done on whole time windows, so no window is
  shared between train / val / test;
* every statistic (category vocabularies, scaling, feature selection) is fitted
  on the training split only;
* IP addresses and ports are never model features; IPs only define graph nodes.
"""
from __future__ import annotations

import difflib

import numpy as np
import pandas as pd

from .utils import LOG

TCP_BITS = 8
SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- loading
def _read_dtypes(columns: list[str], dcfg: dict) -> dict:
    """Compact dtypes: float32 for flow features, category for IPs/labels, float64 for time."""
    text = {dcfg["src_ip_col"], dcfg["dst_ip_col"], dcfg["label_col"]}
    dtypes = {}
    for c in columns:
        if c in text:
            dtypes[c] = "category"
        elif c == dcfg["time_col"]:
            dtypes[c] = "float64" if dcfg["time_unit"] in ("ms", "s") else "object"
        else:
            dtypes[c] = "float32"
    return dtypes


def load_flows(dcfg: dict) -> pd.DataFrame:
    """Read the CSV(s) with compact dtypes, skipping dropped columns, to keep RAM low."""
    paths = dcfg["csv"]
    if paths is None:
        raise ValueError("data.csv is not set in the config")
    if isinstance(paths, str):
        paths = [paths]
    frames = []
    remaining = dcfg.get("max_rows")
    drop = set(dcfg.get("drop_cols", []))
    for p in paths:
        with open(p) as fh:
            header = fh.readline()
        sep = "\t" if header.count("\t") > header.count(",") else ","
        columns = [c.strip() for c in header.rstrip("\r\n").split(sep)]
        use = [c for c in columns if c not in drop]
        LOG.info(f"reading {p} (separator {'tab' if sep == chr(9) else 'comma'}, "
                 f"{len(use)}/{len(columns)} columns)")
        kw = dict(sep=sep, nrows=remaining, usecols=lambda c: c.strip() in use)
        try:
            df = pd.read_csv(p, dtype={c: t for c, t in _read_dtypes(columns, dcfg).items()}, **kw)
        except (ValueError, TypeError) as err:
            LOG.warning(f"compact read failed ({err}); falling back to default dtypes (uses more RAM)")
            df = pd.read_csv(p, low_memory=False, **kw)
        frames.append(df)
        if remaining is not None:
            remaining -= len(df)
            if remaining <= 0:
                break
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    del frames
    df.columns = [c.strip() for c in df.columns]
    mem = df.memory_usage(deep=True).sum() / 1e9
    LOG.info(f"loaded {len(df):,} flows, {df.shape[1]} columns ({mem:.2f} GB in memory)")
    return df


def _require(df: pd.DataFrame, col: str, what: str) -> None:
    if col not in df.columns:
        close = difflib.get_close_matches(col, list(df.columns), n=3)
        raise KeyError(f"{what} column {col!r} not found. Close matches: {close}. "
                       f"Set it in the config (data section).")


def to_milliseconds(series: pd.Series, unit: str) -> np.ndarray:
    if unit == "ms":
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if unit == "s":
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64) * 1000.0
    if unit == "datetime":
        ts = pd.to_datetime(series, errors="coerce", dayfirst=True)
        out = ts.astype("int64").to_numpy(dtype=np.float64) / 1e6
        out[ts.isna().to_numpy()] = np.nan
        return out
    raise ValueError(f"unknown time_unit {unit!r}")


def clean(df: pd.DataFrame, dcfg: dict) -> pd.DataFrame:
    for key, what in [("time_col", "time"), ("src_ip_col", "source IP"),
                      ("dst_ip_col", "destination IP"), ("label_col", "label")]:
        _require(df, dcfg[key], what)
    n0 = len(df)
    for c in df.columns:  # inf -> NaN column by column (no full-table copy)
        if df[c].dtype.kind == "f":
            v = df[c].to_numpy()
            bad = ~np.isfinite(v) & ~np.isnan(v)
            if bad.any():
                LOG.info(f"column {c!r}: {bad.sum():,} infinite values ({bad.mean():.1%}) set to missing")
                df.loc[bad, c] = np.nan
    if dcfg.get("dedup", True):
        df = df.drop_duplicates(ignore_index=True)
        LOG.info(f"removed {n0 - len(df):,} exact duplicate rows")
    df["_t_ms"] = to_milliseconds(df[dcfg["time_col"]], dcfg["time_unit"])
    bad_t = int(df["_t_ms"].isna().sum())
    if bad_t:
        LOG.warning(f"dropping {bad_t:,} rows with an unreadable timestamp")
        df = df[df["_t_ms"].notna()]
    lab = dcfg["label_col"]
    df[lab] = df[lab].astype(str).str.strip().astype("category")
    df = df.sort_values("_t_ms", kind="stable", ignore_index=True)
    return df


# --------------------------------------------------------------------------- windows + split
def assign_windows(t_ms: np.ndarray, window_seconds: float, t0: float | None = None) -> np.ndarray:
    t0 = float(t_ms.min()) if t0 is None else t0
    return np.floor((t_ms - t0) / (window_seconds * 1000.0)).astype(np.int64)


def chronological_split(window_id: np.ndarray, fractions) -> np.ndarray:
    """Return split index per row (0 train, 1 val, 2 test), cutting only between windows.

    A window goes to the split in which the cumulative flow count *before* it falls.
    """
    f_train, f_val = fractions[0], fractions[0] + fractions[1]
    wins, counts = np.unique(window_id, return_counts=True)
    before = (np.cumsum(counts) - counts) / counts.sum()
    win_split = np.where(before < f_train, 0, np.where(before < f_val, 1, 2))
    lookup = dict(zip(wins.tolist(), win_split.tolist()))
    return pd.Series(window_id).map(lookup).to_numpy(dtype=np.int64)


# --------------------------------------------------------------------------- column groups
def column_groups(df: pd.DataFrame, dcfg: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric, tcp_flag, categorical) feature columns present in df."""
    exclude = {dcfg["time_col"], dcfg["src_ip_col"], dcfg["dst_ip_col"], dcfg["src_port_col"],
               dcfg["label_col"], "_t_ms", *dcfg.get("drop_cols", [])}
    tcp = [c for c in dcfg.get("tcp_flag_cols", []) if c in df.columns]
    cat = [c for c in dcfg.get("categorical_cols", []) if c in df.columns]
    missing = [c for c in dcfg.get("tcp_flag_cols", []) + dcfg.get("categorical_cols", [])
               if c not in df.columns]
    if missing:
        LOG.warning(f"configured categorical columns not in data (skipped): {missing}")
    numeric = []
    for c in df.columns:
        if c in exclude or c in tcp or c in cat:
            continue
        if df[c].dtype.kind in "fiu":  # numeric column (missing / infinite values become 0 later)
            missing = df[c].isna().mean()
            if missing > 0.05:
                LOG.info(f"column {c!r}: {missing:.1%} missing values, filled with 0")
            numeric.append(c)
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().mean() > 0.95:
            numeric.append(c)
        else:
            LOG.warning(f"column {c!r} is not numeric, dropped")
    return numeric, tcp, cat


# --------------------------------------------------------------------------- encoders
def _cat_str(series: pd.Series) -> pd.Series:
    """String form of a categorical column that does not depend on int/float storage."""
    if series.dtype.kind in "fiu":
        return series.astype(np.float64).astype(str)
    return series.astype(str)


class CategoricalEncoder:
    """TCP flags -> 8 bit columns; other categoricals -> one-hot of top-k train values + other."""

    def __init__(self, tcp_cols: list[str], cat_cols: list[str], top_k: int):
        self.tcp_cols, self.cat_cols, self.top_k = tcp_cols, cat_cols, top_k
        self.vocab: dict[str, list] = {}

    def fit(self, df_train: pd.DataFrame) -> "CategoricalEncoder":
        for c in self.cat_cols:
            vc = _cat_str(df_train[c]).value_counts()
            self.vocab[c] = vc.index[: self.top_k].tolist()
        return self

    @property
    def names(self) -> list[str]:
        out = [f"{c}_bit{b}" for c in self.tcp_cols for b in range(TCP_BITS)]
        for c in self.cat_cols:
            out += [f"{c}={v}" for v in self.vocab[c]] + [f"{c}=other"]
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        for c in self.tcp_cols:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int64).to_numpy()
            blocks.append(((v[:, None] >> np.arange(TCP_BITS)) & 1).astype(np.uint8))
        for c in self.cat_cols:
            s = _cat_str(df[c])
            vocab = self.vocab[c]
            idx = pd.Categorical(s, categories=vocab).codes  # -1 when not in vocab
            onehot = np.zeros((len(df), len(vocab) + 1), dtype=np.uint8)
            idx = np.where(idx < 0, len(vocab), idx)
            onehot[np.arange(len(df)), idx] = 1.0
            blocks.append(onehot)
        if not blocks:
            return np.zeros((len(df), 0), dtype=np.uint8)
        return np.concatenate(blocks, axis=1)

    def state(self) -> dict:
        return {"tcp_cols": self.tcp_cols, "cat_cols": self.cat_cols,
                "top_k": self.top_k, "vocab": self.vocab}

    @classmethod
    def from_state(cls, s: dict) -> "CategoricalEncoder":
        enc = cls(s["tcp_cols"], s["cat_cols"], s["top_k"])
        enc.vocab = s["vocab"]
        return enc


def numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = np.zeros((len(df), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        if c in df.columns:
            col = df[c] if df[c].dtype.kind == "f" else pd.to_numeric(df[c], errors="coerce")
            out[:, j] = col.fillna(0).to_numpy(dtype=np.float32)
    return out


def fit_transform_stats(num_train: np.ndarray, cat_train: np.ndarray, log_mask: np.ndarray) -> dict:
    """Mean/std of the transformed features (pass a random sample of training rows to save RAM)."""
    x = apply_transform_np(num_train, cat_train, log_mask, None, None)
    mu = x.mean(axis=0, dtype=np.float64)
    sd = x.std(axis=0, dtype=np.float64)
    sd[sd < 1e-6] = 1.0
    return {"log_mask": log_mask, "mu": mu.astype(np.float32), "sigma": sd.astype(np.float32)}


def apply_transform_np(num, cat, log_mask, mu, sigma) -> np.ndarray:
    num = num.astype(np.float32, copy=True)
    num[:, log_mask] = np.log1p(np.clip(num[:, log_mask], 0, None))
    x = np.concatenate([num, cat.astype(np.float32)], axis=1)
    if mu is not None:
        x = (x - mu) / sigma
    return x.astype(np.float32)


# --------------------------------------------------------------------------- labels
def build_label_space(labels_train: pd.Series, benign: str) -> list[str]:
    present = set(labels_train.astype(str).unique())
    classes = sorted(present - {benign})
    if benign not in present:
        raise ValueError(f"benign label {benign!r} not found in the training split")
    return [benign] + classes


def encode_labels(labels: pd.Series, classes: list[str], benign: str) -> tuple[np.ndarray, np.ndarray]:
    """Multiclass index (-1 for classes unseen in training) and binary (0 benign / 1 attack)."""
    lookup = {c: i for i, c in enumerate(classes)}
    lab = labels.astype(str)
    y = lab.map(lookup).fillna(-1).astype(np.int64).to_numpy()
    y_bin = (lab != benign).astype(np.int64).to_numpy()
    return y, y_bin


def report_class_counts(labels: pd.Series, split: np.ndarray, classes: list[str]) -> dict:
    split_names = pd.Series(split, index=labels.index, name="split").map(dict(enumerate(SPLIT_NAMES)))
    table = pd.crosstab(labels.astype(str).rename("class"), split_names)
    for s in SPLIT_NAMES:
        if s not in table.columns:
            table[s] = 0
    table = table[list(SPLIT_NAMES)]
    LOG.info("class counts per split:\n" + table.to_string())
    unseen = [c for c in table.index if c not in classes]
    if unseen:
        LOG.warning(f"classes absent from the training split (scored only in binary metrics): {unseen}")
    no_test = [c for c in classes if c not in table.index or table.loc[c, "test"] == 0]
    if no_test:
        LOG.warning(f"classes with no test flows (per-class F1 undefined): {no_test}")
    return {str(k): {s: int(v) for s, v in row.items()} for k, row in table.iterrows()}
