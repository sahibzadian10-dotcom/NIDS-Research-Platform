"""Leakage-aware, stratified, seeded train/validation/test splits.

Flows from the same host session (source IP + 10-minute bucket) never straddle two splits,
and exact feature duplicates are removed before splitting, so no test flow has a twin in training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import SHARED_FEATURES


def dedupe_shared(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n = len(df)
    out = df.drop_duplicates(subset=SHARED_FEATURES + ["label"]).reset_index(drop=True)
    return out, n - len(out)


def leakage_aware_split(df: pd.DataFrame, seed: int, test_folds: int = 4, val_folds: int = 5) -> dict:
    y = df["label"].to_numpy()
    g = df["session_id"].to_numpy()
    idx = np.arange(len(df))
    sgk = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    trainval, test = next(sgk.split(idx, y, g))
    sgk2 = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed)
    tr_rel, va_rel = next(sgk2.split(trainval, y[trainval], g[trainval]))
    train, val = trainval[tr_rel], trainval[va_rel]
    return {"train": train, "val": val, "test": test}


def leakage_report(df: pd.DataFrame, split: dict) -> dict:
    g = df["session_id"].to_numpy()
    sets = {k: set(g[v]) for k, v in split.items()}
    feats = df[SHARED_FEATURES].round(6).astype(str).agg("|".join, axis=1)
    key = feats + "#" + df["label"].astype(str)
    ks = {k: set(key.iloc[v]) for k, v in split.items()}
    return {
        "session_overlap_train_test": len(sets["train"] & sets["test"]),
        "session_overlap_train_val": len(sets["train"] & sets["val"]),
        "session_overlap_val_test": len(sets["val"] & sets["test"]),
        "duplicate_overlap_train_test": len(ks["train"] & ks["test"]),
        "duplicate_overlap_train_val": len(ks["train"] & ks["val"]),
        "sizes": {k: int(len(v)) for k, v in split.items()},
    }
