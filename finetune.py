"""STRETCH experiment (G5): small-budget target fine-tuning with 1-5% labelled target flows.

Kept strictly separate from the zero-label results: labelled target flows come only from the target TRAIN
partition; evaluation is on the target TEST partition (same as the zero-label experiment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CLASSES
from src.evaluation.metrics import alert_metrics, prf
from src.models import MODEL_BUILDERS


def run_finetune(src, tgt, sp_src, sp_tgt, pre, cfg, seed) -> dict:
    fr = cfg["stretch"]["fractions"]
    tr_t = tgt.iloc[sp_tgt["train"]]
    te_t = tgt.iloc[sp_tgt["test"]]
    Xs, ys = pre.transform(src.iloc[sp_src["train"]]), src.iloc[sp_src["train"]]["label"].to_numpy()
    Xte, yte = pre.transform(te_t), te_t["label"].to_numpy()
    out = {}
    for f in [0.0] + list(fr):
        if f == 0.0:
            X, y = Xs, ys
        else:
            n = max(int(round(f * len(tr_t))), len(CLASSES) * 2)
            _, samp = train_test_split(np.arange(len(tr_t)), test_size=n, stratify=tr_t["label"].to_numpy(), random_state=seed)
            X = np.vstack([Xs, pre.transform(tr_t.iloc[samp])])
            y = np.concatenate([ys, tr_t.iloc[samp]["label"].to_numpy()])
        m = MODEL_BUILDERS["xgboost"](cfg, seed)
        m.fit(X, y)
        pred = m.predict_proba(Xte).argmax(axis=1)
        r = prf(yte, pred)
        a = alert_metrics(yte, pred)
        out[f"{f:.2f}"] = {
            "fraction": f,
            "labelled_target_flows": 0 if f == 0.0 else int(len(samp)),
            "macro_f1": r["macro_f1"], "recall": r["recall"],
            "fpr": a["fpr"], "alerts_per_10k_benign": a["alerts_per_10k_benign"], "detection_rate": a["detection_rate"],
        }
    return out
