"""Detection, alert-quality and calibration metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support

from src.config import ATTACK_CLASSES, CLASSES

K = len(CLASSES)


def prf(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(K)), zero_division=0)
    return {
        "macro_f1": float(np.mean(f)),
        "precision": {c: float(p[i]) for i, c in enumerate(CLASSES)},
        "recall": {c: float(r[i]) for i, c in enumerate(CLASSES)},
        "f1": {c: float(f[i]) for i, c in enumerate(CLASSES)},
        "support": {c: int(s[i]) for i, c in enumerate(CLASSES)},
        "macro_attack_recall": float(np.mean(r[1:])),
    }


def alert_metrics(y_true, y_pred) -> dict:
    """Binary view: an 'alert' is any non-benign prediction."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    benign = y_true == 0
    attack = ~benign
    alerts = y_pred != 0
    fp = int(np.sum(alerts & benign))
    n_benign = int(benign.sum())
    return {
        "fpr": fp / max(n_benign, 1),
        "alerts_per_10k_benign": 1e4 * fp / max(n_benign, 1),
        "false_alerts": fp,
        "n_benign": n_benign,
        "total_alerts": int(alerts.sum()),
        "alerts_per_10k_flows": 1e4 * alerts.sum() / max(len(y_true), 1),
        "detection_rate": float(np.sum(alerts & attack) / max(attack.sum(), 1)),
        "alert_precision": float(np.sum(alerts & attack) / max(alerts.sum(), 1)),
    }


def expected_cost(y_true, y_pred, ac: dict) -> float:
    """Mean per-flow cost under the analyst cost matrix (FP / missed attack per class / attack-class confusion)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cost = np.zeros(len(y_true))
    fp = (y_true == 0) & (y_pred != 0)
    cost[fp] = ac["cost_fp"]
    for ci, c in enumerate(CLASSES):
        if ci == 0:
            continue
        miss = (y_true == ci) & (y_pred == 0)
        cost[miss] = ac["cost_fn"][c]
        conf = (y_true == ci) & (y_pred != 0) & (y_pred != ci)
        cost[conf] = ac["cost_confusion"]
    return float(cost.mean())


def pr_auc(y_true, P) -> dict:
    y_true = np.asarray(y_true)
    per = {}
    for ci, c in enumerate(CLASSES):
        yy = (y_true == ci).astype(int)
        per[c] = float(average_precision_score(yy, P[:, ci])) if yy.sum() > 0 else float("nan")
    att = (y_true != 0).astype(int)
    return {
        "per_class": per,
        "macro": float(np.nanmean(list(per.values()))),
        "attack_vs_benign": float(average_precision_score(att, 1.0 - P[:, 0])) if att.sum() > 0 else float("nan"),
    }


def ece(y_true, P, n_bins: int = 15) -> float:
    """Top-label expected calibration error."""
    y_true = np.asarray(y_true)
    conf, pred = P.max(axis=1), P.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def brier(y_true, P) -> float:
    Y = np.eye(K)[np.asarray(y_true)]
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def reliability_curve(y_true, P, n_bins: int = 10) -> list[dict]:
    """Attack-vs-benign reliability: predicted attack probability vs observed attack frequency."""
    p = 1.0 - P[:, 0]
    a = (np.asarray(y_true) != 0).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() > 0:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "pred": float(p[m].mean()), "obs": float(a[m].mean()), "n": int(m.sum())})
    return rows


def score_stats(y_true, P) -> dict:
    """Score-quality metrics that depend only on the probability matrix (computed once per score type)."""
    pa = pr_auc(y_true, P)
    return {
        "pr_auc_macro": pa["macro"],
        "pr_auc_attack": pa["attack_vs_benign"],
        "pr_auc_per_class": pa["per_class"],
        "ece": ece(y_true, P),
        "brier": brier(y_true, P),
    }


def full_report(y_true, y_pred, ac: dict, scores: dict | None = None) -> dict:
    rep = prf(y_true, y_pred)
    rep.update(alert_metrics(y_true, y_pred))
    rep["cost_per_flow"] = expected_cost(y_true, y_pred, ac)
    rep["confusion"] = confusion_matrix(y_true, y_pred, labels=list(range(K))).tolist()
    if scores:
        rep.update(scores)
    return rep
