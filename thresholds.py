"""Score calibration and decision policies (G2).

Policies (all learned on SOURCE validation data only; the target dataset is never touched):
  default_argmax       raw model probabilities, arg-max class (the usual 0.5-style default)
  calibrated_argmax    isotonic-calibrated probabilities, arg-max class
  global_threshold     ONE shared threshold t for every attack class, minimising analyst cost on source-val
  per_class_cost       one threshold per attack class (coordinate descent), minimising the same cost
  per_class_fpr_cap    per-class threshold = lowest t whose benign false-alarm rate on source-val <= cap
                       (target-FPR style, cf. Ha & Kim 2026)
Decision rule: flow is attributed to the attack class with the highest probability among classes whose
probability passes their threshold; if none passes, it is Benign.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.config import CLASSES

K = len(CLASSES)


class IsotonicCalibrator:
    """One-vs-rest isotonic regression per class, then row-renormalisation."""

    def __init__(self):
        self.iso = {}

    def fit(self, P: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        for c in range(K):
            yy = (y == c).astype(float)
            if yy.sum() >= 5 and yy.sum() < len(yy):
                self.iso[c] = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(P[:, c], yy)
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        Q = np.array(P, dtype=float, copy=True)
        for c, m in self.iso.items():
            Q[:, c] = m.predict(P[:, c])
        s = Q.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return Q / s


def decide(P: np.ndarray, thresholds) -> np.ndarray:
    """thresholds: array of length K-1 (one per attack class)."""
    t = np.asarray(thresholds, dtype=float)
    att = P[:, 1:]
    passes = att >= t[None, :]
    masked = np.where(passes, att, -1.0)
    best = masked.argmax(axis=1) + 1
    return np.where(passes.any(axis=1), best, 0)


def argmax_decision(P: np.ndarray) -> np.ndarray:
    return P.argmax(axis=1)


def _cost(y, pred, ac) -> float:
    n = len(y)
    c = 0.0
    c += ac["cost_fp"] * np.sum((y == 0) & (pred != 0))
    for ci, name in enumerate(CLASSES):
        if ci == 0:
            continue
        c += ac["cost_fn"][name] * np.sum((y == ci) & (pred == 0))
        c += ac["cost_confusion"] * np.sum((y == ci) & (pred != 0) & (pred != ci))
    return c / n


def _grid(ac) -> np.ndarray:
    return np.round(np.arange(ac["grid_min"], ac["grid_max"] + 1e-9, ac["grid_step"]), 4)


def optimize_global(P, y, ac) -> np.ndarray:
    best_t, best_c = 0.5, np.inf
    for t in _grid(ac):
        c = _cost(y, decide(P, [t] * (K - 1)), ac)
        if c < best_c - 1e-12 or (abs(c - best_c) <= 1e-12 and t > best_t):
            best_t, best_c = t, c
    return np.full(K - 1, best_t)


def optimize_per_class_cost(P, y, ac, passes: int = 3) -> np.ndarray:
    t = optimize_global(P, y, ac).copy()
    grid = _grid(ac)
    cur = _cost(y, decide(P, t), ac)
    for _ in range(passes):
        improved = False
        for k in range(K - 1):
            for g in grid:
                cand = t.copy()
                cand[k] = g
                c = _cost(y, decide(P, cand), ac)
                if c < cur - 1e-12:
                    t, cur, improved = cand, c, True
        if not improved:
            break
    return t


def optimize_fpr_cap(P, y, ac) -> np.ndarray:
    benign = P[y == 0]
    cap = ac["fpr_cap_per_class"]
    t = np.zeros(K - 1)
    grid = _grid(ac)
    for k in range(K - 1):
        s = benign[:, k + 1]
        chosen = grid[-1]
        for g in grid:
            if np.mean(s >= g) <= cap:
                chosen = g
                break
        t[k] = chosen
    return t


def learn_policies(P_val_cal: np.ndarray, y_val: np.ndarray, ac: dict) -> dict:
    return {
        "global_threshold": optimize_global(P_val_cal, y_val, ac),
        "per_class_cost": optimize_per_class_cost(P_val_cal, y_val, ac),
        "per_class_fpr_cap": optimize_fpr_cap(P_val_cal, y_val, ac),
    }


POLICY_LABELS = {
    "default_argmax": "Default (raw arg-max)",
    "calibrated_argmax": "Calibrated arg-max",
    "global_threshold": "Global threshold",
    "per_class_cost": "Per-class thresholds (cost)",
    "per_class_fpr_cap": "Per-class thresholds (FPR cap)",
}
POLICY_ORDER = list(POLICY_LABELS)


def apply_policy(name: str, P_raw: np.ndarray, P_cal: np.ndarray, thr: dict) -> np.ndarray:
    if name == "default_argmax":
        return argmax_decision(P_raw)
    if name == "calibrated_argmax":
        return argmax_decision(P_cal)
    return decide(P_cal, thr[name])
