"""Shapley-value explanations (G3).

`sampling_shapley` is a model-agnostic permutation-sampling Shapley estimator (Strumbelj & Kononenko, 2014),
the same quantity SHAP approximates.  It is used here because the `shap` package could not be installed in
the build sandbox.  When `shap` IS installed and the model is XGBoost, `tree_shap_values` gives exact TreeSHAP.

Every permutation's contributions telescope exactly to f(x) - f(z) for its background draw z, so the
estimator satisfies local accuracy (completeness) by construction; `completeness_error` reports how far the
mean attribution sum is from f(x) - E[f(background)] (pure background-sampling noise).
"""
from __future__ import annotations

import numpy as np


def sampling_shapley(f, X: np.ndarray, background: np.ndarray, n_perm: int, rng: np.random.Generator, chunk: int = 200):
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    phi = np.zeros((N, d))
    for s in range(0, N, chunk):
        Xs = X[s:s + chunk]
        n = len(Xs)
        ar = np.arange(n)
        acc = np.zeros((n, d))
        for _ in range(n_perm):
            order = np.argsort(rng.random((n, d)), axis=1)
            cur = background[rng.integers(0, len(background), n)].copy()
            rows = np.empty((n, d + 1, d))
            rows[:, 0, :] = cur
            for t in range(d):
                j = order[:, t]
                cur[ar, j] = Xs[ar, j]
                rows[:, t + 1, :] = cur
            vals = f(rows.reshape(-1, d)).reshape(n, d + 1)
            delta = vals[:, 1:] - vals[:, :-1]
            acc[ar[:, None], order] += delta
        phi[s:s + n] = acc / n_perm
    base = float(np.mean(f(background)))
    return phi, base


def completeness_error(f, X, phi, base) -> float:
    return float(np.mean(np.abs(phi.sum(axis=1) - (f(X) - base))))


def tree_shap_values(model, X):  # pragma: no cover - needs `shap` + xgboost
    import shap

    return shap.TreeExplainer(model).shap_values(X)
