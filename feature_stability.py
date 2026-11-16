"""Cross-dataset stability of explanations (G3): top-k overlap, rank correlation, link to errors."""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, spearmanr


def importance_share(phi: np.ndarray) -> np.ndarray:
    imp = np.abs(phi).mean(axis=0)
    return imp / max(imp.sum(), 1e-12)


def ranks_desc(v: np.ndarray) -> np.ndarray:
    order = np.argsort(-v, kind="stable")
    r = np.empty(len(v), dtype=int)
    r[order] = np.arange(1, len(v) + 1)
    return r


def stability(imp_a: np.ndarray, imp_b: np.ndarray, k: int) -> dict:
    ta, tb = set(np.argsort(-imp_a)[:k]), set(np.argsort(-imp_b)[:k])
    rho = spearmanr(imp_a, imp_b).statistic
    return {
        "top_k": k,
        "top_k_overlap": len(ta & tb),
        "top_k_overlap_frac": len(ta & tb) / k,
        "jaccard": len(ta & tb) / len(ta | tb),
        "spearman_rho": float(rho) if rho == rho else 0.0,
    }


def error_enrichment(phi_err: np.ndarray, phi_ok: np.ndarray) -> np.ndarray:
    """Per-feature ratio of attribution share among errors vs among correct predictions (>1 = more active on errors)."""
    return importance_share(phi_err) / np.maximum(importance_share(phi_ok), 1e-9)


def safe_spearman(a, b) -> float:
    r = spearmanr(a, b).statistic
    return float(r) if r == r else 0.0


def ks_per_feature(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.array([ks_2samp(A[:, j], B[:, j]).statistic for j in range(A.shape[1])])
