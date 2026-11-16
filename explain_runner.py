"""Explanation-stability experiment for the proposed model (G3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLASSES, SHARED_FEATURES
from src.evaluation.thresholds import apply_policy
from src.explainability.feature_stability import (error_enrichment, importance_share, ks_per_feature, ranks_desc, safe_spearman, stability)
from src.explainability.shap_analysis import completeness_error, sampling_shapley

F = SHARED_FEATURES


def _sample(df: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=int(rng.integers(0, 2**31 - 1)))


def explain_direction(src, tgt, sp_src, sp_tgt, artifacts, cfg, seed, local_examples: bool = False) -> dict:
    sc = cfg["shap"]
    rng = np.random.default_rng(1000 + seed)
    pre, e = artifacts["pre"], artifacts["fitted"]["xgboost"]
    model = e["model"]
    f_attack = lambda A: 1.0 - model.predict_proba(A)[:, 0]  # noqa: E731

    s_test, t_test = src.iloc[sp_src["test"]], tgt.iloc[sp_tgt["test"]]
    s_samp, t_samp = _sample(s_test, sc["n_explain"], rng), _sample(t_test, sc["n_explain"], rng)
    Xs, Xt = pre.transform(s_samp), pre.transform(t_samp)
    bg_s = pre.transform(_sample(src.iloc[sp_src["train"]], sc["n_background"], rng))
    bg_t = pre.transform(_sample(tgt.iloc[sp_tgt["train"]], sc["n_background"], rng))  # unlabelled target features only

    phi_s, base_s = sampling_shapley(f_attack, Xs, bg_s, sc["n_perm"], rng)
    phi_t, base_t = sampling_shapley(f_attack, Xt, bg_t, sc["n_perm"], rng)
    imp_s, imp_t = importance_share(phi_s), importance_share(phi_t)
    stab = stability(imp_s, imp_t, sc["top_k"])
    stab3 = stability(imp_s, imp_t, 3)
    rank_s, rank_t = ranks_desc(imp_s), ranks_desc(imp_t)
    rank_shift = np.abs(rank_s - rank_t)

    # error probe on the TARGET: attribution share on misclassified vs correct flows (default arg-max decisions)
    Xt_full = pre.transform(t_test)
    pred = model.predict_proba(Xt_full).argmax(axis=1)
    y_t = t_test["label"].to_numpy()
    err_idx, ok_idx = np.where(pred != y_t)[0], np.where(pred == y_t)[0]
    n_probe = min(sc["n_error_probe"], len(err_idx))
    enrich = np.ones(len(F))
    if n_probe >= 20:
        ei = rng.choice(err_idx, n_probe, replace=False)
        oi = rng.choice(ok_idx, n_probe, replace=False)
        phi_e, _ = sampling_shapley(f_attack, Xt_full[ei], bg_t, max(10, sc["n_perm"] // 2), rng)
        phi_o, _ = sampling_shapley(f_attack, Xt_full[oi], bg_t, max(10, sc["n_perm"] // 2), rng)
        enrich = error_enrichment(phi_e, phi_o)

    # distribution shift per feature (benign-only to separate shift from class mix), raw feature space
    ks = ks_per_feature(s_test[s_test["label"] == 0][F].to_numpy(), t_test[t_test["label"] == 0][F].to_numpy())

    res = {
        "importance_source": {f: float(v) for f, v in zip(F, imp_s)},
        "importance_target": {f: float(v) for f, v in zip(F, imp_t)},
        "rank_source": {f: int(v) for f, v in zip(F, rank_s)},
        "rank_target": {f: int(v) for f, v in zip(F, rank_t)},
        "rank_shift": {f: int(v) for f, v in zip(F, rank_shift)},
        "error_enrichment": {f: float(v) for f, v in zip(F, enrich)},
        "ks_benign": {f: float(v) for f, v in zip(F, ks)},
        "stability_topk": stab,
        "stability_top3": stab3,
        "rho_rankshift_vs_ks": safe_spearman(rank_shift, ks),
        "rho_rankshift_vs_enrichment": safe_spearman(rank_shift, enrich),
        "n_error_probe": int(n_probe),
        "completeness_error_source": completeness_error(f_attack, Xs, phi_s, base_s),
        "completeness_error_target": completeness_error(f_attack, Xt, phi_t, base_t),
        "mean_attack_score_scale": float(np.mean(np.abs(phi_s.sum(axis=1)))),
    }
    if local_examples:
        res["local"] = _local(t_test, Xt_full, y_t, artifacts, bg_t, rng, model, e, pre)
    return res


def _local(t_test, Xt_full, y_t, artifacts, bg_t, rng, model, e, pre) -> list:
    """Per-alert explanations for a handful of target flows: true alerts, false alerts, missed attacks."""
    P_raw = model.predict_proba(Xt_full)
    P_cal = e["cal"].transform(P_raw)
    pred = apply_policy("per_class_cost", P_raw, P_cal, e["thr"])
    tp = np.where((y_t != 0) & (pred == y_t))[0]
    fp = np.where((y_t == 0) & (pred != 0))[0]
    fn = np.where((y_t != 0) & (pred == 0))[0]
    pick = []
    for kind, pool, k in (("True alert", tp, 8), ("False alert", fp, 6), ("Missed attack", fn, 6)):
        if len(pool):
            pick += [(kind, int(i)) for i in rng.choice(pool, min(k, len(pool)), replace=False)]
    out = []
    for kind, i in pick:
        cls = int(pred[i]) if pred[i] != 0 else int(y_t[i])  # explain the predicted attack class, or the missed one
        f_c = lambda A, c=cls: model.predict_proba(A)[:, c]  # noqa: E731
        phi, base = sampling_shapley(f_c, Xt_full[i:i + 1], bg_t, 300, rng)
        raw_row = t_test.iloc[i][F]
        out.append({
            "kind": kind,
            "true": CLASSES[int(y_t[i])],
            "predicted": CLASSES[int(pred[i])],
            "explained_class": CLASSES[cls],
            "prob_explained_class": float(P_raw[i, cls]),
            "base_value": base,
            "features": {f: float(raw_row[f]) for f in F},
            "phi": {f: float(v) for f, v in zip(F, phi[0])},
        })
    return out
