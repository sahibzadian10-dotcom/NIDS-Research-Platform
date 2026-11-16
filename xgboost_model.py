"""Gradient-boosted trees: the 'proposed' threshold-calibration + SHAP model.

Uses XGBoost when installed. If XGBoost is unavailable (as in the sandbox where the shipped results were
produced) it falls back to scikit-learn's HistGradientBoostingClassifier, which is the same model family
(histogram-based gradient-boosted decision trees).  The backend actually used is recorded in the results.
"""
from __future__ import annotations

BACKEND = "xgboost"
try:  # pragma: no cover - depends on environment
    import xgboost as xgb
except Exception:  # noqa: BLE001
    xgb = None
    BACKEND = "sklearn-HistGradientBoosting (XGBoost not installed)"
from sklearn.ensemble import HistGradientBoostingClassifier


def build(cfg: dict, seed: int):
    p = cfg["models"]["xgboost"]
    if xgb is not None:  # pragma: no cover
        return xgb.XGBClassifier(
            n_estimators=p["max_iter"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            objective="multi:softprob", tree_method="hist", n_jobs=2, random_state=seed, verbosity=0,
        )
    return HistGradientBoostingClassifier(
        max_iter=p["max_iter"], learning_rate=p["learning_rate"], max_depth=p["max_depth"], random_state=seed,
    )
