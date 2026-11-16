"""Source-fitted preprocessing. Fit ONLY on source-train; applied unchanged to the target (zero-label transfer)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LOG_FEATURES, SHARED_FEATURES


class Preprocessor:
    def __init__(self):
        self.log_idx = [SHARED_FEATURES.index(f) for f in LOG_FEATURES]
        self.mean_ = None
        self.std_ = None

    def _tx(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        X[:, self.log_idx] = np.log1p(np.maximum(X[:, self.log_idx], 0))
        return X

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        Z = self._tx(df[SHARED_FEATURES].to_numpy())
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0) + 1e-9
        return self

    def transform(self, df_or_arr) -> np.ndarray:
        X = df_or_arr[SHARED_FEATURES].to_numpy() if isinstance(df_or_arr, pd.DataFrame) else np.asarray(df_or_arr)
        return (self._tx(X) - self.mean_) / self.std_
