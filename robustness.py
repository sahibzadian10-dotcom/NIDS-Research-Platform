"""Aggregate numeric results over random seeds (mean +/- std)."""
from __future__ import annotations

import numpy as np


def aggregate(dicts: list):
    """Recursively aggregate a list of identically-shaped nested dicts of numbers.

    Numbers -> {"mean","std","n"}; matrices (lists) -> element-wise mean; strings -> first value.
    """
    first = dicts[0]
    if isinstance(first, dict):
        return {k: aggregate([d[k] for d in dicts]) for k in first}
    if isinstance(first, list) and first and isinstance(first[0], dict):
        return first  # list of records (e.g. reliability bins): keep the first run
    if isinstance(first, list):
        return np.mean(np.array(dicts, dtype=float), axis=0).tolist()
    if isinstance(first, (int, float, np.integer, np.floating)) and not isinstance(first, bool):
        arr = np.array(dicts, dtype=float)
        return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr, ddof=0)), "n": int(len(arr))}
    return first
