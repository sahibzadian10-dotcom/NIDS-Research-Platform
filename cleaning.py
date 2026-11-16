"""Cleaning with a step-by-step audit trail (every dropped row is counted)."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_NUMERIC = {
    "UNSW-NB15": ["dur", "sbytes", "dbytes", "spkts", "dpkts"],
    "CIC-IDS2017": ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets"],
}


def clean_raw(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, list[dict]]:
    """Return (clean_df, audit). Order: duplicates -> inf/NaN -> invalid values."""
    audit = [{"step": "raw rows", "removed": 0, "remaining": len(df)}]
    out = df.copy()

    n = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    audit.append({"step": "exact duplicate rows", "removed": n - len(out), "remaining": len(out)})

    n = len(out)
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    req = REQUIRED_NUMERIC[name]
    for c in req:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if name == "UNSW-NB15":
        out["dsport"] = pd.to_numeric(out["dsport"], errors="coerce")  # hex ports -> NaN
        out = out.dropna(subset=req + ["dsport"])
    else:
        out = out.dropna(subset=req)
    audit.append({"step": "NaN / inf / unparsable values", "removed": n - len(out), "remaining": len(out)})

    n = len(out)
    out = out[(out[req] >= 0).all(axis=1)].reset_index(drop=True)
    audit.append({"step": "negative durations / sizes", "removed": n - len(out), "remaining": len(out)})
    return out, audit
