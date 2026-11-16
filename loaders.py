"""Load raw dataset CSVs in their native format and profile them."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# The four official UNSW-NB15 raw files (UNSW-NB15_1.csv ... _4.csv) ship WITHOUT a header row; this is the column
# order documented in NUSW-NB15_features.csv.
UNSW_RAW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth",
    "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label",
]
UNSW_RENAME = {"Spkts": "spkts", "Dpkts": "dpkts", "Sload": "sload"}

# Columns the aligners need. Checked up front so a wrong file variant fails with an explanation, not a KeyError.
REQUIRED_COLUMNS = {
    "UNSW-NB15": ["srcip", "dsport", "proto", "dur", "sbytes", "dbytes", "spkts", "dpkts", "Stime", "attack_cat"],
    "CIC-IDS2017": ["Source IP", "Destination Port", "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packets",
                    "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Label"],
}
FILE_HINT = {
    "UNSW-NB15": "Use the four raw files UNSW-NB15_1.csv ... UNSW-NB15_4.csv (they carry srcip, dsport and Stime). "
                 "The 'training-set/testing-set' partition files do not have these columns.",
    "CIC-IDS2017": "Use the 'TrafficLabelling' / GeneratedLabelledFlows CSVs (they carry Source IP, Protocol and Timestamp). "
                   "The 'MachineLearningCVE' CSVs do not have these columns.",
}


def _read_one(f: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(f, low_memory=False, encoding_errors="replace")
    df.columns = [str(c).strip() for c in df.columns]  # CIC-IDS2017 headers have leading spaces
    if name == "UNSW-NB15" and "dur" not in df.columns and df.shape[1] == len(UNSW_RAW_COLUMNS):
        df = pd.read_csv(f, low_memory=False, encoding_errors="replace", header=None, names=UNSW_RAW_COLUMNS)
    if name == "UNSW-NB15":
        df = df.rename(columns=UNSW_RENAME)
        if "attack_cat" in df.columns:  # blank for normal traffic in the official files
            df["attack_cat"] = df["attack_cat"].fillna("Normal")
    return df


def _read_all(folder: Path, name: str = "") -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}. Run `python -m src.data.synthetic` or place the real CSVs there.")
    df = pd.concat([_read_one(f, name) for f in files], ignore_index=True)
    missing = [c for c in REQUIRED_COLUMNS.get(name, []) if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: columns {missing} not found in {folder}. {FILE_HINT.get(name, '')}")
    return df


def load_dataset(name: str, raw_dir: Path) -> pd.DataFrame:
    folder = Path(raw_dir) / name
    df = _read_all(folder, name)
    df.attrs["dataset"] = name
    return df


def profile_raw(df: pd.DataFrame, name: str, label_col: str) -> dict:
    num = df.select_dtypes(include=[np.number])
    return {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_cells": int(df.isna().sum().sum()),
        "inf_cells": int(np.isinf(num).sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "raw_label_counts": {str(k).strip(): int(v) for k, v in df[label_col].astype(str).str.strip().value_counts().items()},
    }
