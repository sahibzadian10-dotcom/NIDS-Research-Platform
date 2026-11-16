"""Shared-feature alignment. We never assume similarly named columns mean the same thing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SHARED_FEATURES

HEADER_BYTES_PER_PKT = 40  # assumed minimum IP+TCP header; CIC reports payload-only lengths (approximation)
PROTO_NUM = {"tcp": 6, "udp": 17, "icmp": 1}


def port_class(port: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(port, dtype=float)
    return np.where(p < 1024, 0, np.where(p < 49152, 1, 2))


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features computed identically for datasets and for the replay flow exporter."""
    dur_ms = df["FLOW_DURATION_MS"].to_numpy(dtype=float)
    tot_b = (df["IN_BYTES"] + df["OUT_BYTES"]).to_numpy(dtype=float)
    tot_p = (df["IN_PKTS"] + df["OUT_PKTS"]).to_numpy(dtype=float)
    denom_s = np.maximum(dur_ms, 1.0) / 1000.0  # 1 ms floor avoids inf rates for single-packet flows
    df["MEAN_PKT_SIZE"] = tot_b / np.maximum(tot_p, 1)
    df["MEAN_IN_PKT_SIZE"] = df["IN_BYTES"] / np.maximum(df["IN_PKTS"], 1)
    df["MEAN_OUT_PKT_SIZE"] = df["OUT_BYTES"] / np.maximum(df["OUT_PKTS"], 1)
    df["BYTES_PER_SEC"] = tot_b / denom_s
    df["PKTS_PER_SEC"] = tot_p / denom_s
    df["IN_OUT_PKT_RATIO"] = df["IN_PKTS"] / (df["OUT_PKTS"] + 1.0)
    return df


def _session_id(srcip: pd.Series, epoch_s: pd.Series) -> pd.Series:
    return srcip.astype(str) + "@" + (epoch_s // 600).astype("int64").astype(str)  # 10-minute bucket


def align_unsw(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["PROTOCOL"] = df["proto"].astype(str).str.strip().str.lower().map(PROTO_NUM).fillna(0).astype(int)
    out["DST_PORT_CLASS"] = port_class(df["dsport"])
    out["FLOW_DURATION_MS"] = df["dur"].astype(float) * 1000.0  # seconds -> ms
    out["IN_BYTES"] = df["sbytes"].astype(float)
    out["OUT_BYTES"] = df["dbytes"].astype(float)
    out["IN_PKTS"] = df["spkts"].astype(float)
    out["OUT_PKTS"] = df["dpkts"].astype(float)
    out = add_derived(out)
    out["session_id"] = _session_id(df["srcip"], df["Stime"].astype("int64"))
    return out


def align_cic(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["PROTOCOL"] = pd.to_numeric(df["Protocol"], errors="coerce").fillna(0).astype(int)
    out["DST_PORT_CLASS"] = port_class(df["Destination Port"])
    out["FLOW_DURATION_MS"] = df["Flow Duration"].astype(float) / 1000.0  # microseconds -> ms
    out["IN_PKTS"] = df["Total Fwd Packets"].astype(float)
    out["OUT_PKTS"] = df["Total Backward Packets"].astype(float)
    # CIC lengths are payload only: add assumed headers so the byte counts are comparable with UNSW.
    out["IN_BYTES"] = df["Total Length of Fwd Packets"].astype(float) + HEADER_BYTES_PER_PKT * out["IN_PKTS"]
    out["OUT_BYTES"] = df["Total Length of Bwd Packets"].astype(float) + HEADER_BYTES_PER_PKT * out["OUT_PKTS"]
    out = add_derived(out)
    ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M", errors="coerce")
    epoch = (ts.astype("int64") // 10**9).where(ts.notna(), 0) if ts.notna().any() else pd.Series(0, index=df.index)
    out["session_id"] = _session_id(df["Source IP"], epoch)
    return out[["PROTOCOL", "DST_PORT_CLASS", "FLOW_DURATION_MS", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
                "MEAN_PKT_SIZE", "MEAN_IN_PKT_SIZE", "MEAN_OUT_PKT_SIZE", "BYTES_PER_SEC", "PKTS_PER_SEC",
                "IN_OUT_PKT_RATIO", "session_id"]]


ALIGNERS = {"UNSW-NB15": align_unsw, "CIC-IDS2017": align_cic}


# Documented schema decisions (shown in the dashboard's Dataset Intelligence tab).
SCHEMA_DECISIONS = [
    {"feature": "PROTOCOL", "unsw": "proto (string)", "cic": "Protocol (IANA number)", "transform": "tcp/udp/icmp -> 6/17/1", "status": "shared", "note": "Exact semantic match."},
    {"feature": "DST_PORT_CLASS", "unsw": "dsport", "cic": "Destination Port", "transform": "<1024 / <49152 / else -> 0/1/2", "status": "shared", "note": "Bucketed to avoid memorising dataset-specific port numbers. UNSW has hex-valued ports (dropped)."},
    {"feature": "FLOW_DURATION_MS", "unsw": "dur (seconds)", "cic": "Flow Duration (microseconds)", "transform": "x1000 / x0.001", "status": "shared", "note": "Unit conversion; flow-termination rules differ between exporters (Argus vs CICFlowMeter)."},
    {"feature": "IN_BYTES / OUT_BYTES", "unsw": "sbytes / dbytes", "cic": "Total Length of Fwd/Bwd Packets", "transform": "CIC + 40 B x packets", "status": "approximate", "note": "CIC reports payload only; header bytes are re-added with an assumed 40 B/packet. Residual mismatch remains."},
    {"feature": "IN_PKTS / OUT_PKTS", "unsw": "spkts / dpkts", "cic": "Total Fwd / Backward Packets", "transform": "none", "status": "shared", "note": "Directional counts. Direction definitions follow each exporter."},
    {"feature": "MEAN_PKT_SIZE (+IN/OUT)", "unsw": "derived", "cic": "derived", "transform": "bytes / packets", "status": "derived", "note": "Computed from aligned fields, NOT from native smean/dmean or Average Packet Size."},
    {"feature": "BYTES_PER_SEC / PKTS_PER_SEC", "unsw": "derived", "cic": "derived", "transform": "total / max(dur, 1 ms)", "status": "derived", "note": "Native CIC Flow Bytes/s contains inf/NaN, so it is recomputed identically for both datasets."},
    {"feature": "IN_OUT_PKT_RATIO", "unsw": "derived", "cic": "derived", "transform": "in / (out + 1)", "status": "derived", "note": "Direction asymmetry signal."},
    {"feature": "sttl / dttl", "unsw": "sttl", "cic": "(none)", "transform": "-", "status": "excluded", "note": "No CIC equivalent; also a known shortcut feature in UNSW (attack traffic generator leaves TTL fingerprints)."},
    {"feature": "state", "unsw": "state (FIN/CON/INT/...)", "cic": "Flag counts (SYN, ACK, ...)", "transform": "-", "status": "excluded", "note": "Argus connection states and CIC TCP flag counts are not equivalent."},
    {"feature": "sinpkt vs Flow IAT Mean", "unsw": "sinpkt", "cic": "Flow IAT Mean", "transform": "-", "status": "excluded", "note": "Similar-sounding names, different definitions (source-only ms vs bidirectional us). Not assumed equal."},
    {"feature": "Fwd Header Length", "unsw": "(none)", "cic": "Fwd Header Length", "transform": "-", "status": "excluded", "note": "CICFlowMeter-specific; no UNSW counterpart."},
    {"feature": "srcip + timestamp", "unsw": "srcip, Stime", "cic": "Source IP, Timestamp", "transform": "host + 10-min bucket -> session_id", "status": "split-only", "note": "Used ONLY to build leakage-aware groups for splitting; never given to a model."},
]


def finalize(df_aligned: pd.DataFrame, labels: pd.Series, dataset: str) -> pd.DataFrame:
    out = df_aligned.copy()
    out["label"] = labels.to_numpy()
    out["dataset"] = dataset
    return out[SHARED_FEATURES + ["session_id", "label", "dataset"]]
