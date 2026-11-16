"""Minimal packet -> bidirectional-flow exporter (a stand-in for CICFlowMeter in the replay test).

It aggregates packet records (ts, src, dst, sport, dport, proto, size) by 5-tuple, treats the first packet
of a flow as the forward ('IN') direction, and emits exactly the shared features used by the models.
It is NOT CICFlowMeter: no TCP-state termination, no flag features.  Replay results therefore measure this
pipeline's cost, not CICFlowMeter's.
"""
from __future__ import annotations

import numpy as np

from src.config import SHARED_FEATURES


class FlowExporter:
    __slots__ = ("flows", "idle_timeout", "n_packets")

    def __init__(self, idle_timeout: float = 120.0):
        self.flows: dict = {}
        self.idle_timeout = idle_timeout
        self.n_packets = 0

    def ingest(self, ts: float, src, dst, sport: int, dport: int, proto: int, size: int) -> None:
        self.n_packets += 1
        flows = self.flows
        key = (src, sport, dst, dport, proto)
        st = flows.get(key)
        if st is not None:
            st[1] = ts
            st[2] += 1
            st[4] += size
            return
        rkey = (dst, dport, src, sport, proto)
        st = flows.get(rkey)
        if st is not None:
            st[1] = ts
            st[3] += 1
            st[5] += size
            return
        # [first_ts, last_ts, in_pkts, out_pkts, in_bytes, out_bytes, dport, proto]
        flows[key] = [ts, ts, 1, 0, size, 0, dport, proto]

    def finalize_all(self) -> list:
        out = list(self.flows.values())
        self.flows = {}
        return out


def state_to_features(st) -> np.ndarray:
    """Flow state -> 13-feature row in SHARED_FEATURES order (pure python/numpy for low per-flow latency)."""
    dur_ms = (st[1] - st[0]) * 1000.0
    in_p, out_p, in_b, out_b = float(st[2]), float(st[3]), float(st[4]), float(st[5])
    dport = st[6]
    pc = 0 if dport < 1024 else (1 if dport < 49152 else 2)
    tot_b, tot_p = in_b + out_b, in_p + out_p
    denom_s = max(dur_ms, 1.0) / 1000.0
    return np.array([
        st[7], pc, dur_ms, in_b, out_b, in_p, out_p,
        tot_b / max(tot_p, 1.0), in_b / max(in_p, 1.0), out_b / max(out_p, 1.0),
        tot_b / denom_s, tot_p / denom_s, in_p / (out_p + 1.0),
    ], dtype=float)


assert len(SHARED_FEATURES) == 13
