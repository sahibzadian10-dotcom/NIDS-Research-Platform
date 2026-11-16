"""Deployment-style replay (G4): packets -> flow exporter -> model -> alerts, with measured cost.

The packets are SYNTHESISED from held-out target-dataset flows (no real PCAP is available here), replayed
through the exporter in timestamp order.  What is measured is real wall-clock time on the machine running
the code; what is NOT demonstrated is CICFlowMeter behaviour or live-network accuracy.
"""
from __future__ import annotations

import pickle
import platform
import time

import numpy as np
import pandas as pd

from src.config import CLASSES, SHARED_FEATURES
from src.evaluation.metrics import alert_metrics, prf
from src.replay.flow_exporter import FlowExporter, state_to_features

WK = [80, 443, 53, 22, 21, 25, 445, 110]  # all < 1024


def synthesize_packets(flows: pd.DataFrame, rng: np.random.Generator, window_s: float):
    """Expand each flow row into packets whose re-aggregation reproduces the flow's features."""
    pk = []
    starts = rng.uniform(0, window_s, len(flows))
    for i, (row, t0) in enumerate(zip(flows[SHARED_FEATURES].to_numpy(), starts)):
        proto, pc, dur_ms, in_b, out_b, in_p, out_p = int(row[0]), int(row[1]), row[2], int(round(row[3])), int(round(row[4])), int(row[5]), int(row[6])
        dport = int(rng.choice(WK)) if pc == 0 else int(rng.integers(1024, 49152)) if pc == 1 else int(rng.integers(49152, 65535))
        sport = 40000 + (i % 20000)
        src, dst = i, -1
        n = in_p + out_p
        dur = dur_ms / 1000.0
        ts = np.sort(rng.uniform(0, dur, n)) if n > 1 else np.zeros(1)
        if n > 1:
            ts[0], ts[-1] = 0.0, dur
        dirs = ["in"] * in_p + ["out"] * out_p
        first = dirs.pop(0)  # forward direction = first packet (always an 'in' packet since in_pkts >= 1)
        rng.shuffle(dirs)
        dirs = [first] + dirs

        def split(total, k):
            if k == 0:
                return []
            base, rem = divmod(int(total), k)
            return [base + (1 if j < rem else 0) for j in range(k)]

        in_sizes, out_sizes = split(in_b, in_p), split(out_b, out_p)
        ii = oi = 0
        for d, t in zip(dirs, ts):
            if d == "in":
                pk.append((t0 + t, src, dst, sport, dport, proto, in_sizes[ii]))
                ii += 1
            else:
                pk.append((t0 + t, dst, src, dport, sport, proto, out_sizes[oi]))
                oi += 1
    pk.sort(key=lambda r: r[0])
    return pk


def _stats_ms(samples_ns: list) -> dict:
    a = np.array(samples_ns, dtype=float) / 1e6
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}


def platform_info() -> dict:
    import os
    import sklearn

    return {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(), "sklearn": sklearn.__version__, "numpy": np.__version__}


def run_replay(target_test: pd.DataFrame, models: dict, pipeline, pre, cfg: dict, seed: int) -> dict:
    rc = cfg["replay"]
    rng = np.random.default_rng(seed)
    d = target_test[(target_test["IN_PKTS"] + target_test["OUT_PKTS"] <= rc["max_pkts_per_flow"]) & (target_test["FLOW_DURATION_MS"] / 1000.0 <= rc["max_flow_dur_s"])]
    n_eligible, n_total = len(d), len(target_test)
    if len(d) > rc["max_flows"]:
        d = d.sample(rc["max_flows"], random_state=seed)
    d = d.reset_index(drop=True)
    y = d["label"].to_numpy()

    packets = synthesize_packets(d, rng, rc["window_s"])

    # 1) flow export
    exp = FlowExporter()
    t0 = time.perf_counter()
    for p in packets:
        exp.ingest(*p)
    states = exp.finalize_all()
    t_export = time.perf_counter() - t0

    # restore ground-truth label order: flow index == src id (state key not retained -> recompute by src via sort of first packet)
    # exporter dict preserves insertion order = order of first packet; map back through the first-packet src id
    first_src = {}
    for p in packets:
        if p[1] >= 0 and p[1] not in first_src:
            first_src[p[1]] = None
    order = list(first_src)  # flow indices in order of first appearance (forward packets have src>=0)
    # Forward direction is the first packet of each flow which always has src>=0 (src = flow index)
    labels = y[np.array(order)]
    X = np.vstack([state_to_features(s) for s in states])
    ref = d[SHARED_FEATURES].to_numpy()[np.array(order)]
    fidelity = float(np.max(np.abs(X - ref) / np.maximum(np.abs(ref), 1.0)))

    # 2) per-flow latency (single-flow online scoring), proposed pipeline end-to-end from flow state
    idx = rng.choice(len(states), size=min(rc["single_flow_samples"], len(states)), replace=False)
    for i in idx[:20]:  # warm-up
        pipeline.predict(state_to_features(states[i])[None, :])
    lat_total, lat_feat, lat_inf = [], [], []
    for i in idx:
        a = time.perf_counter_ns()
        x = state_to_features(states[i])[None, :]
        b = time.perf_counter_ns()
        pipeline.predict(x)
        c = time.perf_counter_ns()
        lat_feat.append(b - a)
        lat_inf.append(c - b)
        lat_total.append(c - a)

    # 3) batch scoring and alert volume
    t1 = time.perf_counter()
    preds = np.concatenate([pipeline.predict(X[i:i + rc["batch_size"]]) for i in range(0, len(X), rc["batch_size"])])
    t_batch = time.perf_counter() - t1
    am = alert_metrics(labels, preds)

    # 4) per-model inference latency (model only, after preprocessing)
    per_model = {}
    for name, m in models.items():
        Xs = pre.transform(X[idx])
        for r in Xs[:10]:
            m.predict_proba(r[None, :])
        samples = []
        for r in Xs:
            a = time.perf_counter_ns()
            m.predict_proba(r[None, :])
            samples.append(time.perf_counter_ns() - a)
        Xall = pre.transform(X)
        b0 = time.perf_counter()
        for i in range(0, len(Xall), rc["batch_size"]):
            m.predict_proba(Xall[i:i + rc["batch_size"]])
        tb = time.perf_counter() - b0
        per_model[name] = {"single_flow": _stats_ms(samples), "batch_flows_per_s": len(Xall) / tb}

    n_pk = len(packets)
    return {
        "flows_replayed": int(len(states)),
        "packets_replayed": int(n_pk),
        "eligible_flows": int(n_eligible),
        "test_flows_total": int(n_total),
        "replay_window_s": rc["window_s"],
        "feature_roundtrip_max_rel_error": fidelity,
        "exporter": {"seconds": t_export, "packets_per_s": n_pk / t_export, "us_per_packet": 1e6 * t_export / n_pk},
        "latency_per_flow_proposed": {"feature_build": _stats_ms(lat_feat), "inference_and_decision": _stats_ms(lat_inf), "total": _stats_ms(lat_total)},
        "batch": {"batch_size": rc["batch_size"], "seconds": t_batch, "flows_per_s": len(X) / t_batch},
        "end_to_end": {"packets_per_s": n_pk / (t_export + t_batch), "flows_per_s": len(X) / (t_export + t_batch)},
        "alerts": {
            "policy": pipeline.policy,
            "total_alerts": am["total_alerts"],
            "alerts_per_10k_benign": am["alerts_per_10k_benign"],
            "alerts_per_10k_flows": am["alerts_per_10k_flows"],
            "false_alerts": am["false_alerts"],
            "n_benign": am["n_benign"],
            "detection_rate": am["detection_rate"],
            "macro_f1": prf(labels, preds)["macro_f1"],
        },
        "per_model": per_model,
        "environment": platform_info(),
    }


def model_size_bytes(obj) -> int:
    return len(pickle.dumps(obj))
