"""Train on a source dataset, evaluate same-dataset and cross-dataset under every decision policy.

Protocol (zero-label transfer): preprocessing, model, calibrator and thresholds are ALL learned from the
source dataset (train / val partitions).  The target dataset contributes only its held-out TEST partition,
purely for evaluation.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import CLASSES, SHARED_FEATURES
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import full_report, reliability_curve, score_stats
from src.evaluation.thresholds import POLICY_ORDER, IsotonicCalibrator, apply_policy, learn_policies
from src.models import MODEL_BUILDERS
from src.models.neural_network import n_parameters
from src.models.pipeline import NIDSPipeline
from src.replay.performance import model_size_bytes


def fit_source(src: pd.DataFrame, split: dict, cfg: dict, seed: int):
    """Fit preprocessor, every model, calibrators and threshold policies on the SOURCE dataset only."""
    ac = cfg["alert_control"]
    pre = Preprocessor().fit(src.iloc[split["train"]])
    Xtr, ytr = pre.transform(src.iloc[split["train"]]), src.iloc[split["train"]]["label"].to_numpy()
    Xva, yva = pre.transform(src.iloc[split["val"]]), src.iloc[split["val"]]["label"].to_numpy()
    fitted = {}
    for name, build in MODEL_BUILDERS.items():
        m = build(cfg, seed)
        t = time.perf_counter()
        m.fit(Xtr, ytr)
        train_s = time.perf_counter() - t
        assert list(m.classes_) == list(range(len(CLASSES)))
        P_va = m.predict_proba(Xva)
        cal = IsotonicCalibrator().fit(P_va, yva)
        thr = learn_policies(cal.transform(P_va), yva, ac)
        fitted[name] = {
            "model": m, "cal": cal, "thr": thr, "train_time_s": train_s,
            "size_kb": model_size_bytes(m) / 1024.0, "n_params": n_parameters(m),
        }
    return pre, fitted


def evaluate_bundle(entry: dict, pre: Preprocessor, sets: dict, ac: dict, keep_scores: bool = False):
    """sets: name -> DataFrame (with 'label'). Returns reports per set per policy (+ optional raw arrays)."""
    out, raw = {}, {}
    for sname, df in sets.items():
        X, y = pre.transform(df), df["label"].to_numpy()
        P_raw = entry["model"].predict_proba(X)
        P_cal = entry["cal"].transform(P_raw)
        ss = {"raw": score_stats(y, P_raw), "cal": score_stats(y, P_cal)}
        out[sname] = {}
        for pol in POLICY_ORDER:
            pred = apply_policy(pol, P_raw, P_cal, entry["thr"])
            out[sname][pol] = full_report(y, pred, ac, ss["raw" if pol == "default_argmax" else "cal"])
        if keep_scores:
            raw[sname] = {"P_raw": P_raw, "P_cal": P_cal, "y": y}
    return out, raw


def run_direction(src: pd.DataFrame, tgt: pd.DataFrame, sp_src: dict, sp_tgt: dict, cfg: dict, seed: int):
    ac = cfg["alert_control"]
    pre, fitted = fit_source(src, sp_src, cfg, seed)
    sets = {"source_test": src.iloc[sp_src["test"]], "target_test": tgt.iloc[sp_tgt["test"]]}
    result, artifacts = {}, {"pre": pre, "fitted": fitted}
    for name, entry in fitted.items():
        rep, raw = evaluate_bundle(entry, pre, sets, ac, keep_scores=True)
        result[name] = {
            "eval": rep,
            "thresholds": {k: [float(x) for x in v] for k, v in entry["thr"].items()},
            "train_time_s": entry["train_time_s"],
            "size_kb": entry["size_kb"],
            "n_params": entry["n_params"],
        }
        artifacts.setdefault("scores", {})[name] = raw
    # reliability diagrams for the proposed model (target test), raw vs calibrated
    tp = artifacts["scores"]["xgboost"]["target_test"]
    result["xgboost"]["reliability_target"] = {
        "raw": reliability_curve(tp["y"], tp["P_raw"]),
        "calibrated": reliability_curve(tp["y"], tp["P_cal"]),
    }
    return result, artifacts


def build_pipeline(artifacts: dict, policy: str = "per_class_cost") -> NIDSPipeline:
    e = artifacts["fitted"]["xgboost"]
    return NIDSPipeline(artifacts["pre"], e["model"], e["cal"], e["thr"], policy)
