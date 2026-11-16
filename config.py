"""Central configuration: class set, shared feature set, default experiment config."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "raw"
META_DIR = ROOT / "datasets" / "metadata"
RESULTS_DIR = ROOT / "experiments" / "results"
CONFIG_DIR = ROOT / "experiments" / "configurations"

# Common class set after label mapping (index = integer label used everywhere).
CLASSES = ["Benign", "DoS", "Recon", "Exploit", "Botnet"]
ATTACK_CLASSES = CLASSES[1:]

# Shared NetFlow-style feature set (13 features) available in BOTH datasets.
SHARED_FEATURES = [
    "PROTOCOL",
    "DST_PORT_CLASS",
    "FLOW_DURATION_MS",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "MEAN_PKT_SIZE",
    "MEAN_IN_PKT_SIZE",
    "MEAN_OUT_PKT_SIZE",
    "BYTES_PER_SEC",
    "PKTS_PER_SEC",
    "IN_OUT_PKT_RATIO",
]
# Heavy-tailed features that get log1p before scaling.
LOG_FEATURES = [f for f in SHARED_FEATURES if f not in ("PROTOCOL", "DST_PORT_CLASS")]

DATASETS = ["UNSW-NB15", "CIC-IDS2017"]

DEFAULT_CONFIG = {
    "project": "A Lightweight and Explainable NIDS for Cross-Dataset Generalization and False-Positive Reduction",
    "authors": ["Sahibzada Hayam", "Malak Arbaz Khan"],
    "institution": "Abdul Wali Khan University Mardan",
    "data_mode": "simulated",  # "simulated" generates schema-faithful stand-in data; "real" expects real CSVs in datasets/raw
    "dataset_seed": 2026,
    "n_flows": {"UNSW-NB15": 60000, "CIC-IDS2017": 60000},
    "seeds": [0, 1, 2, 3, 4],
    "split": {"test_folds": 4, "val_folds": 5},
    "models": {
        "random_forest": {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 2},
        "xgboost": {"max_iter": 150, "learning_rate": 0.1, "max_depth": 6},
        "mlp": {"hidden": [32, 16], "max_iter": 60, "alpha": 1e-4},
    },
    "alert_control": {
        "cost_fp": 1.0,
        "cost_confusion": 1.0,
        "cost_fn": {"DoS": 5.0, "Recon": 3.0, "Exploit": 8.0, "Botnet": 8.0},
        "fpr_cap_per_class": 0.001,
        "grid_min": 0.05,
        "grid_max": 0.99,
        "grid_step": 0.01,
    },
    "shap": {"n_explain": 400, "n_perm": 40, "n_background": 200, "top_k": 5, "n_error_probe": 300},
    "replay": {"max_flows": 4000, "max_pkts_per_flow": 100, "max_flow_dur_s": 30.0, "window_s": 60.0, "single_flow_samples": 300, "batch_size": 256},
    "stretch": {"fractions": [0.01, 0.02, 0.05]},
}


def load_config(path: str | Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, encoding="utf-8") as fh:
            user = json.load(fh)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg
