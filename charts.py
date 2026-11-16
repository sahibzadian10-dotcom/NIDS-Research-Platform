"""Static matplotlib figures for the written report (the dashboard has the interactive versions)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FWD, REV = "UNSW-NB15->CIC-IDS2017", "CIC-IDS2017->UNSW-NB15"
C = {"same": "#4c78a8", "cross": "#e4572e", "a": "#4c78a8", "b": "#f2a541"}


def _m(x):
    return x["mean"]


def _s(x):
    return x["std"]


def make_charts(R: dict, out_dir) -> list:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    E, X, RP = R["experiments"], R["explain"], R["replay"]
    models = list(R["meta"]["models"])
    files = []

    # 1. same vs cross macro-F1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, k in zip(axes, (FWD, REV)):
        x = np.arange(len(models))
        s = [_m(E[k][m]["eval"]["source_test"]["default_argmax"]["macro_f1"]) for m in models]
        t = [_m(E[k][m]["eval"]["target_test"]["default_argmax"]["macro_f1"]) for m in models]
        se = [_s(E[k][m]["eval"]["source_test"]["default_argmax"]["macro_f1"]) for m in models]
        te = [_s(E[k][m]["eval"]["target_test"]["default_argmax"]["macro_f1"]) for m in models]
        ax.bar(x - 0.2, s, 0.4, yerr=se, color=C["same"], label="same-dataset")
        ax.bar(x + 0.2, t, 0.4, yerr=te, color=C["cross"], label="cross-dataset")
        ax.set_xticks(x, ["RF", "GB", "MLP"])
        ax.set_title(k.replace("->", " → "))
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("macro-F1")
    axes[0].legend()
    fig.suptitle("Same-dataset vs cross-dataset macro-F1 (SIMULATED data)")
    fig.tight_layout()
    f = out / "fig_macro_f1.png"
    fig.savefig(f, dpi=160)
    plt.close(fig)
    files.append(f.name)

    # 2. alert burden per policy
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    pols = R["meta"]["policy_order"]
    for ax, k in zip(axes, (FWD, REV)):
        v = [_m(E[k]["xgboost"]["eval"]["target_test"][p]["alerts_per_10k_benign"]) for p in pols]
        ax.bar(range(len(pols)), v, color=C["cross"])
        ax.set_xticks(range(len(pols)), [R["meta"]["policies"][p].replace(" ", "\n", 1) for p in pols], fontsize=7)
        ax.set_title(k.replace("->", " → "))
    axes[0].set_ylabel("false alerts per 10,000 benign flows")
    fig.suptitle("Alert burden on the unseen dataset by decision policy (SIMULATED data)")
    fig.tight_layout()
    f = out / "fig_alert_burden.png"
    fig.savefig(f, dpi=160)
    plt.close(fig)
    files.append(f.name)

    # 3. SHAP importance both datasets
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, k in zip(axes, (FWD, REV)):
        feats = list(X[k]["importance_source"])
        a = np.array([_m(X[k]["importance_source"][f]) for f in feats])
        b = np.array([_m(X[k]["importance_target"][f]) for f in feats])
        order = np.argsort(-(a + b))
        y = np.arange(len(feats))
        ax.barh(y - 0.2, a[order], 0.4, color=C["a"], label="source")
        ax.barh(y + 0.2, b[order], 0.4, color=C["b"], label="target")
        ax.set_yticks(y, [feats[i] for i in order], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{k.replace('->', ' → ')}\nSpearman ρ = {_m(X[k]['stability_topk']['spearman_rho']):.2f}")
    axes[0].legend()
    fig.suptitle("Mean |attribution| share (attack score) — source vs target (SIMULATED data)")
    fig.tight_layout()
    f = out / "fig_shap_stability.png"
    fig.savefig(f, dpi=160)
    plt.close(fig)
    files.append(f.name)

    # 4. latency
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, k in enumerate((FWD, REV)):
        l = RP[k]["latency_per_flow_proposed"]["total"]
        ax.bar(i * 4 + np.arange(3), [l["p50_ms"], l["p95_ms"], l["p99_ms"]], color=C["a"] if i == 0 else C["b"])
    ax.set_xticks([0, 1, 2, 4, 5, 6], ["p50", "p95", "p99"] * 2)
    ax.set_ylabel("ms per flow")
    ax.set_title("Single-flow latency, proposed pipeline (left UNSW→CIC, right CIC→UNSW)", fontsize=8)
    fig.tight_layout()
    f = out / "fig_latency.png"
    fig.savefig(f, dpi=160)
    plt.close(fig)
    files.append(f.name)
    return files
