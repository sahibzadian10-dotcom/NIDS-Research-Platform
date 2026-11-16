"""Run the full study: data intelligence -> models -> cross-dataset -> alert control -> SHAP -> replay -> stretch.

Usage:
    python run_experiments.py                 # default config, 5 seeds
    python run_experiments.py --seeds 0 1     # fewer seeds (quick run)
    python run_experiments.py --config experiments/configurations/default.json
Outputs experiments/results/results.json (consumed by the dashboard) and experiments/results/run_log.txt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ATTACK_CLASSES, CLASSES, DATASETS, RAW_DIR, RESULTS_DIR, SHARED_FEATURES, load_config
from src.data.cleaning import clean_raw
from src.data.feature_alignment import ALIGNERS, SCHEMA_DECISIONS, finalize
from src.data.label_mapping import map_labels
from src.data.loaders import load_dataset, profile_raw
from src.data.splitting import dedupe_shared, leakage_aware_split, leakage_report
from src.data.synthetic import write_raw_datasets
from src.evaluation.cross_dataset import build_pipeline, run_direction
from src.evaluation.finetune import run_finetune
from src.evaluation.robustness import aggregate
from src.evaluation.thresholds import POLICY_LABELS, POLICY_ORDER
from src.explainability.explain_runner import explain_direction
from src.explainability.feature_stability import ks_per_feature
from src.models import BACKENDS, MODEL_LABELS
from src.replay.performance import platform_info, run_replay
from src.reporting.reports import make_findings, write_markdown_report

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
LABEL_COL = {"UNSW-NB15": "attack_cat", "CIC-IDS2017": "Label"}
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def prepare_dataset(name: str, cfg: dict):
    raw = load_dataset(name, RAW_DIR)
    profile = profile_raw(raw, name, LABEL_COL[name])
    clean, audit = clean_raw(raw, name)
    mapped, table = map_labels(clean[LABEL_COL[name]], name)
    keep = mapped.notna().to_numpy()
    aligned = ALIGNERS[name](clean)
    fin = finalize(aligned[keep], mapped[keep].map({c: i for i, c in enumerate(CLASSES)}).astype(int), name)
    n_before_dedupe = len(fin)
    fin, n_dup = dedupe_shared(fin)
    info = {
        "raw_profile": profile,
        "cleaning_audit": audit,
        "label_mapping": table.to_dict("records"),
        "rows_dropped_unmapped_labels": int((~keep).sum()),
        "rows_after_mapping": int(n_before_dedupe),
        "shared_feature_duplicates_removed": int(n_dup),
        "final_rows": int(len(fin)),
        "n_sessions": int(fin["session_id"].nunique()),
        "class_counts": {c: int((fin["label"] == i).sum()) for i, c in enumerate(CLASSES)},
        "class_share": {c: float((fin["label"] == i).mean()) for i, c in enumerate(CLASSES)},
    }
    return fin, info


def shift_report(frames: dict) -> dict:
    U, C = frames["UNSW-NB15"], frames["CIC-IDS2017"]
    F = SHARED_FEATURES
    ks_all = ks_per_feature(U[F].to_numpy(), C[F].to_numpy())
    ks_ben = ks_per_feature(U[U.label == 0][F].to_numpy(), C[C.label == 0][F].to_numpy())
    ks_att = ks_per_feature(U[U.label != 0][F].to_numpy(), C[C.label != 0][F].to_numpy())
    hist = {}
    for f in ["FLOW_DURATION_MS", "IN_PKTS", "MEAN_PKT_SIZE", "BYTES_PER_SEC"]:
        allv = np.log10(1 + np.concatenate([U[f].to_numpy(), C[f].to_numpy()]))
        edges = np.linspace(np.percentile(allv, 0.5), np.percentile(allv, 99.5), 31)
        h = {"edges": [float(x) for x in edges]}
        for dsn, df in (("UNSW-NB15", U), ("CIC-IDS2017", C)):
            for grp, m in (("benign", df.label == 0), ("attack", df.label != 0)):
                v = np.log10(1 + df.loc[m, f].to_numpy())
                cnt, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
                h[f"{dsn}|{grp}"] = [float(x) for x in cnt / max(cnt.sum(), 1)]
        hist[f] = h
    prof = {}
    for dsn, df in frames.items():
        prof[dsn] = {c: {f: float(df.loc[df.label == i, f].median()) for f in ["FLOW_DURATION_MS", "IN_PKTS", "OUT_PKTS", "MEAN_PKT_SIZE", "IN_OUT_PKT_RATIO"]} for i, c in enumerate(CLASSES)}
    return {
        "ks_all": {f: float(v) for f, v in zip(F, ks_all)},
        "ks_benign": {f: float(v) for f, v in zip(F, ks_ben)},
        "ks_attack": {f: float(v) for f, v in zip(F, ks_att)},
        "histograms": hist,
        "class_medians": prof,
    }


def explorer_export(art, tgt, sp_tgt, direction_result):
    """Calibrated target-test scores of the proposed model (seed 0) for the dashboard's live threshold explorer."""
    sc = art["scores"]["xgboost"]["target_test"]
    return {
        "classes": CLASSES,
        "y": [int(v) for v in sc["y"]],
        "p": [int(round(1000 * v)) for v in sc["P_cal"].ravel()],
        "thresholds": direction_result["xgboost"]["thresholds"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "results.json"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.seeds:
        cfg["seeds"] = args.seeds
    t_start = time.time()

    missing = [d for d in DATASETS if not any((RAW_DIR / d).glob("*.csv"))]
    if missing:
        if cfg["data_mode"] != "simulated":
            sys.exit(f"Missing real dataset CSVs in datasets/raw/: {missing}")
        log("generating simulated raw datasets ...")
        write_raw_datasets(RAW_DIR, cfg)

    # ---------------- data intelligence ----------------
    frames, dinfo = {}, {}
    for name in DATASETS:
        frames[name], dinfo[name] = prepare_dataset(name, cfg)
        log(f"{name}: raw {dinfo[name]['raw_profile']['rows']} -> final {dinfo[name]['final_rows']} flows; classes {dinfo[name]['class_counts']}")
    shift = shift_report(frames)
    file_hashes = {d: hashlib.sha256(Path(sorted((RAW_DIR / d).glob('*.csv'))[0]).read_bytes()).hexdigest()[:16] for d in DATASETS}

    # ---------------- seeds x directions ----------------
    directions = [("UNSW-NB15", "CIC-IDS2017"), ("CIC-IDS2017", "UNSW-NB15")]
    runs = {f"{s}->{t}": {} for s, t in directions}
    explain_runs = {f"{s}->{t}": {} for s, t in directions}
    stretch_runs = {f"{s}->{t}": {} for s, t in directions}
    local_ex, replay_res, explorer = {}, {}, {}
    leak = {}
    for seed in cfg["seeds"]:
        splits = {n: leakage_aware_split(frames[n], seed, **cfg["split"]) for n in DATASETS}
        if seed == cfg["seeds"][0]:
            leak = {n: leakage_report(frames[n], splits[n]) for n in DATASETS}
        for s, t in directions:
            key = f"{s}->{t}"
            t0 = time.time()
            res, art = run_direction(frames[s], frames[t], splits[s], splits[t], cfg, seed)
            runs[key][seed] = res
            first = seed == cfg["seeds"][0]
            explain_runs[key][seed] = explain_direction(frames[s], frames[t], splits[s], splits[t], art, cfg, seed, local_examples=first)
            stretch_runs[key][seed] = run_finetune(frames[s], frames[t], splits[s], splits[t], art["pre"], cfg, seed)
            if first:
                local_ex[key] = explain_runs[key][seed].pop("local", [])
                explorer[key] = explorer_export(art, frames[t], splits[t], res)
                pipe = build_pipeline(art)
                models = {k: v["model"] for k, v in art["fitted"].items()}
                joblib.dump({"pipeline": pipe, "models": models, "preprocessing": art["pre"], "pipeline_thresholds": res["xgboost"]["thresholds"],
                             "seed": seed, "direction": key, "data_mode": cfg["data_mode"]},
                            MODELS_DIR / f"{s}_to_{t}_seed{seed}.pkl")
                replay_res[key] = run_replay(frames[t].iloc[splits[t]["test"]], models, pipe, art["pre"], cfg, seed)
                log(f"replay {key}: {replay_res[key]['flows_replayed']} flows, {replay_res[key]['packets_replayed']} pkts")
            f1s = res["xgboost"]["eval"]
            log(f"seed {seed} {key}: GB macroF1 src={f1s['source_test']['default_argmax']['macro_f1']:.3f} tgt={f1s['target_test']['default_argmax']['macro_f1']:.3f} ({time.time()-t0:.0f}s)")

    # ---------------- aggregate ----------------
    experiments = {}
    for key, per_seed in runs.items():
        seeds = sorted(per_seed)
        experiments[key] = {}
        for m in per_seed[seeds[0]]:
            agg = aggregate([per_seed[s][m] for s in seeds])
            agg["thresholds"] = per_seed[seeds[0]][m]["thresholds"]
            if "reliability_target" in per_seed[seeds[0]][m]:
                agg["reliability_target"] = per_seed[seeds[0]][m]["reliability_target"]
            experiments[key][m] = agg

    transfer = {}
    k1, k2 = list(runs)[0], list(runs)[1]
    for m in MODEL_LABELS:
        per_seed = []
        for s in sorted(runs[k1]):
            d = {}
            for pol in POLICY_ORDER:
                f = lambda k, ev: runs[k][s][m]["eval"][ev][pol]["macro_f1"]  # noqa: E731
                d[pol] = {
                    "drop_fwd": f(k1, "source_test") - f(k1, "target_test"),
                    "drop_rev": f(k2, "source_test") - f(k2, "target_test"),
                    "asymmetry": abs(f(k1, "target_test") - f(k2, "target_test")),
                }
            per_seed.append(d)
        transfer[m] = aggregate(per_seed)

    explain = {}
    for key, per_seed in explain_runs.items():
        seeds = sorted(per_seed)
        explain[key] = aggregate([per_seed[s] for s in seeds])
        explain[key]["local"] = local_ex.get(key, [])
    stretch = {key: aggregate([per_seed[s] for s in sorted(per_seed)]) for key, per_seed in stretch_runs.items()}

    results = {
        "meta": {
            "title": cfg["project"], "authors": cfg["authors"], "institution": cfg["institution"],
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "data_mode": cfg["data_mode"],
            "data_notice": ("SIMULATED DATA. The numbers in this report come from schema-faithful synthetic stand-ins for UNSW-NB15 and CIC-IDS2017, "
                            "not from the real datasets. They demonstrate the pipeline and the measurement methodology, not real-world performance."),
            "seeds": cfg["seeds"], "config": cfg,
            "classes": CLASSES, "attack_classes": ATTACK_CLASSES, "features": SHARED_FEATURES,
            "models": MODEL_LABELS, "backends": BACKENDS,
            "policies": POLICY_LABELS, "policy_order": POLICY_ORDER,
            "dataset_file_hashes": file_hashes, "environment": platform_info(),
            "runtime_s": round(time.time() - t_start, 1),
        },
        "datasets": dinfo, "leakage": leak, "schema": SCHEMA_DECISIONS, "shift": shift,
        "experiments": experiments, "transfer": transfer, "explain": explain,
        "replay": replay_res, "stretch": stretch, "explorer": explorer,
    }
    results["findings"], results["limitations"] = make_findings(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, separators=(",", ":")), encoding="utf-8")
    write_markdown_report(results, out.parent / "RESULTS.md")
    (out.parent / "run_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    log(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB) in {time.time()-t_start:.0f}s")
    return results


if __name__ == "__main__":
    main()
