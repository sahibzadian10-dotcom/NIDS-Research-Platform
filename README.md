# NIDS Research Platform

**A Lightweight and Explainable Network Intrusion Detection System for Cross-Dataset Generalization and False-Positive Reduction**
Sahibzada Hayam · Malak Arbaz Khan — BS Computer Science, Abdul Wali Khan University Mardan

An experimental research prototype that measures the four gaps in your proposal (G1–G4) and the optional stretch (G5), plus a dashboard to explore the results.

> ## Read this first: the shipped numbers are from SIMULATED data
> The UNSW-NB15 and CIC-IDS2017 downloads were not reachable from the environment this was built in. The shipped results therefore come from **schema-faithful synthetic stand-ins** (same column names, units, label strings and known quirks as the real files, with a designed distribution shift between the two). They prove the pipeline works and show what each measurement looks like. **They are not findings about the real datasets and must not go into your report.** Re-run on the real CSVs (section 4) and every table, figure and dashboard number is regenerated from them.

---

## 1. What it does (mapped to your proposal)

| Proposal item | Where it lives | Output |
|---|---|---|
| Data pipeline: clean, de-duplicate, align shared features, map labels, leakage-aware splits | `src/data/` | Dataset Intelligence tab; audit of every dropped row; leakage report |
| Classical baseline (Random Forest, XGBoost role) + compact MLP | `src/models/` | Size, train time, latency per model |
| Same-dataset and cross-dataset evaluation, both directions | `src/evaluation/cross_dataset.py` | Macro-F1, per-class precision/recall, PR-AUC, transfer drop and asymmetry |
| **G1** alert burden after transfer | `src/evaluation/metrics.py` | FPR and alerts per 10,000 benign flows on the unseen dataset |
| **G2** thresholds learned on source, tested on target | `src/evaluation/thresholds.py` | Default vs calibrated arg-max vs global vs per-class (cost) vs per-class (FPR cap); calibration error |
| **G3** explanation stability | `src/explainability/` | Top-k overlap, Spearman rank correlation, rank shift vs drift and vs errors; per-flow explanations |
| **G4** deployment-style replay | `src/replay/` | Per-flow latency (p50/p95/p99), throughput, alert volume |
| **G5** 1–5 % target labels (stretch, kept separate) | `src/evaluation/finetune.py` | Macro-F1 vs labelled fraction |
| Seeds, mean ± std, saved configuration | `run_experiments.py`, `experiments/configurations/default.json` | `results.json`, `RESULTS.md`, `run_log.txt` |

## 2. Quick start (Windows)

**Just show the dashboard (no installation, works offline):** double-click `open_dashboard.bat`, or open `dashboard/index.html` in Chrome/Edge. The results it shows are already saved in `experiments/results/`.

**Re-run everything (PowerShell, Python 3.11 or newer):**

```powershell
cd <folder where you unzipped>\NIDS-Research-Platform
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked:  Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
$env:PYTHONUTF8 = "1"

python -m pytest -q                        # 15 tests, a few seconds
python run_experiments.py                  # full study, about 4 to 8 minutes, writes experiments/results/
python -m src.reporting.build_all          # figures + RESULTS.md + dashboard/index.html
```

`run_all.bat` does all of the above in one go. `python run_experiments.py --seeds 0 1` is a quick run.

`requirements.txt` holds only the packages the shipped results were produced with. `requirements-optional.txt` (XGBoost, PyTorch, SHAP, Jupyter, Streamlit) switches on code paths that were never run; install it only if you can re-run and compare before presenting.

Optional: `streamlit run dashboard/app.py` (syntax-checked only). `notebooks/01_quick_tour.ipynb` needs `pip install jupyter` and a finished `run_experiments.py` run, because the saved models are not in the zip.

## 3. What was and was not exercised when this was built

| Component | Status |
|---|---|
| Whole pipeline, tests, dashboard | Run end to end on simulated data; 15/15 tests pass |
| Gradient-boosted model | **scikit-learn `HistGradientBoostingClassifier` stands in for XGBoost** (same model family). `src/models/xgboost_model.py` switches to `xgboost` automatically when installed; that branch was **not executed** |
| Neural model | **scikit-learn `MLPClassifier` stands in for PyTorch.** `src/models/neural_network.py` switches to a small PyTorch MLP when `torch` is installed; that branch was **not executed** |
| SHAP | A permutation-sampling **Shapley estimator** (the quantity SHAP approximates) with a completeness check. The `shap` package was not installed. Install `shap` and use `tree_shap_values` in `src/explainability/shap_analysis.py` for exact TreeSHAP on XGBoost |
| Flow exporter | A minimal Python exporter on packets **synthesised from held-out flows**, not real PCAPs and not CICFlowMeter. Latency and throughput are real wall-clock measurements, but on a 2-CPU cloud container, **so re-measure on your laptop** before reporting G4 |
| Real-file loaders | Tested only on small fixtures written in the documented official layouts, never on the real files |

Because the XGBoost and PyTorch branches are unexecuted, run the pipeline once after installing them and compare against the stand-in results before relying on those paths.

## 4. Switching to the real datasets

1. Check each dataset's licence, then download.
2. **UNSW-NB15:** put `UNSW-NB15_1.csv` … `UNSW-NB15_4.csv` in `datasets/raw/UNSW-NB15/` (these carry `srcip`, `dsport`, `Stime`; they have no header row and the loader adds the official one). The `training-set`/`testing-set` partition files lack those columns and are rejected with an explanation.
3. **CIC-IDS2017:** put the `TrafficLabelling` CSVs in `datasets/raw/CIC-IDS2017/` (they carry `Source IP`, `Protocol`, `Timestamp`). The `MachineLearningCVE` CSVs lack them and are rejected with an explanation. Engelen et al. (2021) describe labelling and flow-construction errors in the original CSVs; the corrected release is preferable if you can obtain it.
4. Delete the `*_SIMULATED.csv` files in `datasets/raw/*/`, set `"data_mode": "real"` in `experiments/configurations/default.json`, and run `python run_experiments.py --config experiments/configurations/default.json`.
5. 16 GB RAM is enough for UNSW-NB15 but CIC-IDS2017 is about 2.8 M rows; subsample (for example one or two day files) if memory is tight and **say so in the report**.
6. Review `src/data/label_mapping.py` (merged classes such as web attacks → Exploit and Backdoor → Botnet are approximations) and the "excluded" rows in the Data tab (TTL, TCP state, inter-arrival time were left out because the two exporters define them differently).

Everything the models see is a 13-feature NetFlow-style set that exists in both datasets. Source IP and timestamp are used **only** to group flows into sessions so no session straddles train and test.

## 5. Leakage controls (all covered by tests)

- Duplicates removed before splitting; the report shows zero duplicate or session overlap between train, validation and test.
- Scaling statistics and calibrators are fitted on source training / validation data only.
- All thresholds are learned on **source validation** data. The target dataset is only ever used to score.
- The stretch experiment (G5) draws labelled target flows from the target *train* partition and evaluates on the target *test* partition, and never feeds back into the zero-label results.

## 6. Project layout

```
run_experiments.py            end-to-end study -> experiments/results/
src/config.py                 classes, shared features, default configuration
src/data/                     synthetic.py (simulator) loaders.py cleaning.py feature_alignment.py
                              label_mapping.py splitting.py preprocessing.py
src/models/                   random_forest.py xgboost_model.py neural_network.py pipeline.py
src/evaluation/               metrics.py cross_dataset.py thresholds.py robustness.py finetune.py
src/explainability/           shap_analysis.py feature_stability.py explain_runner.py
src/replay/                   flow_exporter.py performance.py
src/reporting/                charts.py reports.py build_all.py
dashboard/                    template.html + build_dashboard.py -> index.html ; app.py (Streamlit, untested)
experiments/configurations/   default.json
experiments/results/          results.json RESULTS.md run_log.txt figures/
models/                       saved seed-0 pipelines (*.pkl, git-ignored)
tests/                        15 tests
notebooks/                    01_quick_tour.ipynb
```

## 7. Honest scope notes

- Standard deviations come from five seeds that vary splits and model initialisation; the two datasets are fixed, so dataset-sampling variability is not included.
- With 13 features, the rank-shift correlations in the explanation study are indicative only.
- The proposal's own scope applies: no payload inspection, no live-network claims, no automated blocking. The replay is a feasibility check.
- Negative findings are results. The shipped simulated run, for instance, shows per-class cost thresholds performing about the same as a single global threshold; the dashboard and `RESULTS.md` report this as measured.
