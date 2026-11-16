# A Lightweight and Explainable NIDS for Cross-Dataset Generalization and False-Positive Reduction

_Authors: Sahibzada Hayam, Malak Arbaz Khan — Abdul Wali Khan University Mardan_

> **SIMULATED DATA. The numbers in this report come from schema-faithful synthetic stand-ins for UNSW-NB15 and CIC-IDS2017, not from the real datasets. They demonstrate the pipeline and the measurement methodology, not real-world performance.**

Generated 2026-11-16T22:29:47 · seeds [0, 1, 2, 3, 4] · runtime 156.1 s

## Findings

**[G1] UNSW→CIC: macro-F1 falls from 0.91 to 0.47**  
Same-dataset macro-F1 is 0.914 ± 0.004 but the unseen dataset gives 0.470 ± 0.005. Benign false-alarm rate moves from 1.30 ± 0.10 % to 1.12 ± 0.09 % (130 → 112 alerts per 10,000 benign flows) while the attack detection rate is 37.0 ± 0.7 %. Classes that collapse (recall < 0.2): DoS, Exploit, Botnet. Classes that transfer well (recall ≥ 0.8): Recon.

**[G1] CIC→UNSW: macro-F1 falls from 0.88 to 0.38**  
Same-dataset macro-F1 is 0.879 ± 0.003 but the unseen dataset gives 0.377 ± 0.008. Benign false-alarm rate moves from 1.36 ± 0.07 % to 1.68 ± 0.12 % (136 → 168 alerts per 10,000 benign flows) while the attack detection rate is 18.6 ± 1.0 %. Classes that collapse (recall < 0.2): DoS, Recon, Botnet. No class transfers with recall ≥ 0.8.

**[G1] Transfer is asymmetric**  
Macro-F1 drop is 0.445 ± 0.004 for UNSW→CIC and 0.503 ± 0.010 for CIC→UNSW; the two target scores differ by 0.093 ± 0.007. Which dataset is used as source matters, as reported by Layeghy & Portmann (2022).

**[G2] UNSW→CIC: alerts per 10k benign — default 112, global 213, per-class 208, FPR-cap 36**  
On the unseen dataset, per-class cost thresholds give about the same benign alert volume as a single global threshold (208 vs 213 per 10,000 benign flows), with macro-F1 0.543 vs 0.543. Against the default arg-max rule (0.470, 112 alerts) they raise macro-F1 but produce more false alerts, because the cost matrix charges a missed attack more than a false alarm. The FPR-cap policy is the false-positive-reduction setting: 36 alerts per 10,000 benign flows at macro-F1 0.333. All thresholds were learned on source validation data only.

**[G2] CIC→UNSW: alerts per 10k benign — default 168, global 622, per-class 691, FPR-cap 47**  
On the unseen dataset, per-class cost thresholds give higher benign alert volume than a single global threshold (691 vs 622 per 10,000 benign flows), with macro-F1 0.553 vs 0.545. Against the default arg-max rule (0.377, 168 alerts) they raise macro-F1 but produce more false alerts, because the cost matrix charges a missed attack more than a false alarm. The FPR-cap policy is the false-positive-reduction setting: 47 alerts per 10,000 benign flows at macro-F1 0.293. All thresholds were learned on source validation data only.

**[G2] Source-fitted calibration under shift**  
Top-label ECE on the unseen dataset: UNSW→CIC 0.084 → 0.082 after isotonic calibration; CIC→UNSW 0.098 → 0.084. Calibration is fitted on source validation data, so a change here measures how well it survives the shift.

**[G3] UNSW→CIC: top-5 SHAP overlap 4.2 / 5**  
Top-5 feature overlap between source and target is 4.2 ± 0.4 of 5; Spearman rank correlation of importance is 0.86 ± 0.04. Across features, rank shift correlates with benign distribution shift (KS) at ρ = 0.49 ± 0.19 and with attribution enrichment on errors at ρ = 0.13 ± 0.21 (up to 300 errors probed per seed; 13 features, so treat these as indicative).

**[G3] CIC→UNSW: top-5 SHAP overlap 3.8 / 5**  
Top-5 feature overlap between source and target is 3.8 ± 0.4 of 5; Spearman rank correlation of importance is 0.56 ± 0.09. Across features, rank shift correlates with benign distribution shift (KS) at ρ = -0.19 ± 0.16 and with attribution enrichment on errors at ρ = 0.40 ± 0.21 (up to 300 errors probed per seed; 13 features, so treat these as indicative).

**[G4] UNSW→CIC replay: p50 2.96 ms/flow, 821562 pkts/s**  
4000 flows / 71637 packets replayed. Single-flow latency (feature build + preprocessing + model + calibration + decision): p50 2.96 ms, p95 3.88 ms, p99 4.52 ms. Flow export 2227962 pkts/s; micro-batched scoring 72671 flows/s; end-to-end 821562 pkts/s. Alert volume: 354 alerts, 215 per 10,000 benign flows. Exporter feature round-trip max relative error: 3.50e-12. Measured on 12 CPUs; re-measure on your own machine for the report.

**[G4] CIC→UNSW replay: p50 2.62 ms/flow, 864893 pkts/s**  
4000 flows / 70812 packets replayed. Single-flow latency (feature build + preprocessing + model + calibration + decision): p50 2.62 ms, p95 3.66 ms, p99 3.86 ms. Flow export 2254556 pkts/s; micro-batched scoring 79262 flows/s; end-to-end 864893 pkts/s. Alert volume: 464 alerts, 622 per 10,000 benign flows. Exporter feature round-trip max relative error: 1.10e-12. Measured on 12 CPUs; re-measure on your own machine for the report.

**[G5] UNSW→CIC: fine-tuning macro-F1 (0% → 0.470 ± 0.005, 1% → 0.474 ± 0.010, 2% → 0.490 ± 0.011, 5% → 0.540 ± 0.014)**  
Stretch experiment, kept separate from zero-label transfer. Labelled target flows come only from the target train partition; evaluation is on the target test partition. Macro-F1 by labelled fraction: 0% → 0.470 ± 0.005, 1% → 0.474 ± 0.010, 2% → 0.490 ± 0.011, 5% → 0.540 ± 0.014.

**[G5] CIC→UNSW: fine-tuning macro-F1 (0% → 0.377 ± 0.008, 1% → 0.487 ± 0.014, 2% → 0.568 ± 0.021, 5% → 0.672 ± 0.019)**  
Stretch experiment, kept separate from zero-label transfer. Labelled target flows come only from the target train partition; evaluation is on the target test partition. Macro-F1 by labelled fraction: 0% → 0.377 ± 0.008, 1% → 0.487 ± 0.014, 2% → 0.568 ± 0.021, 5% → 0.672 ± 0.019.

## Cross-dataset macro-F1 (default arg-max, mean ± std over seeds)

| Model | Direction | Same-dataset | Cross-dataset |
|---|---|---|---|
| Random Forest | UNSW→CIC | 0.913 ± 0.004 | 0.460 ± 0.004 |
| Random Forest | CIC→UNSW | 0.878 ± 0.001 | 0.317 ± 0.008 |
| Gradient-boosted trees (XGBoost role) | UNSW→CIC | 0.914 ± 0.004 | 0.470 ± 0.005 |
| Gradient-boosted trees (XGBoost role) | CIC→UNSW | 0.879 ± 0.003 | 0.377 ± 0.008 |
| Compact MLP | UNSW→CIC | 0.915 ± 0.003 | 0.498 ± 0.008 |
| Compact MLP | CIC→UNSW | 0.883 ± 0.007 | 0.504 ± 0.030 |

## Alert burden on the unseen dataset (proposed model)

| Direction | Policy | Macro-F1 | FPR % | Alerts / 10k benign | Detection rate % | Cost / flow |
|---|---|---|---|---|---|---|
| UNSW→CIC | Default (raw arg-max) | 0.470 ± 0.005 | 1.12 ± 0.09 | 112 ± 9 | 37.0 ± 0.7 | 0.488 ± 0.003 |
| UNSW→CIC | Calibrated arg-max | 0.480 ± 0.013 | 1.25 ± 0.15 | 125 ± 15 | 38.3 ± 1.5 | 0.483 ± 0.007 |
| UNSW→CIC | Global threshold | 0.543 ± 0.011 | 2.13 ± 0.13 | 213 ± 13 | 43.4 ± 0.9 | 0.452 ± 0.007 |
| UNSW→CIC | Per-class thresholds (cost) | 0.543 ± 0.011 | 2.08 ± 0.09 | 208 ± 9 | 43.2 ± 0.8 | 0.452 ± 0.007 |
| UNSW→CIC | Per-class thresholds (FPR cap) | 0.333 ± 0.013 | 0.36 ± 0.05 | 36 ± 5 | 17.1 ± 3.5 | 0.580 ± 0.013 |
| CIC→UNSW | Default (raw arg-max) | 0.377 ± 0.008 | 1.68 ± 0.12 | 168 ± 12 | 18.6 ± 1.0 | 0.653 ± 0.013 |
| CIC→UNSW | Calibrated arg-max | 0.465 ± 0.019 | 3.08 ± 0.28 | 308 ± 28 | 29.6 ± 1.1 | 0.576 ± 0.013 |
| CIC→UNSW | Global threshold | 0.545 ± 0.014 | 6.22 ± 0.47 | 622 ± 47 | 45.5 ± 3.1 | 0.473 ± 0.032 |
| CIC→UNSW | Per-class thresholds (cost) | 0.553 ± 0.011 | 6.91 ± 0.74 | 691 ± 74 | 48.2 ± 4.0 | 0.450 ± 0.037 |
| CIC→UNSW | Per-class thresholds (FPR cap) | 0.293 ± 0.022 | 0.47 ± 0.09 | 47 ± 9 | 7.6 ± 2.4 | 0.746 ± 0.025 |

## Limitations

- All numbers come from SIMULATED stand-ins for UNSW-NB15 and CIC-IDS2017. They validate the pipeline and measurement method, not real-world detection performance. Re-run on the real CSVs before reporting any result.
- The distribution shift between the two datasets was designed into the generator (signature shifts per attack class). Which classes collapse and by how much is therefore a property of the simulation, not a finding about the real datasets.
- The gradient-boosted model is scikit-learn's HistGradientBoosting (XGBoost was not installable here); the MLP is scikit-learn's MLPClassifier (PyTorch was not installable). The XGBoost/PyTorch code paths are included but were not executed here.
- SHAP values are computed with a permutation-sampling Shapley estimator (the quantity SHAP approximates), not the `shap` package. With the package installed, exact TreeSHAP can be used for XGBoost.
- Replay uses packets synthesised from held-out flows and a minimal Python flow exporter, not real PCAPs or CICFlowMeter. Latency/throughput are real wall-clock measurements on the sandbox machine (CPU count in the environment block) and will differ on your laptop.
- Only flow features shared by both datasets are used (13). Port number is bucketed, TTL, TCP state/flags and inter-arrival features are excluded because they are not comparable across the two exporters.
- Five seeds vary splits and model initialisation; the two datasets themselves are fixed, so the reported standard deviation does not include dataset-sampling variability.
- Class mapping merges CIC web attacks into 'Exploit' and UNSW 'Backdoor' into 'Botnet'. These are approximations and should be reviewed against the real data.