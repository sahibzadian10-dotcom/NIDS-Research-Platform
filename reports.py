"""Data-driven findings + Markdown report. Every sentence is generated from results.json numbers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

FWD, REV = "UNSW-NB15->CIC-IDS2017", "CIC-IDS2017->UNSW-NB15"
SHORT = {"UNSW-NB15": "UNSW", "CIC-IDS2017": "CIC"}


def _m(x):
    return x["mean"] if isinstance(x, dict) and "mean" in x else x


def pm(x, nd=3, pct=False):
    if not isinstance(x, dict):
        return f"{x:.{nd}f}"
    s = 100 if pct else 1
    return f"{x['mean']*s:.{nd}f} ± {x['std']*s:.{nd}f}"


def short_dir(k):
    a, b = k.split("->")
    return f"{SHORT[a]}→{SHORT[b]}"


def make_findings(R: dict):
    E, T, X, RP, ST = R["experiments"], R["transfer"], R["explain"], R["replay"], R["stretch"]
    px = "xgboost"
    F: list[dict] = []

    # ---- G1: cross-dataset alert burden ----
    for k in (FWD, REV):
        s = E[k][px]["eval"]["source_test"]["default_argmax"]
        t = E[k][px]["eval"]["target_test"]["default_argmax"]
        weak = [c for c in R["meta"]["attack_classes"] if _m(t["recall"][c]) < 0.2]
        strong = [c for c in R["meta"]["attack_classes"] if _m(t["recall"][c]) >= 0.8]
        F.append({
            "gap": "G1", "title": f"{short_dir(k)}: macro-F1 falls from {_m(s['macro_f1']):.2f} to {_m(t['macro_f1']):.2f}",
            "text": (f"Same-dataset macro-F1 is {pm(s['macro_f1'])} but the unseen dataset gives {pm(t['macro_f1'])}. "
                     f"Benign false-alarm rate moves from {pm(s['fpr'], 2, True)} % to {pm(t['fpr'], 2, True)} % "
                     f"({_m(s['alerts_per_10k_benign']):.0f} → {_m(t['alerts_per_10k_benign']):.0f} alerts per 10,000 benign flows) while the attack detection rate is {pm(t['detection_rate'], 1, True)} %. "
                     + (f"Classes that collapse (recall < 0.2): {', '.join(weak)}. " if weak else "No class collapses below 0.2 recall. ")
                     + (f"Classes that transfer well (recall ≥ 0.8): {', '.join(strong)}." if strong else "No class transfers with recall ≥ 0.8.")),
        })
    d = T[px]["default_argmax"]
    F.append({"gap": "G1", "title": "Transfer is asymmetric",
              "text": (f"Macro-F1 drop is {pm(d['drop_fwd'])} for UNSW→CIC and {pm(d['drop_rev'])} for CIC→UNSW; the two target scores differ by {pm(d['asymmetry'])}. "
                       "Which dataset is used as source matters, as reported by Layeghy & Portmann (2022).")})

    # ---- G2: thresholds ----
    for k in (FWD, REV):
        t = E[k][px]["eval"]["target_test"]
        rows = {p: t[p] for p in R["meta"]["policy_order"]}
        d_a, g_a, p_a, f_a = (_m(rows["default_argmax"]["alerts_per_10k_benign"]), _m(rows["global_threshold"]["alerts_per_10k_benign"]),
                              _m(rows["per_class_cost"]["alerts_per_10k_benign"]), _m(rows["per_class_fpr_cap"]["alerts_per_10k_benign"]))
        d_f, g_f, p_f, f_f = (_m(rows["default_argmax"]["macro_f1"]), _m(rows["global_threshold"]["macro_f1"]),
                              _m(rows["per_class_cost"]["macro_f1"]), _m(rows["per_class_fpr_cap"]["macro_f1"]))
        if abs(p_a - g_a) <= 0.03 * max(g_a, 1.0):
            better = "about the same benign alert volume as"
        elif p_a < g_a:
            better = "lower benign alert volume than"
        else:
            better = "higher benign alert volume than"
        vs_default = "more" if p_a > d_a * 1.03 else ("fewer" if p_a < d_a * 0.97 else "about the same number of")
        F.append({
            "gap": "G2", "title": f"{short_dir(k)}: alerts per 10k benign — default {d_a:.0f}, global {g_a:.0f}, per-class {p_a:.0f}, FPR-cap {f_a:.0f}",
            "text": (f"On the unseen dataset, per-class cost thresholds give {better} a single global threshold ({p_a:.0f} vs {g_a:.0f} per 10,000 benign flows), "
                     f"with macro-F1 {p_f:.3f} vs {g_f:.3f}. Against the default arg-max rule ({d_f:.3f}, {d_a:.0f} alerts) they raise macro-F1 but produce {vs_default} false alerts, "
                     f"because the cost matrix charges a missed attack more than a false alarm. "
                     f"The FPR-cap policy is the false-positive-reduction setting: {f_a:.0f} alerts per 10,000 benign flows at macro-F1 {f_f:.3f}. "
                     "All thresholds were learned on source validation data only.")
        })
    k = FWD
    r_raw = _m(E[k][px]["eval"]["target_test"]["default_argmax"]["ece"])
    r_cal = _m(E[k][px]["eval"]["target_test"]["calibrated_argmax"]["ece"])
    k2 = REV
    r_raw2 = _m(E[k2][px]["eval"]["target_test"]["default_argmax"]["ece"])
    r_cal2 = _m(E[k2][px]["eval"]["target_test"]["calibrated_argmax"]["ece"])
    F.append({"gap": "G2", "title": "Source-fitted calibration under shift",
              "text": (f"Top-label ECE on the unseen dataset: UNSW→CIC {r_raw:.3f} → {r_cal:.3f} after isotonic calibration; CIC→UNSW {r_raw2:.3f} → {r_cal2:.3f}. "
                       "Calibration is fitted on source validation data, so a change here measures how well it survives the shift.")})

    # ---- G3: explanations ----
    for k in (FWD, REV):
        x = X[k]
        tk = int(round(_m(x["stability_topk"]["top_k"])))
        n_err = int(round(_m(x["n_error_probe"])))
        F.append({
            "gap": "G3", "title": f"{short_dir(k)}: top-{tk} SHAP overlap {_m(x['stability_topk']['top_k_overlap']):.1f} / {tk}",
            "text": (f"Top-{tk} feature overlap between source and target is {pm(x['stability_topk']['top_k_overlap'], 1)} of {tk}; "
                     f"Spearman rank correlation of importance is {pm(x['stability_topk']['spearman_rho'], 2)}. "
                     f"Across features, rank shift correlates with benign distribution shift (KS) at ρ = {pm(x['rho_rankshift_vs_ks'], 2)} "
                     f"and with attribution enrichment on errors at ρ = {pm(x['rho_rankshift_vs_enrichment'], 2)} (up to {n_err} errors probed per seed; "
                     "13 features, so treat these as indicative).")
        })

    # ---- G4: replay ----
    for k in (FWD, REV):
        r = RP[k]
        F.append({
            "gap": "G4", "title": f"{short_dir(k)} replay: p50 {r['latency_per_flow_proposed']['total']['p50_ms']:.2f} ms/flow, {r['end_to_end']['packets_per_s']:.0f} pkts/s",
            "text": (f"{r['flows_replayed']} flows / {r['packets_replayed']} packets replayed. Single-flow latency (feature build + preprocessing + model + calibration + decision): "
                     f"p50 {r['latency_per_flow_proposed']['total']['p50_ms']:.2f} ms, p95 {r['latency_per_flow_proposed']['total']['p95_ms']:.2f} ms, p99 {r['latency_per_flow_proposed']['total']['p99_ms']:.2f} ms. "
                     f"Flow export {r['exporter']['packets_per_s']:.0f} pkts/s; micro-batched scoring {r['batch']['flows_per_s']:.0f} flows/s; end-to-end {r['end_to_end']['packets_per_s']:.0f} pkts/s. "
                     f"Alert volume: {r['alerts']['total_alerts']} alerts, {r['alerts']['alerts_per_10k_benign']:.0f} per 10,000 benign flows. "
                     f"Exporter feature round-trip max relative error: {r['feature_roundtrip_max_rel_error']:.2e}. Measured on {r['environment']['cpu_count']} CPUs; re-measure on your own machine for the report.")
        })

    # ---- G5 stretch ----
    for k in (FWD, REV):
        s = ST[k]
        keys = sorted(s)
        ms = ", ".join(f"{float(q)*100:.0f}% → {pm(s[q]['macro_f1'])}" for q in keys)
        F.append({"gap": "G5", "title": f"{short_dir(k)}: fine-tuning macro-F1 ({ms})",
                  "text": "Stretch experiment, kept separate from zero-label transfer. Labelled target flows come only from the target train partition; evaluation is on the target test partition. "
                          + "Macro-F1 by labelled fraction: " + ms + "."})

    L = [
        "All numbers come from SIMULATED stand-ins for UNSW-NB15 and CIC-IDS2017. They validate the pipeline and measurement method, not real-world detection performance. Re-run on the real CSVs before reporting any result.",
        "The distribution shift between the two datasets was designed into the generator (signature shifts per attack class). Which classes collapse and by how much is therefore a property of the simulation, not a finding about the real datasets.",
        "The gradient-boosted model is scikit-learn's HistGradientBoosting (XGBoost was not installable here); the MLP is scikit-learn's MLPClassifier (PyTorch was not installable). The XGBoost/PyTorch code paths are included but were not executed here.",
        "SHAP values are computed with a permutation-sampling Shapley estimator (the quantity SHAP approximates), not the `shap` package. With the package installed, exact TreeSHAP can be used for XGBoost.",
        "Replay uses packets synthesised from held-out flows and a minimal Python flow exporter, not real PCAPs or CICFlowMeter. Latency/throughput are real wall-clock measurements on the sandbox machine (CPU count in the environment block) and will differ on your laptop.",
        "Only flow features shared by both datasets are used (13). Port number is bucketed, TTL, TCP state/flags and inter-arrival features are excluded because they are not comparable across the two exporters.",
        "Five seeds vary splits and model initialisation; the two datasets themselves are fixed, so the reported standard deviation does not include dataset-sampling variability.",
        "Class mapping merges CIC web attacks into 'Exploit' and UNSW 'Backdoor' into 'Botnet'. These are approximations and should be reviewed against the real data.",
    ]
    return F, L


def write_markdown_report(R: dict, path) -> None:
    E, T = R["experiments"], R["transfer"]
    px = "xgboost"
    M = R["meta"]
    lines = [f"# {M['title']}", "", f"_Authors: {', '.join(M['authors'])} — {M['institution']}_", "",
             f"> **{M['data_notice']}**", "",
             f"Generated {M['generated_at']} · seeds {M['seeds']} · runtime {M['runtime_s']} s", ""]
    lines += ["## Findings", ""]
    for f in R["findings"]:
        lines += [f"**[{f['gap']}] {f['title']}**  ", f["text"], ""]
    lines += ["## Cross-dataset macro-F1 (default arg-max, mean ± std over seeds)", "",
              "| Model | Direction | Same-dataset | Cross-dataset |", "|---|---|---|---|"]
    for m, name in M["models"].items():
        for k in (FWD, REV):
            ev = E[k][m]["eval"]
            lines.append(f"| {name} | {short_dir(k)} | {pm(ev['source_test']['default_argmax']['macro_f1'])} | {pm(ev['target_test']['default_argmax']['macro_f1'])} |")
    lines += ["", "## Alert burden on the unseen dataset (proposed model)", "",
              "| Direction | Policy | Macro-F1 | FPR % | Alerts / 10k benign | Detection rate % | Cost / flow |", "|---|---|---|---|---|---|---|"]
    for k in (FWD, REV):
        for p in M["policy_order"]:
            r = E[k][px]["eval"]["target_test"][p]
            lines.append(f"| {short_dir(k)} | {M['policies'][p]} | {pm(r['macro_f1'])} | {pm(r['fpr'], 2, True)} | {pm(r['alerts_per_10k_benign'], 0)} | {pm(r['detection_rate'], 1, True)} | {pm(r['cost_per_flow'], 3)} |")
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in R["limitations"]]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
