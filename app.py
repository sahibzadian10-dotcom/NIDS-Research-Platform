"""Optional Streamlit front end for the same results the HTML dashboard shows.

    pip install streamlit
    streamlit run dashboard/app.py

NOTE: Streamlit was not installable in the sandbox where this project was generated, so this file has been
syntax-checked but NOT run. The self-contained dashboard/index.html is the tested interface.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results" / "results.json"
R = json.loads(RESULTS.read_text(encoding="utf-8"))
M = R["meta"]


def ms(x, digits=3):
    return f"{x['mean']:.{digits}f} ± {x['std']:.{digits}f}"


st.set_page_config(page_title="NIDS Research Platform", layout="wide")
st.title("NIDS Research Platform")
st.caption(f"{M['title']} · {', '.join(M['authors'])} · {M['institution']}")
if M["data_mode"] != "real":
    st.warning(M["data_notice"])

direction = st.sidebar.radio("Direction (source → target)", list(R["experiments"].keys()))
model = st.sidebar.selectbox("Model", list(M["models"].keys()), index=1, format_func=lambda k: M["models"][k])
exp = R["experiments"][direction][model]

t_over, t_xfer, t_alert, t_expl, t_dep, t_find = st.tabs(["Overview", "Transfer", "Alert control", "Explanations", "Deployment", "Findings"])

with t_over:
    c1, c2, c3 = st.columns(3)
    same = exp["eval"]["source_test"]["default_argmax"]
    cross = exp["eval"]["target_test"]["default_argmax"]
    c1.metric("Macro-F1, same dataset", ms(same["macro_f1"]))
    c2.metric("Macro-F1, unseen dataset", ms(cross["macro_f1"]))
    c3.metric("Alerts / 10k benign, unseen", f"{cross['alerts_per_10k_benign']['mean']:.0f}")
    st.subheader("Class distribution after cleaning")
    st.dataframe(pd.DataFrame({d: v["class_counts"] for d, v in R["datasets"].items()}))

with t_xfer:
    rows = []
    for m, label in M["models"].items():
        for key, ex in R["experiments"].items():
            rows.append({"Model": label, "Direction": key,
                         "Same-dataset macro-F1": ms(ex[m]["eval"]["source_test"]["default_argmax"]["macro_f1"]),
                         "Cross-dataset macro-F1": ms(ex[m]["eval"]["target_test"]["default_argmax"]["macro_f1"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    rec = {c: exp["eval"]["target_test"]["default_argmax"]["recall"][c]["mean"] for c in M["classes"]}
    st.subheader("Per-class recall on the unseen dataset")
    st.bar_chart(pd.Series(rec))

with t_alert:
    rows = []
    for pol in M["policy_order"]:
        for where in ("source_test", "target_test"):
            e = exp["eval"][where][pol]
            rows.append({"Rule": M["policies"][pol], "Evaluated on": "source" if where == "source_test" else "unseen",
                         "Alerts / 10k benign": round(e["alerts_per_10k_benign"]["mean"]),
                         "Detection rate %": round(100 * e["detection_rate"]["mean"], 1),
                         "Macro-F1": ms(e["macro_f1"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Thresholds are learned on source-validation data only and applied unchanged to the unseen dataset.")

with t_expl:
    ex = R["explain"][direction]
    df = pd.DataFrame({"source": ex["importance_source"], "target": ex["importance_target"]})
    df = df.apply(lambda col: col.map(lambda v: v["mean"] if isinstance(v, dict) else v))
    st.bar_chart(df)
    st.write("Top-5 overlap (of 5):", ms(ex["stability_topk"]["top_k_overlap"], 2), "· Spearman ρ:", ms(ex["stability_topk"]["spearman_rho"], 2))

with t_dep:
    rp = R["replay"][direction]
    st.metric("Single-flow latency p50 (ms)", f"{rp['latency_per_flow_proposed']['total']['p50_ms']:.2f}")
    st.json({k: rp[k] for k in ("flows_replayed", "packets_replayed", "end_to_end", "batch", "alerts")})
    st.caption(f"Measured on: {json.dumps(rp['environment'])}. Re-measure on your own machine before reporting.")

with t_find:
    for f in R["findings"]:
        st.markdown(f"**{f.get('title', '')}**  \n{f.get('body', f.get('text', ''))}" if isinstance(f, dict) else f)
    st.subheader("Limitations")
    for lim in R["limitations"]:
        st.markdown(f"- {lim}")
