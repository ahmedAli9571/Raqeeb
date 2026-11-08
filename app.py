"""
Raqeeb dashboard.  Run:  streamlit run app.py   (or double-click run.bat on Windows)
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from raqeeb.data import load_all
from raqeeb.features import add_cause_features, site_context
from raqeeb.models import CLASSES, load, tomorrow_risk, train, triage
from raqeeb.notes import note

st.set_page_config(page_title="Raqeeb | رقيب", page_icon="💧", layout="wide")

PRIORITY_COLOURS = {"1 Urgent (E. coli)": "#c62828", "2 High: likely real": "#ef6c00",
                    "3 Medium": "#f9a825", "3 Check lab first": "#6a1b9a", "4 Resample": "#1565c0"}
CAUSE_COLOURS = {"real_event": "#d84315", "sampling_error": "#1565c0", "analytical_error": "#6a1b9a"}


# ------------------------------------------------------------------ loading
@st.cache_data(show_spinner="Loading data…")
def get_data():
    return load_all()


@st.cache_resource(show_spinner="Loading models…")
def get_models():
    if not os.path.exists(os.path.join(config.MODEL_DIR, "meta.json")):
        train(get_data())
    return load()


def refresh():
    get_data.clear()
    get_models.clear()


t = get_data()
expect, cause, meta = get_models()
scored = triage(t, expect, cause, meta)
last_day = t["results"]["date"].max()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("💧 Raqeeb | رقيب")
st.sidebar.caption("AI triage for abnormal water quality results")
page = st.sidebar.radio("Page", ["Inbox", "Map", "Site history", "Tomorrow's sampling", "Model report"])
st.sidebar.divider()
st.sidebar.caption(f"Data: {t['results']['date'].min():%d %b %Y} to {last_day:%d %b %Y} · "
                   f"{len(t['results']):,} results · {t['sites'].shape[0]} sites")
if st.sidebar.button("Retrain models", help="Re-reads the data folder and trains both models again"):
    with st.spinner("Training…"):
        refresh()
        train(load_all())
    st.sidebar.success("Retrained.")
    st.rerun()
st.sidebar.caption("Practice data is synthetic. Limits in config.py are placeholders until set to "
                   "your lab's Omani standard edition.")


def window_picker(key, default_days=31):
    c1, c2 = st.columns(2)
    start = c1.date_input("From", (last_day - pd.Timedelta(days=default_days)).date(), key=key + "s")
    end = c2.date_input("To", last_day.date(), key=key + "e")
    return pd.Timestamp(start), pd.Timestamp(end)


# ------------------------------------------------------------------ pages
if page == "Inbox":
    st.title("Inbox: abnormal results, most urgent first")
    st.caption("Default dates = Oct–Dec, the months the model never saw in training.")
    s, e = window_picker("inbox", default_days=91)
    view = scored[(scored["date"] >= s) & (scored["date"] <= e)]
    if view.empty:
        st.info("No abnormal results in this window.")
        st.stop()

    k = st.columns(4)
    k[0].metric("Abnormal results", len(view))
    k[1].metric("Urgent or likely real", int(view["priority"].str.startswith(("1", "2")).sum()))
    k[2].metric("Check lab first", int((view["pred_cause"] == "analytical_error").sum()))
    k[3].metric("Resample only", int((view["pred_cause"] == "sampling_error").sum()))

    table = view.assign(
        likely_cause=view["pred_cause"].map(lambda c: config.CAUSE_LABELS[c]["en"]),
        confidence=(view["confidence"] * 100).round().astype(int).astype(str) + "%",
        date_=view["date"].dt.strftime("%Y-%m-%d"),
    )[["priority", "date_", "sample_id", "site_id", "likely_cause", "confidence", "fail_reasons"]]
    table.columns = ["Priority", "Date", "Sample", "Site", "Likely cause", "Confidence", "Failed on"]
    st.dataframe(table, hide_index=True, height=300,
                 column_config={"Priority": st.column_config.TextColumn(width="medium")})

    st.subheader("Investigate a result")
    pick = st.selectbox("Sample", view["sample_id"],
                        format_func=lambda sid: f"{sid} · {view.set_index('sample_id').loc[sid, 'site_id']} · "
                                                f"{view.set_index('sample_id').loc[sid, 'priority']}")
    x = view.set_index("sample_id").loc[pick].copy()
    x["sample_id"] = pick
    c = x["pred_cause"]

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"<div style='padding:12px;border-left:6px solid {CAUSE_COLOURS[c]};background:rgba(0,0,0,0.03)'>"
                    f"<b>Likely cause: {config.CAUSE_LABELS[c]['en']}</b> ({x['confidence']:.0%} confidence)<br>"
                    f"<span style='font-size:0.9em'>{x['priority']}</span></div>", unsafe_allow_html=True)
        st.markdown("**Why Raqeeb thinks so**")
        for reason in x["reasons"] or ["no single strong signal"]:
            st.markdown(f"- {reason}")
        st.markdown(f"**Recommended action:** {config.ACTIONS[c]['en']}")
        st.caption("Raqeeb orders the work. A failed result stays open until confirmed under the lab SOP.")
    with right:
        probs = pd.DataFrame({"cause": [config.CAUSE_LABELS[k]["en"] for k in CLASSES],
                              "p": [x.get(f"p_{k}", 0) for k in CLASSES]})
        fig = px.bar(probs, x="p", y="cause", orientation="h", range_x=[0, 1], height=180,
                     color="cause", color_discrete_sequence=[CAUSE_COLOURS[k] for k in CLASSES])
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="probability", yaxis_title="")
        st.plotly_chart(fig, config={"displayModeBar": False})
        ev = pd.DataFrame({
            "Evidence": ["Free chlorine (mg/L)", "Expected chlorine (mg/L)", "Turbidity (NTU)", "Total coliform /100 mL",
                         "E. coli /100 mL", "Water temp (°C)", "Water age (h)", "Repairs in DMA (last days)",
                         "Quality complaints (last days)", "Batch QC"],
            "Value": [f"{x['free_cl_mgL']:.2f}", f"{x['cl_expected']:.2f}", f"{x['turbidity_NTU']:.2f}",
                      int(x["total_coliform"]), int(x["ecoli"]), f"{x['water_temp_C']:.1f}", f"{x['water_age_h']:.0f}",
                      int(x["repairs_recent"]), int(x["quality_complaints_recent"]),
                      "OUT of control" if x["qc_out_of_control"] else "in control"]})
        st.dataframe(ev, hide_index=True)

    en, ar = note(x)
    t1, t2 = st.tabs(["Investigation note (English)", "مذكرة التحقيق (عربي)"])
    t1.code(en, language=None, wrap_lines=True)
    t2.markdown(f"<div dir='rtl' style='text-align:right;white-space:pre-wrap;line-height:1.9'>{ar}</div>",
                unsafe_allow_html=True)

elif page == "Map":
    st.title("Map: where the problems are")
    s, e = window_picker("map")
    w = scored[(scored["date"] >= s) & (scored["date"] <= e)]
    near = t["results"][(t["results"]["date"] >= s) & (t["results"]["date"] <= e) & t["results"]["near_miss"]]
    sites = t["sites"].copy()
    status = {}
    for sid in sites["site_id"]:
        sw = w[w["site_id"] == sid]
        if len(sw):
            status[sid] = sw["priority"].min()
        elif sid in set(near["site_id"]):
            status[sid] = "5 Near miss"
        else:
            status[sid] = "6 OK"
    sites["status"] = sites["site_id"].map(status)
    colours = {**PRIORITY_COLOURS, "5 Near miss": "#9e9d24", "6 OK": "#2e7d32"}
    fig = px.scatter_map(sites.sort_values("status", ascending=False), lat="lat", lon="lon", color="status",
                         hover_name="site_id", hover_data={"governorate": True, "zone": True, "lat": False, "lon": False},
                         color_discrete_map=colours, zoom=5.6, height=600, map_style="open-street-map",
                         center={"lat": 23.5, "lon": 57.6})
    fig.update_traces(marker={"size": 13})
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Worst status in window")
    st.plotly_chart(fig)
    st.caption("Positions are approximate (synthetic). The map background needs internet; the points work offline.")

elif page == "Site history":
    st.title("Site history: measured vs expected chlorine")
    sid = st.selectbox("Site", t["sites"]["site_id"],
                       format_func=lambda s: f"{s} · {t['sites'].set_index('site_id').loc[s, 'governorate']} · "
                                             f"{t['sites'].set_index('site_id').loc[s, 'zone']}")
    r = site_context(t, meta["site_medians"])
    r = add_cause_features(r, t, expect, meta["resid_sd"])
    h = r[r["site_id"] == sid].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["date"], y=h["cl_expected"], name="Expected (model)", line=dict(color="#90a4ae", dash="dot")))
    fig.add_trace(go.Scatter(x=h["date"], y=h["free_cl_mgL"], name="Measured", mode="lines+markers",
                             line=dict(color="#1565c0"), marker=dict(size=5)))
    bad = h[h["is_abnormal"]]
    fig.add_trace(go.Scatter(x=bad["date"], y=bad["free_cl_mgL"], name="Abnormal result", mode="markers",
                             marker=dict(color="#c62828", size=11, symbol="x"), text=bad["fail_reasons"]))
    fig.add_hline(y=config.LIMITS["free_chlorine_min_mgL"], line_dash="dash", line_color="#c62828",
                  annotation_text="minimum", annotation_position="bottom right")
    fig.update_layout(height=450, yaxis_title="Free chlorine (mg/L)", margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig)
    d = t["daily"][t["daily"]["site_id"] == sid]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=d["date"], y=d["water_temp_C"], name="Water temp (°C)", line=dict(color="#ef6c00")))
    fig2.add_trace(go.Scatter(x=d["date"], y=d["water_age_h"], name="Water age (h)", line=dict(color="#6d4c41"), yaxis="y2"))
    fig2.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.15),
                       yaxis=dict(title="°C"), yaxis2=dict(title="hours", overlaying="y", side="right"))
    st.plotly_chart(fig2)

elif page == "Tomorrow's sampling":
    st.title("Tomorrow's sampling: sites most likely to fall below the chlorine minimum")
    max_asof = (t["daily"]["date"].max() - pd.Timedelta(days=1)).date()
    as_of = st.date_input("Today is", min(pd.Timestamp("2025-07-15").date(), max_asof), max_value=max_asof)
    risk = tomorrow_risk(t, expect, meta, as_of)
    top = risk.head(15)
    fig = px.bar(top[::-1], x="p_below_limit", y="site_id", orientation="h", height=450,
                 color="zone", color_discrete_map={"rural": "#ef6c00", "urban": "#1565c0"},
                 hover_data={"governorate": True, "pred_cl_mgL": ":.2f"})
    fig.update_layout(xaxis_title="Chance chlorine is below the minimum tomorrow", xaxis_tickformat=".0%",
                      yaxis_title="", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)
    out = top[["site_id", "governorate", "zone", "water_temp_C", "water_age_h", "pred_cl_mgL", "p_below_limit"]].copy()
    out["p_below_limit"] = (out["p_below_limit"] * 100).round(0).astype(int).astype(str) + "%"
    out.columns = ["Site", "Governorate", "Zone", "Water temp (°C)", "Water age (h)", "Predicted Cl (mg/L)", "Chance below min"]
    st.dataframe(out.round(2), hide_index=True)
    st.caption("Uses tomorrow's forecast water temperature and SCADA water age. Try a July date, then a January date.")

else:
    st.title("Model report: how good is Raqeeb on data it has not seen?")
    te = meta.get("test", {})
    rb = meta.get("robustness_seed7")
    k = st.columns(3)
    k[0].metric("Chlorine model typical error", f"± {meta['resid_sd']:.2f} mg/L")
    if te:
        k[1].metric(f"Cause accuracy, test months Oct–Dec (n={te['n']})", f"{te['accuracy']:.0%}")
    if rb:
        k[2].metric(f"Cause accuracy, brand-new dataset (n={rb['n']})", f"{rb['accuracy']:.0%}")
    if te:
        rows = [{"Cause": config.CAUSE_LABELS[c]["en"], "Precision": f"{te['report'][c]['precision']:.0%}",
                 "Recall": f"{te['report'][c]['recall']:.0%}", "Cases": int(te["report"][c]["support"])}
                for c in te["labels"]]
        st.subheader("Per cause (test months)")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        cm = pd.DataFrame(te["confusion"], index=[config.CAUSE_LABELS[c]["en"] for c in te["labels"]],
                          columns=[config.CAUSE_LABELS[c]["en"] for c in te["labels"]])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", height=330,
                        labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig)
    imp = pd.Series(meta["importance"]).sort_values()
    imp = imp[imp > 0]
    from raqeeb.models import READABLE
    fig = px.bar(x=imp.values, y=[READABLE.get(i, i) for i in imp.index], orientation="h", height=420)
    fig.update_layout(xaxis_title="Importance (total gain)", yaxis_title="", margin=dict(l=0, r=0, t=10, b=0))
    st.subheader("What evidence the classifier relies on")
    st.plotly_chart(fig)
    st.warning("Honest caveat: these scores come from synthetic data made by one generator. The 'brand-new dataset' "
               "test uses different random sites and events but the same rules, so real-world accuracy will be lower. "
               "On real data, the classifier learns from every investigation the chemist closes.")
