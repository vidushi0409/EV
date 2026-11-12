import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.evaluate_models import COMPARISON_FILE, EVAL_DIR, IMPORTANCE_FILE, PREDICTIONS_FILE, read_best_model
from src.plot_style import MODEL_COLORS
from src.train_models import MODEL_NAMES
from src.ui import csv_download, get_service, kpi_row, setup_page

setup_page("ℹ️ About the Model")
best = read_best_model()
if best is None or not COMPARISON_FILE.exists():
    st.error("Model results not found. Run `python train.py` in the project folder.")
    st.stop()
svc = get_service()

FRIENDLY = {
    "volume_lag24": "Same hour yesterday", "volume_lag168": "Same hour last week",
    "volume_lag1": "Previous hour", "volume_roll24_mean": "Last 24 h average",
    "duration_lag1": "Charging time (prev. hour)", "occupancy_lag1": "Chargers in use (prev. hour)",
    "neighbor_volume_lag1": "Neighbouring zones (prev. hour)", "hour": "Hour of day",
    "hour_sin": "Hour of day", "hour_cos": "Hour of day", "day": "Day of month", "dayofweek": "Day of week",
    "dow_sin": "Day of week", "dow_cos": "Day of week", "is_weekend": "Weekend", "is_holiday": "Holiday",
    "e_price": "Electricity price", "s_price": "Service price", "num_piles": "Number of chargers",
    "fast_share": "Fast-charger share", "avg_pile_kw": "Charger power", "capacity_kw": "Total charger power",
    "area": "Zone area", "longitude": "Location", "latitude": "Location",
}
FMT = {"MAE": "{:,.1f}", "MSE": "{:,.0f}", "RMSE": "{:,.1f}", "R²": "{:.3f}"}


def metric_table(df):
    t = df[["model", "MAE", "MSE", "RMSE", "R2"]].rename(columns={"model": "Model", "R2": "R²"})
    return (t.style.format(FMT).highlight_min(subset=["MAE", "MSE", "RMSE"], color="#d6f0e4")
            .highlight_max(subset=["R²"], color="#d6f0e4"))


name = best["best_model"]
m = best["test_metrics"]
c1, c2 = st.columns(2)
c1.markdown(f'<div class="card"><h4>📊 Dataset period · {svc.dataset_best}</h4>'
            '<p>Next-hour models using recent demand. Walk-forward: each month predicted by models '
            'trained on earlier months only.</p></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="card"><h4>🔭 Future dates · {svc.future_best}</h4>'
            '<p>Pattern models using calendar, prices and each zone\'s typical demand, trained on all '
            'six months.</p></div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Model comparison", "Month by month", "Future-date models", "Demand drivers"])

with tab1:
    kpi_row([(f"{name} · MAE", f"{m['MAE']:,.1f} kWh"), ("RMSE", f"{m['RMSE']:,.1f} kWh"),
             ("MSE", f"{m['MSE']:,.0f}"), ("R²", f"{m['R2']:.3f}")])
    comp = pd.read_csv(COMPARISON_FILE)
    st.dataframe(metric_table(comp), hide_index=True, use_container_width=True)
    metric = st.radio("Metric", ["RMSE", "MAE", "MSE", "R2"], horizontal=True, label_visibility="collapsed",
                      format_func=lambda x: "R²" if x == "R2" else x)
    fig = go.Figure(go.Bar(x=comp.model, y=comp[metric], marker_color=[MODEL_COLORS[x] for x in comp.model],
                           text=[f"{v:,.3f}" if metric == "R2" else f"{v:,.1f}" for v in comp[metric]],
                           textposition="outside"))
    fig.update_layout(height=320, plot_bgcolor="white", margin=dict(l=10, r=10, t=20, b=10),
                      yaxis=dict(gridcolor="#e6e5e1"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Train Sep–Dec 2022 · select on Jan 2023 · test on Feb 2023. Naive rows = reference forecasts.")

with tab2:
    monthly = svc.monthly
    per_month = monthly[monthly.month != "Oct-Feb (all)"]
    wf_metric = st.radio("Metric", ["RMSE", "MAE", "R2"], horizontal=True, key="wf",
                         label_visibility="collapsed", format_func=lambda x: "R²" if x == "R2" else x)
    fig = go.Figure()
    for mdl in MODEL_NAMES:
        d = per_month[per_month.model == mdl]
        fig.add_trace(go.Scatter(x=d.month, y=d[wf_metric], name=mdl, mode="lines+markers",
                                 line=dict(color=MODEL_COLORS[mdl], width=2), marker=dict(size=8)))
    fig.update_layout(height=320, plot_bgcolor="white", hovermode="x unified",
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.12),
                      yaxis=dict(gridcolor="#e6e5e1"))
    st.plotly_chart(fig, use_container_width=True)
    tbl = monthly.pivot(index="month", columns="model", values=wf_metric)[MODEL_NAMES]
    tbl = tbl.reindex(config.WALK_FORWARD_MONTHS + ["Oct-Feb (all)"])
    st.dataframe(tbl.style.format("{:.3f}" if wf_metric == "R2" else "{:,.1f}"), use_container_width=True)

with tab3:
    ev = svc.profile.evaluation.reset_index()
    st.dataframe(metric_table(ev), hide_index=True, use_container_width=True)
    st.caption("Tested on Feb 2023 after training on earlier months. Reference = the zone's typical "
               "demand for that hour, with no learning.")

with tab4:
    imp = pd.read_csv(IMPORTANCE_FILE)
    if name in imp.model.unique():
        d = imp[imp.model == name].copy()
        d["Factor"] = d.feature.map(FRIENDLY)
        d = d.groupby("Factor", as_index=False).importance.sum().nlargest(8, "importance")[::-1]
        fig = go.Figure(go.Bar(x=d.importance * 100, y=d.Factor, orientation="h",
                               marker_color=MODEL_COLORS.get(name, "#2a78d6"),
                               hovertemplate="%{x:.1f}%<extra></extra>"))
        fig.update_layout(height=340, plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(title="Share of importance (%)", gridcolor="#e6e5e1"))
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Evaluation charts"):
        for fname in ("actual_vs_predicted.png", "residuals.png", "feature_importance.png"):
            if (EVAL_DIR / fname).exists():
                st.image(str(EVAL_DIR / fname), use_container_width=True)

st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
with d1:
    csv_download(pd.read_csv(COMPARISON_FILE), "model_comparison.csv", "⬇ Model comparison")
    csv_download(svc.monthly, "monthly_accuracy.csv", "⬇ Monthly accuracy")
with d2:
    if PREDICTIONS_FILE.exists():
        preds = pd.read_parquet(PREDICTIONS_FILE, columns=["time", "zone", "actual"] + MODEL_NAMES)
        csv_download(preds, "test_predictions.csv", "⬇ Test predictions")
with d3:
    for fname, mime in [("Project_Report.pdf", "application/pdf"),
                        ("Project_Report.docx",
                         "application/vnd.openxmlformats-officedocument.wordprocessingml.document")]:
        path = config.REPORTS_DIR / fname
        if path.exists():
            st.download_button(f"⬇ Report ({path.suffix[1:].upper()})", path.read_bytes(),
                               file_name=fname, mime=mime)
