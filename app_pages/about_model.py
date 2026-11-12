import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.evaluate_models import COMPARISON_FILE, EVAL_DIR, IMPORTANCE_FILE, PREDICTIONS_FILE, read_best_model
from src.plot_style import MODEL_COLORS
from src.train_models import MODEL_NAMES
from src.ui import csv_download, get_service, setup_page

setup_page("ℹ️ About the Model")

best = read_best_model()
if best is None or not COMPARISON_FILE.exists():
    st.error("Model results not found. Run `python train.py` in the project folder.")
    st.stop()
svc = get_service()

FRIENDLY = {
    "volume_lag24": "Demand at the same hour yesterday", "volume_lag168": "Demand at the same hour last week",
    "volume_lag1": "Demand in the previous hour", "volume_roll24_mean": "Average demand over the last 24 hours",
    "duration_lag1": "Charging time in the previous hour", "occupancy_lag1": "Chargers in use in the previous hour",
    "neighbor_volume_lag1": "Neighbouring zones' demand in the previous hour",
    "hour": "Hour of the day", "hour_sin": "Hour of the day (cyclical)", "hour_cos": "Hour of the day (cyclical)",
    "day": "Day of the month", "dayofweek": "Day of the week", "dow_sin": "Day of the week (cyclical)",
    "dow_cos": "Day of the week (cyclical)", "is_weekend": "Weekend", "is_holiday": "Public holiday",
    "e_price": "Electricity price", "s_price": "Service price", "num_piles": "Number of chargers",
    "fast_share": "Share of fast chargers", "avg_pile_kw": "Average charger power",
    "capacity_kw": "Total charger power", "area": "Zone area",
    "longitude": "Location (longitude)", "latitude": "Location (latitude)",
}


def bar(df, metric, height=320):
    fig = go.Figure(go.Bar(x=df.model, y=df[metric], marker_color=[MODEL_COLORS[x] for x in df.model],
                           text=[f"{v:,.3f}" if metric == "R2" else f"{v:,.1f}" for v in df[metric]],
                           textposition="outside"))
    fig.update_layout(height=height, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(title="higher is better" if metric == "R2" else "lower is better",
                                 gridcolor="#e6e5e1"))
    return fig


name = best["best_model"]
st.markdown(f"""
The app uses **two kinds of models**, each trained with **Linear Regression, Random Forest and
LightGBM**:

| | Dataset period (Oct 2022 – Feb 2023) | Future dates |
|---|---|---|
| Inputs | recent real demand (last hour, yesterday, last week) + calendar, prices, zone | calendar, typical prices, zone and its usual demand pattern |
| Trained on | all months **before** the predicted month | all six months |
| Headline model | **{svc.dataset_best}** | **{svc.future_best}** |
| Checked against actual demand | ✅ yes | measured on February 2023 only |
""")

# ---------------------------------------------------------------- main comparison
st.subheader("1 · Main model comparison (next-hour models)")
st.caption("Trained on Sep–Dec 2022, model chosen on January 2023, tested on February 2023 "
           "(184,800 zone-hours never used for training).")
m = best["test_metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{name} · MAE", f"{m['MAE']:,.1f} kWh")
c2.metric("RMSE", f"{m['RMSE']:,.1f} kWh")
c3.metric("MSE", f"{m['MSE']:,.0f}")
c4.metric("R²", f"{m['R2']:.3f}")
comp = pd.read_csv(COMPARISON_FILE)
shown = comp.assign(Type=comp.model.map(lambda x: "Reference (no learning)" if x.startswith("Naive")
                                        else "Machine learning"))
shown = shown[["model", "Type", "MAE", "MSE", "RMSE", "R2"]].rename(columns={"model": "Model", "R2": "R²"})
st.dataframe(shown.style.format({"MAE": "{:,.2f}", "MSE": "{:,.0f}", "RMSE": "{:,.2f}", "R²": "{:.4f}"})
             .highlight_min(subset=["MAE", "MSE", "RMSE"], color="#d6f0e4")
             .highlight_max(subset=["R²"], color="#d6f0e4"), hide_index=True, use_container_width=True)
metric = st.radio("Compare by", ["RMSE", "MAE", "MSE", "R2"], horizontal=True,
                  format_func=lambda x: "R²" if x == "R2" else x)
st.plotly_chart(bar(comp, metric), use_container_width=True)
val = best["validation_rmse"]
st.markdown(f"**Why {name}?** Lowest error on the January validation data "
            f"(RMSE: {', '.join(f'{k} {v:,.1f}' for k, v in val.items())}), "
            f"{'confirmed' if best['selection_confirmed_on_test'] else 'not confirmed'} on February. "
            "All models clearly beat the simple reference forecasts.")

# ---------------------------------------------------------------- walk-forward
st.subheader("2 · Month-by-month accuracy (walk-forward)")
st.caption("Each month was predicted by models trained only on the months before it — this is how "
           "the forecasts in the dataset period of this app were produced.")
monthly = svc.monthly
per_month = monthly[monthly.month != "Oct-Feb (all)"]
wf_metric = st.radio("Metric", ["RMSE", "MAE", "R2"], horizontal=True, key="wf",
                     format_func=lambda x: "R²" if x == "R2" else x)
fig = go.Figure()
for mdl in MODEL_NAMES:
    d = per_month[per_month.model == mdl]
    fig.add_trace(go.Scatter(x=d.month, y=d[wf_metric], name=mdl, mode="lines+markers",
                             line=dict(color=MODEL_COLORS[mdl], width=2), marker=dict(size=8)))
fig.update_layout(height=340, plot_bgcolor="white", hovermode="x unified",
                  margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.12),
                  xaxis_title="Predicted month (trained on all earlier months)",
                  yaxis=dict(title=wf_metric if wf_metric != "R2" else "R²", gridcolor="#e6e5e1"))
st.plotly_chart(fig, use_container_width=True)
tbl = monthly.pivot(index="month", columns="model", values=wf_metric)[MODEL_NAMES]
tbl = tbl.reindex([m_ for m_ in config.WALK_FORWARD_MONTHS] + ["Oct-Feb (all)"])
st.dataframe(tbl.style.format("{:.3f}" if wf_metric == "R2" else "{:,.1f}"), use_container_width=True)

# ---------------------------------------------------------------- future model
st.subheader("3 · Future-date models")
ev = svc.profile.evaluation.reset_index()
st.caption("Trained on 1 Sep 2022 – 31 Jan 2023 and tested on February 2023, using only inputs that "
           "are known for any date (no recent demand). Then retrained on all six months.")
st.dataframe(ev[["model", "MAE", "MSE", "RMSE", "R2"]].rename(columns={"model": "Model", "R2": "R²"})
             .style.format({"MAE": "{:,.2f}", "MSE": "{:,.0f}", "RMSE": "{:,.2f}", "R²": "{:.4f}"})
             .highlight_min(subset=["MAE", "MSE", "RMSE"], color="#d6f0e4")
             .highlight_max(subset=["R²"], color="#d6f0e4"), hide_index=True, use_container_width=True)
st.info("Future estimates are less precise than dataset-period forecasts because the most recent "
        "real demand is unknown. They reflect Sep 2022 – Feb 2023 patterns (no summer months, no EV "
        "growth since 2023).")

# ---------------------------------------------------------------- drivers
st.subheader("4 · What drives charging demand?")
imp = pd.read_csv(IMPORTANCE_FILE)
if name in imp.model.unique():
    d = imp[imp.model == name].copy()
    d["Factor"] = d.feature.map(FRIENDLY)
    d = d.groupby("Factor", as_index=False).importance.sum().nlargest(8, "importance")[::-1]
    fig = go.Figure(go.Bar(x=d.importance * 100, y=d.Factor, orientation="h",
                           marker_color=MODEL_COLORS.get(name, "#2a78d6"),
                           hovertemplate="%{x:.1f}%<extra></extra>"))
    fig.update_layout(height=360, plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="Share of importance (%)", gridcolor="#e6e5e1"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Recent demand history, especially the same hour on the previous day, is the strongest "
               "signal of next-hour demand.")

with st.expander("Detailed evaluation charts"):
    for fname, caption in [("actual_vs_predicted.png", "Actual vs predicted"),
                           ("residuals.png", "Residual analysis"),
                           ("feature_importance.png", "Feature importance and coefficients")]:
        if (EVAL_DIR / fname).exists():
            st.image(str(EVAL_DIR / fname), caption=caption, use_container_width=True)

# ---------------------------------------------------------------- downloads
st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
with d1:
    csv_download(comp, "model_comparison.csv", "Model comparison (CSV)")
    csv_download(monthly, "monthly_accuracy.csv", "Monthly accuracy (CSV)")
with d2:
    if PREDICTIONS_FILE.exists():
        preds = pd.read_parquet(PREDICTIONS_FILE, columns=["time", "zone", "actual"] + MODEL_NAMES)
        csv_download(preds, "test_predictions.csv", "February test predictions (CSV)")
with d3:
    for fname, mime in [("Project_Report.pdf", "application/pdf"),
                        ("Project_Report.docx",
                         "application/vnd.openxmlformats-officedocument.wordprocessingml.document")]:
        path = config.REPORTS_DIR / fname
        if path.exists():
            st.download_button(f"Project report ({path.suffix[1:].upper()})", path.read_bytes(),
                               file_name=fname, mime=mime)
