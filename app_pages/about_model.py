import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.evaluate_models import (COMPARISON_FILE, EVAL_DIR, IMPORTANCE_FILE, PREDICTIONS_FILE,
                                 read_best_model)
from src.plot_style import MODEL_COLORS
from src.ui import csv_download, setup_page

setup_page("ℹ️ About the Model")

best = read_best_model()
if best is None or not COMPARISON_FILE.exists():
    st.error("Model results not found. Run `python train.py` in the project folder.")
    st.stop()

FRIENDLY = {
    "volume_lag24": "Demand at the same hour yesterday",
    "volume_lag168": "Demand at the same hour last week",
    "volume_lag1": "Demand in the previous hour",
    "volume_roll24_mean": "Average demand over the last 24 hours",
    "duration_lag1": "Charging time in the previous hour",
    "occupancy_lag1": "Chargers in use in the previous hour",
    "neighbor_volume_lag1": "Neighbouring zones' demand in the previous hour",
    "hour": "Hour of the day", "hour_sin": "Hour of the day (cyclical)",
    "hour_cos": "Hour of the day (cyclical)", "day": "Day of the month",
    "dayofweek": "Day of the week", "dow_sin": "Day of the week (cyclical)",
    "dow_cos": "Day of the week (cyclical)", "is_weekend": "Weekend", "is_holiday": "Public holiday",
    "e_price": "Electricity price", "s_price": "Service price", "num_piles": "Number of chargers",
    "fast_share": "Share of fast chargers", "avg_pile_kw": "Average charger power",
    "capacity_kw": "Total charger power", "area": "Zone area",
    "longitude": "Location (longitude)", "latitude": "Location (latitude)",
}

name = best["best_model"]
m = best["test_metrics"]
st.markdown(f"Forecasts in this app are produced by **{name}**, selected from three "
            "machine-learning models.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{m['MAE']:,.1f} kWh")
c2.metric("RMSE", f"{m['RMSE']:,.1f} kWh")
c3.metric("MSE", f"{m['MSE']:,.0f}")
c4.metric("R²", f"{m['R2']:.3f}")
st.caption("Measured on February 2023: 184,800 zone-hours that were never used for training.")

# ---------------------------------------------------------------- comparison
st.subheader("Model comparison")
comp = pd.read_csv(COMPARISON_FILE)
shown = comp.copy()
shown["Type"] = shown.model.map(lambda x: "Reference (no learning)" if x.startswith("Naive")
                                else "Machine learning")
shown = shown[["model", "Type", "MAE", "MSE", "RMSE", "R2"]].rename(
    columns={"model": "Model", "R2": "R²"})
st.dataframe(shown.style.format({"MAE": "{:,.2f}", "MSE": "{:,.0f}", "RMSE": "{:,.2f}",
                                 "R²": "{:.4f}"})
             .highlight_min(subset=["MAE", "MSE", "RMSE"], color="#d6f0e4")
             .highlight_max(subset=["R²"], color="#d6f0e4"),
             hide_index=True, use_container_width=True)

metric = st.radio("Compare by", ["RMSE", "MAE", "MSE", "R2"], horizontal=True,
                  format_func=lambda x: "R²" if x == "R2" else x)
fig = go.Figure(go.Bar(x=comp.model, y=comp[metric],
                       marker_color=[MODEL_COLORS[x] for x in comp.model],
                       text=[f"{v:,.3f}" if metric == "R2" else f"{v:,.1f}" for v in comp[metric]],
                       textposition="outside"))
fig.update_layout(height=340, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10),
                  yaxis=dict(title=("higher is better" if metric == "R2" else "lower is better"),
                             gridcolor="#e6e5e1"))
st.plotly_chart(fig, use_container_width=True)

val = best["validation_rmse"]
st.markdown(f"""
**Why {name}?** It had the lowest error on the January 2023 validation data
(RMSE: {', '.join(f'{k} {v:,.1f}' for k, v in val.items())}), and the result was
{'**confirmed**' if best['selection_confirmed_on_test'] else '**not** confirmed'} on the
February test data. All models clearly beat the simple reference forecasts
("repeat last hour" / "repeat last week").
""")

# ---------------------------------------------------------------- drivers
st.subheader("What drives charging demand?")
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
    st.caption("Recent demand history, especially the same hour on the previous day, "
               "is the strongest signal of next-hour demand.")
else:
    st.info("Coefficient-based model: see the project report for the coefficients.")

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
with d2:
    if PREDICTIONS_FILE.exists():
        preds = pd.read_parquet(PREDICTIONS_FILE, columns=["time", "zone", "actual", name])
        preds = preds.rename(columns={"actual": "actual_kwh", name: "predicted_kwh"})
        csv_download(preds, "test_predictions.csv", "Test-period predictions (CSV)")
with d3:
    for fname, mime in [("Project_Report.pdf", "application/pdf"),
                        ("Project_Report.docx",
                         "application/vnd.openxmlformats-officedocument.wordprocessingml.document")]:
        path = config.REPORTS_DIR / fname
        if path.exists():
            st.download_button(f"Project report ({path.suffix[1:].upper()})", path.read_bytes(),
                               file_name=fname, mime=mime)
