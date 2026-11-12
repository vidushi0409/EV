import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import DATASET_END, DATASET_START
from src.train_models import MODEL_NAMES
from src.ui import (ACTUAL_COLOR, MODEL_COLORS, csv_download, get_service, line_layout, setup_page,
                    zone_label)

setup_page("🎯 Prediction vs Actual")
st.write("Choose a period and see how closely each model's predictions followed the real "
         "charging demand recorded in the dataset.")
st.caption(f"Available for {DATASET_START:%d %b %Y} – {DATASET_END:%d %b %Y}, where actual demand is "
           "known. Every prediction was made by models trained only on earlier months.")

svc = get_service()
zones = svc.zone_ids()
c1, c2, c3 = st.columns([1.2, 1, 1])
target = c1.selectbox("Area", ["Whole city"] + zones,
                      format_func=lambda z: z if z == "Whole city" else zone_label(z))
start = c2.date_input("From", value=pd.Timestamp("2023-02-06").date(), min_value=DATASET_START.date(),
                      max_value=DATASET_END.date(), format="DD/MM/YYYY")
end = c3.date_input("To", value=pd.Timestamp("2023-02-12").date(), min_value=DATASET_START.date(),
                    max_value=DATASET_END.date(), format="DD/MM/YYYY")

try:
    hourly, metrics = svc.compare_range(start, end, None if target == "Whole city" else target)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

best = svc.dataset_best
bm = metrics.set_index("model").loc[best]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Actual demand in period", f"{hourly.actual_kwh.sum() / 1000:,.1f} MWh")
k2.metric(f"Predicted ({best})", f"{hourly[best].sum() / 1000:,.1f} MWh",
          f"{bm.total_error_pct:.1f}% total difference", delta_color="off")
k3.metric("Average hourly error", f"{bm.MAE:,.1f} kWh")
k4.metric("R² (share of variation explained)", f"{bm.R2:.3f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=hourly.time, y=hourly.actual_kwh, name="Actual",
                         line=dict(color=ACTUAL_COLOR, width=2.5)))
for m in MODEL_NAMES:
    fig.add_trace(go.Scatter(x=hourly.time, y=hourly[m], name=m,
                             line=dict(color=MODEL_COLORS[m], width=1.6),
                             visible=True if m == best else "legendonly"))
area = "city-wide" if target == "Whole city" else zone_label(target)
line_layout(fig, f"Hourly demand, {area}: predicted vs actual", height=420)
st.plotly_chart(fig, use_container_width=True)
st.caption("Click a model name in the legend to show or hide it.")

st.subheader("Model accuracy for this period")
shown = metrics.rename(columns={"model": "Model", "R2": "R²", "total_error_pct": "Total difference (%)"})
shown = shown[["Model", "MAE", "RMSE", "MSE", "R²", "Total difference (%)"]]
st.dataframe(shown.style.format({"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "MSE": "{:,.0f}", "R²": "{:.3f}",
                                 "Total difference (%)": "{:.1f}"})
             .highlight_min(subset=["MAE", "RMSE", "MSE"], color="#d6f0e4")
             .highlight_max(subset=["R²"], color="#d6f0e4"),
             hide_index=True, use_container_width=True)
st.caption("MAE / RMSE in kWh per hour. Lower is better, except R² (higher is better).")

out = hourly.rename(columns={"time": "Time", "actual_kwh": "Actual (kWh)"})
csv_download(out, f"prediction_vs_actual_{start:%Y%m%d}_{end:%Y%m%d}.csv", "Download comparison (CSV)")
