import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import DATASET_END, DATASET_START
from src.train_models import MODEL_NAMES
from src.ui import (ACTUAL_COLOR, MODEL_COLORS, csv_download, get_service, kpi_row, line_layout,
                    setup_page, zone_label)

setup_page("🎯 Prediction vs Actual")
svc = get_service()

zones = svc.zone_ids()
c1, c2, c3 = st.columns([1.2, 1, 1])
target = c1.selectbox("Area", ["Whole city"] + zones,
                      format_func=lambda z: z if z == "Whole city" else zone_label(z))
limits = dict(min_value=DATASET_START.date(), max_value=DATASET_END.date(), format="DD/MM/YYYY")
start = c2.date_input("From", value=pd.Timestamp("2023-02-06").date(), **limits)
end = c3.date_input("To", value=pd.Timestamp("2023-02-12").date(), **limits)

try:
    hourly, metrics = svc.compare_range(start, end, None if target == "Whole city" else target)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

best = svc.dataset_best
bm = metrics.set_index("model").loc[best]
kpi_row([
    ("Actual", f"{hourly.actual_kwh.sum() / 1000:,.1f} MWh"),
    (f"Predicted · {best}", f"{hourly[best].sum() / 1000:,.1f} MWh", f"{bm.total_error_pct:.1f}% difference"),
    ("Avg hourly error", f"{bm.MAE:,.1f} kWh", "MAE"),
    ("R²", f"{bm.R2:.3f}", "", "Share of the variation in demand explained by the model (1 = perfect)"),
])

fig = go.Figure()
fig.add_trace(go.Scatter(x=hourly.time, y=hourly.actual_kwh, name="Actual",
                         line=dict(color=ACTUAL_COLOR, width=2.5)))
for m in MODEL_NAMES:
    fig.add_trace(go.Scatter(x=hourly.time, y=hourly[m], name=m, line=dict(color=MODEL_COLORS[m], width=1.6),
                             visible=True if m == best else "legendonly"))
line_layout(fig, "Hourly demand" + ("" if target == "Whole city" else f" · {zone_label(target)}"), height=400)
st.plotly_chart(fig, use_container_width=True)

shown = metrics.rename(columns={"model": "Model", "R2": "R²", "total_error_pct": "Total diff. (%)"})
shown = shown[["Model", "MAE", "RMSE", "R²", "Total diff. (%)"]]
st.dataframe(shown.style.format({"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "R²": "{:.3f}", "Total diff. (%)": "{:.1f}"})
             .highlight_min(subset=["MAE", "RMSE"], color="#d6f0e4").highlight_max(subset=["R²"], color="#d6f0e4"),
             hide_index=True, use_container_width=True)
csv_download(hourly.rename(columns={"time": "Time", "actual_kwh": "Actual (kWh)"}),
             f"prediction_vs_actual_{start:%Y%m%d}_{end:%Y%m%d}.csv")
