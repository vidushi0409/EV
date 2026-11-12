import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluate_models import regression_metrics
from src.predict import DATASET_END, DATASET_START
from src.ui import (ACTUAL_COLOR, CSS, PREDICTED_COLOR, get_service, hour_label, kpi_row, line_layout,
                    zone_label)

st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <h1>⚡ EV Charging Demand Prediction</h1>
  <p>Hourly charging-demand forecasts for 275 zones in Shenzhen.</p>
</div>""", unsafe_allow_html=True)

svc = get_service()
city = svc.dataset_summary()
best = svc.dataset_best

m = regression_metrics(city.actual, city[best])
zone_totals = svc.history.groupby("zone").actual.sum()
peak_hour = city.groupby(city.index.hour).actual.sum().idxmax()
kpi_row([
    ("Energy delivered", f"{city.actual.sum() / 1e6:,.1f} GWh", f"{DATASET_START:%b %Y} – {DATASET_END:%b %Y}"),
    ("Busiest zone", zone_label(zone_totals.idxmax())),
    ("Peak hour", hour_label(int(peak_hour))),
    ("Forecast error (city)", f"{100 * m['MAE'] / city.actual.mean():.1f}%", f"{best}",
     "Average absolute error of the hourly city total, as a share of average demand"),
])

daily = city.resample("D").sum() / 1000
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily.index, y=daily.actual, name="Actual", line=dict(color=ACTUAL_COLOR, width=2)))
fig.add_trace(go.Scatter(x=daily.index, y=daily[best], name="Predicted", line=dict(color=PREDICTED_COLOR, width=2)))
line_layout(fig, "Daily city demand (MWh)", "MWh", 320)
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
c1.markdown(f'<div class="card"><h4>📊 {DATASET_START:%d %b %Y} – {DATASET_END:%d %b %Y}</h4>'
            '<p>Prediction compared with actual demand.</p></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="card"><h4>🔭 {DATASET_END + pd.Timedelta(hours=1):%d %b %Y} – today</h4>'
            '<p>Estimates from learnt demand patterns.</p></div>', unsafe_allow_html=True)
st.write("")

st.subheader("Explore")
links = [("app_pages/demand_forecast.py", "Demand Forecast", "🔮"),
         ("app_pages/prediction_vs_actual.py", "Prediction vs Actual", "🎯"),
         ("app_pages/demand_map.py", "Demand Map", "🗺️"),
         ("app_pages/zone_insights.py", "Zone Insights", "📈"),
         ("app_pages/compare_zones.py", "Compare Zones", "⚖️"),
         ("app_pages/about_model.py", "About the Model", "ℹ️")]
cols = st.columns(3)
for i, (page, label, icon) in enumerate(links):
    try:
        cols[i % 3].page_link(page, label=label, icon=icon, use_container_width=True)
    except Exception:   # page opened outside the app's navigation (e.g. in tests)
        cols[i % 3].markdown(f"{icon} {label}")
