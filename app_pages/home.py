import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluate_models import regression_metrics
from src.predict import DATASET_END, DATASET_START
from src.ui import (ACTUAL_COLOR, CSS, PREDICTED_COLOR, get_service, hour_label, line_layout,
                    zone_label)

st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <h1>⚡ EV Charging Demand Prediction System</h1>
  <p>Predict how much energy electric vehicles will draw from charging stations in each
  zone of Shenzhen, for any date, and compare predictions with real demand.</p>
</div>""", unsafe_allow_html=True)

svc = get_service()
city = svc.dataset_summary()
best = svc.dataset_best

# ---------------------------------------------------------------- two modes
c1, c2 = st.columns(2)
c1.markdown(f"""<div class="card"><h4>📊 Dataset period</h4>
<p><b>{DATASET_START:%d %b %Y} – {DATASET_END:%d %b %Y}</b><br>
Each month was predicted by models trained only on earlier months. You see the prediction
<b>and</b> the actual demand, so you can judge the accuracy yourself.</p></div>""",
            unsafe_allow_html=True)
c2.markdown(f"""<div class="card"><h4>🔭 Future dates</h4>
<p><b>{DATASET_END + pd.Timedelta(hours=1):%d %b %Y} – {svc.max_date:%d %b %Y}</b><br>
Pattern-based estimates from models trained on all six months of data. No actual values
exist for these dates, so a typical error range is shown instead.</p></div>""",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- summary cards
st.subheader("Dataset period at a glance")
m = regression_metrics(city.actual, city[best])
zone_totals = svc.history.groupby("zone").actual.sum()
peak_hour = city.groupby(city.index.hour).actual.sum().idxmax()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Actual energy delivered", f"{city.actual.sum() / 1e6:,.1f} GWh")
k2.metric("Busiest zone", zone_label(zone_totals.idxmax()))
k3.metric("Busiest hour of the day", hour_label(int(peak_hour)))
k4.metric("City-wide forecast error", f"{100 * m['MAE'] / city.actual.mean():.1f}%",
          help="Average absolute error of the hourly city total, as a share of average demand.")

daily = city.resample("D").sum() / 1000
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily.index, y=daily.actual, name="Actual",
                         line=dict(color=ACTUAL_COLOR, width=2)))
fig.add_trace(go.Scatter(x=daily.index, y=daily[best], name=f"Predicted ({best})",
                         line=dict(color=PREDICTED_COLOR, width=2)))
line_layout(fig, "Daily city-wide charging demand: predicted vs actual (MWh)", "MWh", 340)
st.plotly_chart(fig, use_container_width=True)
st.caption("The dip in late January is the Spring Festival holiday.")

# ---------------------------------------------------------------- how to use
st.subheader("What you can do")
cards = [
    ("🔮 Demand Forecast", "Pick a zone, any date and hour: see the prediction of all three models."),
    ("🎯 Prediction vs Actual", "Choose a date range and compare predictions with real demand."),
    ("🗺️ Demand Map", "Predicted demand for every zone of the city at a chosen hour."),
    ("📈 Zone Insights", "A zone's 24-hour forecast, peak hours and the best times to charge."),
    ("⚖️ Compare Zones", "Compare up to three zones side by side."),
    ("ℹ️ About the Model", "How accurate the models are, month by month."),
]
for row in (cards[:3], cards[3:]):
    cols = st.columns(3)
    for col, (title, text) in zip(cols, row):
        col.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
    st.write("")

with st.expander("About this project"):
    st.markdown("""
**Problem.** EV charging demand changes strongly with time and location. Knowing it in advance
helps operators manage capacity and pricing, drivers avoid queues, and grid operators plan load.

**Data.** UrbanEV open dataset (Li et al., *Scientific Data*, 2025): 275 charging zones in
Shenzhen, hourly records from 1 Sep 2022 to 28 Feb 2023 (1,194,600 zone-hours), plus zone,
station, charger-power and zone-adjacency information.

**Models.** Linear Regression, Random Forest and LightGBM, trained on the dataset and compared.
""")
