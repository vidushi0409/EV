import plotly.graph_objects as go
import streamlit as st

from src.evaluate_models import read_best_model
from src.ui import (ACTUAL_COLOR, CSS, PREDICTED_COLOR, get_forecaster, hour_label,
                    period_forecasts, zone_label)

st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <h1>⚡ EV Charging Demand Prediction System</h1>
  <p>Forecast how much energy electric vehicles will draw from charging stations,
  zone by zone and hour by hour, across Shenzhen.</p>
</div>""", unsafe_allow_html=True)

fc = get_forecaster()
pf = period_forecasts()
best = read_best_model()

# ---------------------------------------------------------------- summary cards
city_hourly = pf.groupby("time")[["predicted_kwh", "actual_kwh"]].sum()
zone_totals = pf.groupby("zone").predicted_kwh.sum()
peak_hour = pf.groupby(pf.time.dt.hour).predicted_kwh.sum().idxmax()

st.subheader(f"City overview · {fc.start:%B %Y}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total predicted demand", f"{pf.predicted_kwh.sum() / 1e6:,.2f} GWh")
c2.metric("Average city demand per hour", f"{city_hourly.predicted_kwh.mean() / 1000:,.1f} MWh")
c3.metric("Busiest zone", zone_label(zone_totals.idxmax()),
          f"{zone_totals.max() / 1000:,.0f} MWh this month", delta_color="off")
c4.metric("Busiest hour of the day", hour_label(peak_hour))

# ---------------------------------------------------------------- daily chart
daily = pf.groupby(pf.time.dt.date)[["predicted_kwh", "actual_kwh"]].sum() / 1000
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily.index, y=daily.actual_kwh, name="Actual",
                         line=dict(color=ACTUAL_COLOR, width=2)))
fig.add_trace(go.Scatter(x=daily.index, y=daily.predicted_kwh, name="Predicted",
                         line=dict(color=PREDICTED_COLOR, width=2)))
fig.update_layout(title="Daily city-wide charging demand (MWh)", height=340,
                  hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
                  legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                  yaxis_title="MWh", plot_bgcolor="white")
fig.update_yaxes(gridcolor="#e6e5e1")
st.plotly_chart(fig, use_container_width=True)

if best:
    mae = best["test_metrics"]["MAE"]
    st.success(f"Forecast accuracy: on average, predictions are within **{mae:,.0f} kWh** "
               f"of the actual hourly demand of a zone "
               f"(R² = {best['test_metrics']['R2']:.3f} on data the model never saw).")

# ---------------------------------------------------------------- how to use
st.subheader("What you can do")
cols = st.columns(4)
cards = [
    ("🔮 Demand Forecast", "Pick a zone, date and hour to get the predicted charging demand."),
    ("🗺️ Demand Map", "See predicted demand for every zone of the city at a chosen hour."),
    ("📈 Zone Insights", "View a zone's 24-hour forecast, peak hours and the best times to charge."),
    ("⚖️ Compare Zones", "Compare the predicted demand of up to three zones side by side."),
]
for col, (title, text) in zip(cols, cards):
    col.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

with st.expander("About this project"):
    st.markdown("""
**Problem.** EV charging demand changes strongly with time and location. Some stations sit idle
while others are overcrowded at peak hours. Knowing demand in advance helps operators manage
capacity and pricing, helps drivers avoid queues, and helps grid operators plan for EV load.

**Objective.** Predict the energy (kWh) delivered by EV chargers in each zone during the next hour.

**Data.** UrbanEV open dataset (Li et al., *Scientific Data*, 2025): 275 charging zones in
Shenzhen, hourly records from 1 Sep 2022 to 28 Feb 2023 (1,194,600 zone-hours), plus zone,
station, charger-power and zone-adjacency information.

**How forecasts are made.** Three machine-learning models (Linear Regression, Random Forest,
LightGBM) were trained on Sep–Dec 2022 and compared; the most accurate one produces every
forecast in this app. See *About the Model* for details.
""")
