import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import (ACTUAL_COLOR, PREDICTED_COLOR, date_hour_inputs, forecast_period_note,
                    get_forecaster, hour_label, level_badge, setup_page, zone_label)

setup_page("🔮 Demand Forecast")
st.write("Choose a charging zone and a time to see how much energy EVs are expected to draw.")

fc = get_forecaster()
forecast_period_note(fc)

with st.form("forecast_form"):
    zones = fc.zone_ids()
    zone = st.selectbox("Charging zone", zones, index=zones.index(102) if 102 in zones else 0,
                        format_func=zone_label)
    day, hour = date_hour_inputs(fc, "fc")
    submitted = st.form_submit_button("Predict demand", type="primary")

if submitted:
    try:
        ts = pd.Timestamp(day) + pd.Timedelta(hours=int(hour))
        with st.spinner("Forecasting ..."):
            st.session_state["forecast"] = (fc.zone_hour(zone, ts), fc.zone_day(zone, day))
    except ValueError as exc:
        st.error(str(exc))

if "forecast" in st.session_state:
    r, day_df = st.session_state["forecast"]
    st.subheader(f"{zone_label(r['zone'])} · {r['time']:%A %d %B %Y, %H:00}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted demand", f"{r['predicted_kwh']:,.1f} kWh")
    with c2:
        st.markdown('<p class="note">Demand level (vs this zone\'s usual)</p>'
                    + level_badge(r["level"]), unsafe_allow_html=True)
    c3.metric("Estimated charger utilisation", f"{r['utilisation_pct']:.0f}%")
    c4.metric("Charging price", f"{r['price_per_kwh']:.2f} RMB/kWh")

    diff = r["predicted_kwh"] - r["actual_kwh"]
    st.markdown("##### How did the forecast do?")
    a1, a2, a3 = st.columns(3)
    a1.metric("Actual demand", f"{r['actual_kwh']:,.1f} kWh")
    a2.metric("Forecast error", f"{abs(diff):,.1f} kWh",
              f"{'over' if diff > 0 else 'under'}-estimate", delta_color="off")
    a3.metric("Previous hour (actual)", f"{r['previous_hour_kwh']:,.1f} kWh")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=day_df.hour, y=day_df.actual_kwh, name="Actual",
                             line=dict(color=ACTUAL_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=day_df.hour, y=day_df.predicted_kwh, name="Predicted",
                             line=dict(color=PREDICTED_COLOR, width=2), mode="lines+markers",
                             marker=dict(size=6)))
    fig.add_vline(x=r["time"].hour, line_dash="dot", line_color="#8a8a85")
    fig.update_layout(title="Demand over the whole day (selected hour marked)", height=330,
                      hovermode="x unified", plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.15, x=1, xanchor="right"),
                      xaxis=dict(title="Hour", tickmode="array", tickvals=list(range(0, 24, 3)),
                                 ticktext=[hour_label(h) for h in range(0, 24, 3)]),
                      yaxis=dict(title="kWh", gridcolor="#e6e5e1"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Demand level compares the forecast with this zone's normal range: "
               "Low (bottom 25% of its hours), Moderate, High (top 25%), Very High (top 10%). "
               "Utilisation = predicted energy ÷ the zone's total charger power.")
