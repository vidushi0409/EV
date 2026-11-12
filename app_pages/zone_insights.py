import plotly.graph_objects as go
import streamlit as st

from src.ui import (ACTUAL_COLOR, LEVEL_COLORS, csv_download, date_hour_inputs, display_table,
                    forecast_period_note, get_forecaster, hour_label, setup_page, zone_label)

setup_page("📈 Zone Insights")
st.write("A full-day forecast for one zone: when it gets busy, when it is quiet, "
         "and the best times to charge.")

fc = get_forecaster()
forecast_period_note(fc)

zones = fc.zone_ids()
c1, c2 = st.columns(2)
zone = c1.selectbox("Charging zone", zones, index=zones.index(102) if 102 in zones else 0,
                    format_func=zone_label)
with c2:
    day, _ = date_hour_inputs(fc, "zi", with_hour=False)

df = fc.zone_day(zone, day)
info = fc.zones.set_index("zone").loc[zone]

peak = df.loc[df.predicted_kwh.idxmax()]
quiet = df.loc[df.predicted_kwh.idxmin()]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted daily demand", f"{df.predicted_kwh.sum():,.0f} kWh",
          f"actual {df.actual_kwh.sum():,.0f} kWh", delta_color="off")
m2.metric("Peak hour", hour_label(int(peak.hour)), f"{peak.predicted_kwh:,.0f} kWh", delta_color="off")
m3.metric("Quietest hour", hour_label(int(quiet.hour)), f"{quiet.predicted_kwh:,.0f} kWh",
          delta_color="off")
m4.metric("Chargers in zone", f"{int(info.num_piles)}",
          f"{int(info.fast_count)} fast · {int(info.slow_count)} slow", delta_color="off")

fig = go.Figure()
fig.add_trace(go.Bar(x=df.hour, y=df.predicted_kwh, name="Predicted",
                     marker_color=[LEVEL_COLORS[l] for l in df.level],
                     customdata=df[["level", "utilisation_pct"]],
                     hovertemplate="%{y:,.1f} kWh<br>%{customdata[0]} · "
                                   "%{customdata[1]:.0f}% utilisation<extra>Predicted</extra>"))
fig.add_trace(go.Scatter(x=df.hour, y=df.actual_kwh, name="Actual", mode="lines",
                         line=dict(color=ACTUAL_COLOR, width=2)))
fig.update_layout(title=f"{zone_label(zone)} · {day:%A %d %B %Y}", height=380, bargap=0.25,
                  plot_bgcolor="white", hovermode="x unified",
                  margin=dict(l=10, r=10, t=50, b=10),
                  legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                  xaxis=dict(title="Hour", tickmode="array", tickvals=list(range(0, 24, 2)),
                             ticktext=[hour_label(h) for h in range(0, 24, 2)]),
                  yaxis=dict(title="kWh", gridcolor="#e6e5e1"))
st.plotly_chart(fig, use_container_width=True)
st.caption("Bar colour = demand level compared with this zone's usual demand "
           "(light = Low → dark = Very High).")

st.subheader("💡 Best times to charge")
st.write("Hours with the lowest predicted demand (shortest expected queues), with the price "
         "you would pay.")
best = df.nsmallest(3, "predicted_kwh").sort_values("hour")
cheap = df.nsmallest(3, "price_per_kwh").sort_values("hour")
b1, b2 = st.columns(2)
with b1:
    st.markdown("**Quietest hours**")
    for _, r in best.iterrows():
        st.markdown(f"- **{hour_label(int(r.hour))}** · {r.predicted_kwh:,.0f} kWh predicted · "
                    f"{r.price_per_kwh:.2f} RMB/kWh")
with b2:
    st.markdown("**Cheapest hours**")
    for _, r in cheap.iterrows():
        st.markdown(f"- **{hour_label(int(r.hour))}** · {r.price_per_kwh:.2f} RMB/kWh · "
                    f"{r.predicted_kwh:,.0f} kWh predicted")

with st.expander("Hourly forecast table"):
    st.dataframe(display_table(df), hide_index=True, use_container_width=True)
csv_download(display_table(df), f"zone_{zone}_{day:%Y%m%d}_forecast.csv",
             "Download this forecast (CSV)")
