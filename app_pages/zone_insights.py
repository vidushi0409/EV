import plotly.graph_objects as go
import streamlit as st

from src.train_models import MODEL_NAMES
from src.ui import (ACTUAL_COLOR, LEVEL_COLORS, MODEL_COLORS, csv_download, date_inputs,
                    display_table, get_service, hour_label, line_layout, mode_banner, period_note,
                    setup_page, zone_label)

setup_page("📈 Zone Insights")
st.write("A full-day forecast for one zone: when it gets busy, when it is quiet, and the best "
         "times to charge.")

svc = get_service()
period_note()
zones = svc.zone_ids()
c1, c2 = st.columns(2)
zone = c1.selectbox("Charging zone", zones, index=zones.index(102) if 102 in zones else 0,
                    format_func=zone_label)
with c2:
    day, _ = date_inputs(svc, "zi", with_hour=False)

try:
    df = svc.zone_day(zone, day)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
mode = df["mode"].iloc[0]
best = df["best_model"].iloc[0]
mode_banner(mode)
info = svc.zones.set_index("zone").loc[zone]

peak = df.loc[df.predicted_kwh.idxmax()]
quiet = df.loc[df.predicted_kwh.idxmin()]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted daily demand", f"{df.predicted_kwh.sum():,.0f} kWh",
          f"actual {df.actual_kwh.sum():,.0f} kWh" if mode == "dataset" else None, delta_color="off")
m2.metric("Peak hour", hour_label(int(peak.hour)), f"{peak.predicted_kwh:,.0f} kWh", delta_color="off")
m3.metric("Quietest hour", hour_label(int(quiet.hour)), f"{quiet.predicted_kwh:,.0f} kWh",
          delta_color="off")
m4.metric("Chargers in zone", f"{int(info.num_piles)}",
          f"{int(info.fast_count)} fast · {int(info.slow_count)} slow", delta_color="off")

show_all = st.toggle("Show all three models", value=False)
fig = go.Figure()
fig.add_trace(go.Bar(x=df.hour, y=df.predicted_kwh, name=f"Predicted ({best})",
                     marker_color=[LEVEL_COLORS[l] for l in df.level],
                     customdata=df[["level", "utilisation_pct"]],
                     hovertemplate="%{y:,.1f} kWh<br>%{customdata[0]} · "
                                   "%{customdata[1]:.0f}% utilisation<extra>Predicted</extra>"))
if show_all:
    for m in MODEL_NAMES:
        fig.add_trace(go.Scatter(x=df.hour, y=df[m], name=m, mode="lines",
                                 line=dict(color=MODEL_COLORS[m], width=2, dash="dot")))
if mode == "dataset":
    fig.add_trace(go.Scatter(x=df.hour, y=df.actual_kwh, name="Actual", mode="lines",
                             line=dict(color=ACTUAL_COLOR, width=2.5)))
line_layout(fig, f"{zone_label(zone)} · {day:%A %d %B %Y}", hours_axis=True)
fig.update_layout(bargap=0.25)
st.plotly_chart(fig, use_container_width=True)
st.caption("Bar colour = demand level compared with this zone's usual demand "
           "(light = Low → dark = Very High).")

st.subheader("💡 Best times to charge")
st.write("Hours with the lowest predicted demand (shortest expected queues), with the price.")
quietest = df.nsmallest(3, "predicted_kwh").sort_values("hour")
cheapest = df.nsmallest(3, "price_per_kwh").sort_values("hour")
b1, b2 = st.columns(2)
with b1:
    st.markdown("**Quietest hours**")
    for _, r in quietest.iterrows():
        st.markdown(f"- **{hour_label(int(r.hour))}** · {r.predicted_kwh:,.0f} kWh predicted · "
                    f"{r.price_per_kwh:.2f} RMB/kWh")
with b2:
    st.markdown("**Cheapest hours**")
    for _, r in cheapest.iterrows():
        st.markdown(f"- **{hour_label(int(r.hour))}** · {r.price_per_kwh:.2f} RMB/kWh · "
                    f"{r.predicted_kwh:,.0f} kWh predicted")
if mode == "future":
    st.caption("Prices for future dates are the zone's typical tariff from the last 4 weeks of data.")

with st.expander("Hourly table"):
    st.dataframe(display_table(df), hide_index=True, use_container_width=True)
csv_download(display_table(df), f"zone_{zone}_{day:%Y%m%d}_forecast.csv", "Download this forecast (CSV)")
