import plotly.graph_objects as go
import streamlit as st

from src.train_models import MODEL_NAMES
from src.ui import (ACTUAL_COLOR, LEVEL_COLORS, MODEL_COLORS, csv_download, date_inputs, display_table,
                    get_service, hour_label, kpi_row, line_layout, mode_banner, setup_page, zone_label,
                    zone_select)

setup_page("📈 Zone Insights")
svc = get_service()

c1, c2 = st.columns(2)
zone = zone_select(svc, c1)
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
kpi_row([
    ("Daily demand", f"{df.predicted_kwh.sum():,.0f} kWh",
     f"actual {df.actual_kwh.sum():,.0f} kWh" if mode == "dataset" else "estimate"),
    ("Peak hour", hour_label(int(peak.hour)), f"{peak.predicted_kwh:,.0f} kWh"),
    ("Quietest hour", hour_label(int(quiet.hour)), f"{quiet.predicted_kwh:,.0f} kWh"),
    ("Chargers", f"{int(info.num_piles)}", f"{int(info.fast_count)} fast · {int(info.slow_count)} slow"),
])

show_all = st.toggle("Show all models")
fig = go.Figure()
fig.add_trace(go.Bar(x=df.hour, y=df.predicted_kwh, name=f"Predicted ({best})",
                     marker_color=[LEVEL_COLORS[l] for l in df.level], customdata=df[["level"]],
                     hovertemplate="%{y:,.1f} kWh · %{customdata[0]}<extra></extra>"))
if show_all:
    for m in MODEL_NAMES:
        fig.add_trace(go.Scatter(x=df.hour, y=df[m], name=m, mode="lines",
                                 line=dict(color=MODEL_COLORS[m], width=2, dash="dot")))
if mode == "dataset":
    fig.add_trace(go.Scatter(x=df.hour, y=df.actual_kwh, name="Actual", mode="lines",
                             line=dict(color=ACTUAL_COLOR, width=2.5)))
line_layout(fig, f"{zone_label(zone)} · {day:%a %d %b %Y}", hours_axis=True)
fig.update_layout(bargap=0.25)
st.plotly_chart(fig, use_container_width=True)

st.subheader("💡 Best times to charge")
b1, b2 = st.columns(2)
for col, title, frame in [(b1, "Quietest", df.nsmallest(3, "predicted_kwh")),
                          (b2, "Cheapest", df.nsmallest(3, "price_per_kwh"))]:
    lines = "".join(f"<li><b>{hour_label(int(r.hour))}</b> · {r.predicted_kwh:,.0f} kWh · "
                    f"{r.price_per_kwh:.2f} RMB/kWh</li>" for _, r in frame.sort_values("hour").iterrows())
    col.markdown(f'<div class="card"><h4>{title}</h4><ul style="margin:6px 0 0 0">{lines}</ul></div>',
                 unsafe_allow_html=True)
st.write("")

with st.expander("Hourly table"):
    st.dataframe(display_table(df), hide_index=True, use_container_width=True)
csv_download(display_table(df), f"zone_{zone}_{day:%Y%m%d}.csv")
