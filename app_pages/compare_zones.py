import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import (ACTUAL_COLOR, ZONE_COLORS, csv_download, date_inputs, display_table, get_service,
                    hour_label, line_layout, mode_banner, period_note, setup_page, zone_label)

setup_page("⚖️ Compare Zones")
st.write("Compare the predicted demand of up to three zones on the same day.")

svc = get_service()
period_note()
zones = svc.zone_ids()
defaults = [z for z in (102, 1068, 1066) if z in zones][:3] or zones[:2]
c1, c2 = st.columns([2, 1])
chosen = c1.multiselect("Zones (2 or 3)", zones, default=defaults, max_selections=3,
                        format_func=zone_label)
with c2:
    day, _ = date_inputs(svc, "cz", with_hour=False)

if len(chosen) < 2:
    st.info("Select at least two zones to compare.")
    st.stop()
try:
    frames = {z: svc.zone_day(z, day) for z in chosen}
except ValueError as exc:
    st.error(str(exc))
    st.stop()
mode = next(iter(frames.values()))["mode"].iloc[0]
mode_banner(mode)

fig = go.Figure()
for color, (z, df) in zip(ZONE_COLORS, frames.items()):
    fig.add_trace(go.Scatter(x=df.hour, y=df.predicted_kwh, name=f"{zone_label(z)} predicted",
                             mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=6)))
    if mode == "dataset":
        fig.add_trace(go.Scatter(x=df.hour, y=df.actual_kwh, name=f"{zone_label(z)} actual",
                                 mode="lines", line=dict(color=color, width=1.5, dash="dot")))
line_layout(fig, f"Hourly demand · {day:%A %d %B %Y}", hours_axis=True, height=420)
st.plotly_chart(fig, use_container_width=True)
if mode == "dataset":
    st.caption("Solid lines = prediction, dotted lines = actual demand.")

info = svc.zones.set_index("zone")
rows = []
for z, df in frames.items():
    row = {"Zone": zone_label(z),
           "Predicted daily demand (kWh)": round(df.predicted_kwh.sum(), 1)}
    if mode == "dataset":
        row["Actual daily demand (kWh)"] = round(df.actual_kwh.sum(), 1)
    row.update({"Peak hour": hour_label(int(df.loc[df.predicted_kwh.idxmax(), "hour"])),
                "Peak demand (kWh)": round(df.predicted_kwh.max(), 1),
                "Average utilisation (%)": round(df.utilisation_pct.mean(), 1),
                "Chargers": int(info.at[z, "num_piles"]), "Fast chargers": int(info.at[z, "fast_count"])})
    rows.append(row)
st.subheader("Summary")
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

combined = pd.concat([display_table(df) for df in frames.values()], ignore_index=True)
csv_download(combined, f"zone_comparison_{day:%Y%m%d}.csv", "Download comparison (CSV)")
