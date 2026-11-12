import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import (ZONE_COLORS, csv_download, date_inputs, display_table, get_service, hour_label,
                    line_layout, mode_banner, setup_page, zone_label)

setup_page("⚖️ Compare Zones")
svc = get_service()

zones = svc.zone_ids()
defaults = [z for z in (102, 1068, 1066) if z in zones][:3] or zones[:2]
c1, c2 = st.columns([2, 1])
chosen = c1.multiselect("Zones (2–3)", zones, default=defaults, max_selections=3, format_func=zone_label)
with c2:
    day, _ = date_inputs(svc, "cz", with_hour=False)

if len(chosen) < 2:
    st.info("Select at least two zones.")
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
    fig.add_trace(go.Scatter(x=df.hour, y=df.predicted_kwh, name=zone_label(z), mode="lines+markers",
                             line=dict(color=color, width=2), marker=dict(size=5)))
    if mode == "dataset":
        fig.add_trace(go.Scatter(x=df.hour, y=df.actual_kwh, name=f"{zone_label(z)} actual", mode="lines",
                                 line=dict(color=color, width=1.4, dash="dot"), showlegend=False))
title = f"{day:%a %d %b %Y}" + (" · dotted = actual" if mode == "dataset" else "")
line_layout(fig, title, hours_axis=True, height=400)
st.plotly_chart(fig, use_container_width=True)

info = svc.zones.set_index("zone")
rows = []
for z, df in frames.items():
    row = {"Zone": zone_label(z), "Predicted (kWh/day)": df.predicted_kwh.sum()}
    if mode == "dataset":
        row["Actual (kWh/day)"] = df.actual_kwh.sum()
    row.update({"Peak hour": hour_label(int(df.loc[df.predicted_kwh.idxmax(), "hour"])),
                "Peak (kWh)": df.predicted_kwh.max(), "Utilisation (%)": df.utilisation_pct.mean(),
                "Chargers": int(info.at[z, "num_piles"]), "Fast": int(info.at[z, "fast_count"])})
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
             column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                            ("Predicted (kWh/day)", "Actual (kWh/day)", "Peak (kWh)", "Utilisation (%)")})
csv_download(pd.concat([display_table(df) for df in frames.values()], ignore_index=True),
             f"zone_comparison_{day:%Y%m%d}.csv")
