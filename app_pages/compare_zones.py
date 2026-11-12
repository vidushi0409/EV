import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import (ZONE_COLORS, csv_download, date_hour_inputs, display_table,
                    forecast_period_note, get_forecaster, hour_label, setup_page, zone_label)

setup_page("⚖️ Compare Zones")
st.write("Compare the predicted demand of up to three zones on the same day.")

fc = get_forecaster()
forecast_period_note(fc)

zones = fc.zone_ids()
defaults = [z for z in (102, 1068, 1066) if z in zones][:3] or zones[:2]
c1, c2 = st.columns([2, 1])
chosen = c1.multiselect("Zones (2 or 3)", zones, default=defaults, max_selections=3,
                        format_func=zone_label)
with c2:
    day, _ = date_hour_inputs(fc, "cz", with_hour=False)

if len(chosen) < 2:
    st.info("Select at least two zones to compare.")
    st.stop()

frames = {z: fc.zone_day(z, day) for z in chosen}

fig = go.Figure()
for color, (z, df) in zip(ZONE_COLORS, frames.items()):
    fig.add_trace(go.Scatter(x=df.hour, y=df.predicted_kwh, name=zone_label(z), mode="lines+markers",
                             line=dict(color=color, width=2), marker=dict(size=6)))
fig.update_layout(title=f"Predicted hourly demand · {day:%A %d %B %Y}", height=400,
                  plot_bgcolor="white", hovermode="x unified",
                  margin=dict(l=10, r=10, t=50, b=10),
                  legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                  xaxis=dict(title="Hour", tickmode="array", tickvals=list(range(0, 24, 2)),
                             ticktext=[hour_label(h) for h in range(0, 24, 2)]),
                  yaxis=dict(title="Predicted kWh", gridcolor="#e6e5e1"))
st.plotly_chart(fig, use_container_width=True)

info = fc.zones.set_index("zone")
summary = pd.DataFrame([{
    "Zone": zone_label(z),
    "Predicted daily demand (kWh)": round(df.predicted_kwh.sum(), 1),
    "Actual daily demand (kWh)": round(df.actual_kwh.sum(), 1),
    "Peak hour": hour_label(int(df.loc[df.predicted_kwh.idxmax(), "hour"])),
    "Peak demand (kWh)": round(df.predicted_kwh.max(), 1),
    "Average utilisation (%)": round(df.utilisation_pct.mean(), 1),
    "Chargers": int(info.at[z, "num_piles"]),
    "Fast chargers": int(info.at[z, "fast_count"]),
} for z, df in frames.items()])
st.subheader("Summary")
st.dataframe(summary, hide_index=True, use_container_width=True)

combined = pd.concat([display_table(df) for df in frames.values()], ignore_index=True)
csv_download(combined, f"zone_comparison_{day:%Y%m%d}.csv", "Download comparison (CSV)")
