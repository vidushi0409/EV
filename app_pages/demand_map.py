import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import LEVELS
from src.ui import (LEVEL_COLORS, csv_download, date_inputs, display_table, get_service, kpi_row,
                    mode_banner, setup_page, zone_label)

setup_page("🗺️ Demand Map")
svc = get_service()
day, hour = date_inputs(svc, "map")
ts = pd.Timestamp(day) + pd.Timedelta(hours=int(hour))
try:
    city = svc.city_hour(ts)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
mode = city["mode"].iloc[0]
mode_banner(mode)

busy = city.level.isin(["High", "Very High"]).sum()
items = [("Predicted (city)", f"{city.predicted_kwh.sum() / 1000:,.1f} MWh")]
if mode == "dataset":
    items.append(("Actual (city)", f"{city.actual_kwh.sum() / 1000:,.1f} MWh"))
items += [("Busy zones", f"{busy} of {len(city)}", "High or Very High"),
          ("Avg utilisation", f"{city.utilisation_pct.mean():.0f}%")]
kpi_row(items)

city["Zone"] = city.zone.map(zone_label)
hover = {"predicted_kwh": ":,.1f", "utilisation_pct": ":.0f", "latitude": False, "longitude": False}
if mode == "dataset":
    hover["actual_kwh"] = ":,.1f"
fig = px.scatter_mapbox(
    city, lat="latitude", lon="longitude", size="predicted_kwh", color="level",
    category_orders={"level": LEVELS}, color_discrete_map=LEVEL_COLORS, size_max=26,
    hover_name="Zone", hover_data=hover,
    labels={"predicted_kwh": "Predicted (kWh)", "actual_kwh": "Actual (kWh)",
            "utilisation_pct": "Utilisation (%)", "level": "Level"},
    zoom=9.3, height=540)
# Label-free satellite basemap (free, no API key) + our own English district names
fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[{"below": "traces", "sourcetype": "raster", "sourceattribution": "Imagery © Esri",
                    "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/"
                               "World_Imagery/MapServer/tile/{z}/{y}/{x}"]}],
    margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, x=0, title=None))
DISTRICTS = {"Futian": (114.055, 22.541), "Luohu": (114.131, 22.548), "Nanshan": (113.930, 22.533),
             "Bao'an": (113.883, 22.555), "Longhua": (114.045, 22.696), "Longgang": (114.247, 22.720),
             "Guangming": (113.936, 22.748), "Pingshan": (114.346, 22.708), "Yantian": (114.236, 22.557),
             "Dapeng": (114.479, 22.588)}
fig.add_trace(go.Scattermapbox(lon=[v[0] for v in DISTRICTS.values()], lat=[v[1] for v in DISTRICTS.values()],
                               text=list(DISTRICTS), mode="text", hoverinfo="skip", showlegend=False,
                               textfont=dict(size=14, color="#ffffff")))
fig.update_traces(marker=dict(opacity=0.9), selector=dict(mode="markers"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Busiest zones")
top = display_table(city.sort_values("predicted_kwh", ascending=False).head(10))
keep = [c for c in ("Zone", "Predicted (kWh)", "Actual (kWh)", "Demand level", "Utilisation (%)",
                    "Price (RMB/kWh)") if c in top.columns]
st.dataframe(top[keep], hide_index=True, use_container_width=True,
             column_config={"Predicted (kWh)": st.column_config.NumberColumn(format="%.1f"),
                            "Actual (kWh)": st.column_config.NumberColumn(format="%.1f"),
                            "Utilisation (%)": st.column_config.NumberColumn(format="%.0f"),
                            "Price (RMB/kWh)": st.column_config.NumberColumn(format="%.2f")})
csv_download(display_table(city.sort_values("zone")), f"city_forecast_{ts:%Y%m%d_%H00}.csv")
