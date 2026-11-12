import pandas as pd
import plotly.express as px
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
    zoom=9.3, height=540, mapbox_style="carto-positron")
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, x=0, title=None))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Busiest zones")
top = city.sort_values("predicted_kwh", ascending=False).head(10)
st.dataframe(display_table(top), hide_index=True, use_container_width=True)
csv_download(display_table(city.sort_values("zone")), f"city_forecast_{ts:%Y%m%d_%H00}.csv")
