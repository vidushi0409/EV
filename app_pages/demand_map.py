import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import LEVELS
from src.ui import (LEVEL_COLORS, csv_download, date_hour_inputs, display_table,
                    forecast_period_note, get_forecaster, setup_page, zone_label)

setup_page("🗺️ Demand Map")
st.write("Predicted charging demand for every zone in Shenzhen at the chosen hour. "
         "Bubble size shows the predicted energy; colour shows how busy the zone is compared "
         "with its usual demand.")

fc = get_forecaster()
forecast_period_note(fc)
day, hour = date_hour_inputs(fc, "map")
ts = pd.Timestamp(day) + pd.Timedelta(hours=int(hour))
city = fc.city_hour(ts)

c1, c2, c3 = st.columns(3)
c1.metric("City-wide predicted demand", f"{city.predicted_kwh.sum() / 1000:,.1f} MWh")
c2.metric("Zones at High / Very High", f"{city.level.isin(['High', 'Very High']).sum()} of {len(city)}")
c3.metric("Average charger utilisation", f"{city.utilisation_pct.mean():.0f}%")

city["Zone"] = city.zone.map(zone_label)
fig = px.scatter_mapbox(
    city, lat="latitude", lon="longitude", size="predicted_kwh", color="level",
    category_orders={"level": LEVELS}, color_discrete_map=LEVEL_COLORS, size_max=28,
    hover_name="Zone", hover_data={"predicted_kwh": ":,.1f", "actual_kwh": ":,.1f",
                                   "utilisation_pct": ":.0f", "latitude": False,
                                   "longitude": False, "level": True},
    labels={"predicted_kwh": "Predicted (kWh)", "actual_kwh": "Actual (kWh)",
            "utilisation_pct": "Utilisation (%)", "level": "Demand level"},
    zoom=9.3, height=560, mapbox_style="carto-positron")
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                  legend=dict(orientation="h", y=1.02, x=0, title="Demand level"))
st.plotly_chart(fig, use_container_width=True)
st.caption("The street map background needs an internet connection; zone bubbles and "
           "forecasts work offline.")

st.subheader("Busiest zones at this hour")
top = city.sort_values("predicted_kwh", ascending=False).head(10)
st.dataframe(display_table(top), hide_index=True, use_container_width=True)
csv_download(display_table(city.sort_values("zone")), f"city_forecast_{ts:%Y%m%d_%H00}.csv",
             "Download all zones (CSV)")
