"""EV Charging Demand Prediction System - entry point.

Run with:  streamlit run app.py   (opens http://localhost:8501)
"""
import streamlit as st

from src.ui import APP_NAME

st.set_page_config(page_title=APP_NAME, page_icon="⚡", layout="wide")

pages = {
    "Forecasts": [
        st.Page("app_pages/home.py", title="Home", icon="🏠", default=True),
        st.Page("app_pages/demand_forecast.py", title="Demand Forecast", icon="🔮"),
        st.Page("app_pages/prediction_vs_actual.py", title="Prediction vs Actual", icon="🎯"),
        st.Page("app_pages/demand_map.py", title="Demand Map", icon="🗺️"),
        st.Page("app_pages/zone_insights.py", title="Zone Insights", icon="📈"),
        st.Page("app_pages/compare_zones.py", title="Compare Zones", icon="⚖️"),
    ],
    "About": [
        st.Page("app_pages/about_model.py", title="About the Model", icon="ℹ️"),
    ],
}

with st.sidebar:
    st.markdown(f"### ⚡ {APP_NAME}")
    st.caption("Hourly EV charging-demand forecasts for 275 zones in Shenzhen.")

st.navigation(pages).run()
