"""Shared helpers for the Streamlit pages: styling, cached loaders, small widgets."""
import pandas as pd
import streamlit as st

from src.data_loader import missing_files

APP_NAME = "EV Charging Demand Prediction System"

# Ordinal blue ramp for demand levels (light -> dark = low -> very high)
LEVEL_COLORS = {"Low": "#86b6ef", "Moderate": "#3987e5", "High": "#1c5cab", "Very High": "#0d366b"}
PREDICTED_COLOR = "#2a78d6"
ACTUAL_COLOR = "#2b2b29"
ZONE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # fixed order for compared zones

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
div[data-testid="stMetric"] {
    background: #f3f6fb; border: 1px solid #e1e8f2; border-radius: 10px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] p {color: #52606d; font-size: 0.85rem;}
.hero {background: linear-gradient(120deg, #0d366b 0%, #2a78d6 100%); color: white;
       padding: 28px 32px; border-radius: 14px; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0 0 6px 0; font-size: 2rem;}
.hero p {color: #dbe8fb; margin: 0; font-size: 1.05rem;}
.badge {display: inline-block; padding: 4px 12px; border-radius: 999px; color: white;
        font-weight: 600; font-size: 0.95rem;}
.note {color: #52606d; font-size: 0.85rem;}
.card {border: 1px solid #e1e8f2; border-radius: 10px; padding: 16px 18px; height: 100%;}
.card h4 {margin-top: 0;}
</style>
"""


def setup_page(title):
    """Common page header and styling."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)


def level_badge(level):
    color = LEVEL_COLORS[level]
    return f'<span class="badge" style="background:{color}">{level}</span>'


@st.cache_resource(show_spinner="Loading forecasting model ...")
def _load_forecaster():
    from src.predict import Forecaster
    return Forecaster()


def get_forecaster():
    """Return the shared Forecaster or stop the page with a helpful message."""
    missing = missing_files()
    if missing:
        st.error("The UrbanEV dataset was not found. Place it in "
                 "`data/Dataset/UrbanEVDataset/UrbanEVDataset/` (see README).")
        with st.expander("Missing files"):
            st.code("\n".join(missing))
        st.stop()
    try:
        return _load_forecaster()
    except Exception as exc:   # models not trained, corrupted file, ...
        st.error(f"The forecasting model is not ready: {exc}")
        st.info("Run `python train.py` once in the project folder, then refresh this page.")
        st.stop()


@st.cache_data(show_spinner="Preparing forecasts ...")
def period_forecasts():
    return get_forecaster().period()


def zone_label(zone):
    return f"Zone {zone}"


def hour_label(h):
    return f"{h:02d}:00"


def forecast_period_note(fc):
    st.markdown(f'<p class="note">Forecast period: {fc.start:%d %b %Y} – {fc.end:%d %b %Y} '
                '(held-out data from the UrbanEV Shenzhen dataset; actual values are shown '
                'for comparison).</p>', unsafe_allow_html=True)


def date_hour_inputs(fc, key, default_day="2023-02-14", default_hour=18, with_hour=True):
    """Date (and hour) pickers restricted to the forecast period."""
    cols = st.columns(2 if with_hour else 1)
    day = cols[0].date_input("Date", value=pd.Timestamp(default_day).date(),
                             min_value=fc.start.date(), max_value=fc.end.date(), key=f"{key}_day")
    if not with_hour:
        return day, None
    hour = cols[1].selectbox("Hour", list(range(24)), index=default_hour,
                             format_func=hour_label, key=f"{key}_hour")
    return day, hour


def csv_download(df, filename, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", use_container_width=False)


def display_table(df):
    """User-friendly column names for forecast tables."""
    names = {"time": "Time", "zone": "Zone", "hour": "Hour", "predicted_kwh": "Predicted (kWh)",
             "actual_kwh": "Actual (kWh)", "level": "Demand level",
             "utilisation_pct": "Charger utilisation (%)", "price_per_kwh": "Price (RMB/kWh)"}
    cols = [c for c in names if c in df.columns]
    return df[cols].rename(columns=names)
