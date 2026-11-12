"""Shared helpers for the Streamlit pages: styling, cached loaders, small widgets."""
import pandas as pd
import streamlit as st

from src.data_loader import missing_files
from src.train_models import MODEL_NAMES

APP_NAME = "EV Charging Demand Prediction System"

# Ordinal blue ramp for demand levels (light -> dark = low -> very high)
LEVEL_COLORS = {"Low": "#86b6ef", "Moderate": "#3987e5", "High": "#1c5cab", "Very High": "#0d366b"}
PREDICTED_COLOR = "#2a78d6"
ACTUAL_COLOR = "#2b2b29"
ZONE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # fixed order for compared zones
MODEL_COLORS = {"Linear Regression": "#2a78d6", "Random Forest": "#eb6834", "LightGBM": "#1baf7a"}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
div[data-testid="stMetric"] {
    background: #f3f6fb; border: 1px solid #e1e8f2; border-radius: 10px; padding: 14px 16px;
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
.mode {border-radius: 10px; padding: 10px 14px; margin: 6px 0 14px 0; font-size: 0.95rem;}
.mode-dataset {background: #e8f6ef; border: 1px solid #b9e3cd;}
.mode-future {background: #fff6e5; border: 1px solid #f4dca8;}
</style>
"""


def setup_page(title):
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)


def level_badge(level):
    return f'<span class="badge" style="background:{LEVEL_COLORS[level]}">{level}</span>'


@st.cache_resource(show_spinner="Loading forecasting models ...")
def _load_service():
    from src.predict import DemandService
    return DemandService()


def get_service():
    """Return the shared DemandService or stop the page with a helpful message."""
    missing = missing_files()
    if missing:
        st.error("The UrbanEV dataset was not found. Place it in "
                 "`data/Dataset/UrbanEVDataset/UrbanEVDataset/` (see README).")
        with st.expander("Missing files"):
            st.code("\n".join(missing))
        st.stop()
    try:
        return _load_service()
    except Exception as exc:   # models not trained, corrupted file, ...
        st.error(f"The forecasting model is not ready: {exc}")
        st.info("Run `python train.py` once in the project folder, then refresh this page.")
        st.stop()


def zone_label(zone):
    return f"Zone {zone}"


def hour_label(h):
    return f"{h:02d}:00"


def mode_banner(mode):
    if mode == "dataset":
        st.markdown('<div class="mode mode-dataset">📊 <b>Dataset period</b> — this forecast was made '
                    'by models that never saw this month, and is compared with the <b>actual</b> '
                    'demand recorded in the dataset.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode mode-future">🔭 <b>Future date</b> — an <b>estimate</b> based on '
                    'demand patterns learnt from Sep 2022 – Feb 2023. No actual data exists for this '
                    'date, and growth in EV use after Feb 2023 is not included.</div>',
                    unsafe_allow_html=True)


def date_inputs(svc, key, default_day="2023-02-14", default_hour=18, with_hour=True):
    """Date (and hour) pickers covering 1 Oct 2022 up to today."""
    from src.predict import DATASET_START
    cols = st.columns(2 if with_hour else 1)
    day = cols[0].date_input("Date", value=pd.Timestamp(default_day).date(),
                             min_value=DATASET_START.date(), max_value=svc.max_date.date(),
                             key=f"{key}_day", format="DD/MM/YYYY")
    if not with_hour:
        return day, None
    hour = cols[1].selectbox("Hour", list(range(24)), index=default_hour,
                             format_func=hour_label, key=f"{key}_hour")
    return day, hour


def period_note():
    from src.predict import DATASET_END, DATASET_START
    st.caption(f"Pick any date from {DATASET_START:%d %b %Y}. Up to {DATASET_END:%d %b %Y} "
               "forecasts are compared with actual demand; later dates show pattern-based estimates.")


def csv_download(df, filename, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv")


def display_table(df):
    """User-friendly column names for forecast tables."""
    names = {"time": "Time", "zone": "Zone", "hour": "Hour", "predicted_kwh": "Predicted (kWh)",
             "actual_kwh": "Actual (kWh)", "level": "Demand level",
             "utilisation_pct": "Charger utilisation (%)", "price_per_kwh": "Price (RMB/kWh)"}
    cols = [c for c in names if c in df.columns and not (c == "actual_kwh" and df[c].isna().all())]
    out = df[cols].rename(columns=names)
    for m in MODEL_NAMES:
        if m in df.columns:
            out[f"{m} (kWh)"] = df[m].round(2).values
    return out


def line_layout(fig, title, y_title="kWh", height=380, hours_axis=False):
    fig.update_layout(title=title, height=height, hovermode="x unified", plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.14, x=1, xanchor="right"),
                      yaxis=dict(title=y_title, gridcolor="#e6e5e1"))
    if hours_axis:
        fig.update_xaxes(title="Hour", tickmode="array", tickvals=list(range(0, 24, 2)),
                         ticktext=[hour_label(h) for h in range(0, 24, 2)])
    return fig
