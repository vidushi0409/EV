"""Shared helpers for the Streamlit pages: styling, cached loaders, small widgets."""
import html

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
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1 {font-size: 1.9rem !important; margin-bottom: 0.2rem;}
h2, h3 {font-size: 1.2rem !important;}
.kpi {background: #f5f8fc; border: 1px solid #e1e8f2; border-radius: 12px;
      padding: 12px 16px; height: 100%; min-height: 96px;}
.kpi .l {font-size: 0.78rem; color: #52606d; margin-bottom: 4px; white-space: nowrap;
         overflow: hidden; text-overflow: ellipsis;}
.kpi .v {font-size: 1.45rem; font-weight: 700; color: #1f2933; line-height: 1.25;
         white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.kpi .s {font-size: 0.78rem; color: #52606d; margin-top: 4px;}
.hero {background: linear-gradient(120deg, #0d366b 0%, #2a78d6 100%); color: white;
       padding: 24px 28px; border-radius: 14px; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0 0 4px 0; font-size: 1.8rem !important;}
.hero p {color: #dbe8fb; margin: 0; font-size: 1rem;}
.badge {display: inline-block; padding: 3px 12px; border-radius: 999px; color: white;
        font-weight: 600; font-size: 0.95rem;}
.pill {display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.82rem;
       font-weight: 600; margin: 2px 0 12px 0;}
.pill-dataset {background: #e3f4ea; color: #1d6b43;}
.pill-future {background: #fdf1d8; color: #8a5a00;}
.card {border: 1px solid #e1e8f2; border-radius: 12px; padding: 14px 16px; height: 100%;}
.card h4 {margin: 0 0 4px 0; font-size: 1rem;}
.card p {margin: 0; color: #52606d; font-size: 0.88rem;}
.subtle {color: #6b7785; font-size: 0.82rem;}
</style>
"""


def setup_page(title):
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)


def kpi(col, label, value, sub="", tooltip=""):
    """Compact metric card. `value` may contain HTML (e.g. a badge)."""
    tip = f' title="{html.escape(tooltip)}"' if tooltip else ""
    icon = " ⓘ" if tooltip else ""
    col.markdown(f'<div class="kpi"{tip}><div class="l">{html.escape(label)}{icon}</div>'
                 f'<div class="v">{value}</div><div class="s">{sub or "&nbsp;"}</div></div>',
                 unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value, sub, tooltip) tuples (sub/tooltip optional)."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        kpi(col, *item)
    st.write("")


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


MODE_TIPS = {
    "dataset": "Predicted by models that never saw this month, and compared with the actual demand "
               "recorded in the dataset.",
    "future": "Typical demand for this zone, weekday and hour, learnt from Sep 2022 – Feb 2023. "
              "It is the same in every year: the data contains no information on growth after "
              "Feb 2023. No actual data exists for this date.",
}


def mode_banner(mode):
    text = ("📊 Dataset period · compared with actual" if mode == "dataset"
            else "🔭 Typical-pattern estimate · not year-specific")
    st.markdown(f'<span class="pill pill-{mode}" title="{html.escape(MODE_TIPS[mode])}">{text} ⓘ</span>',
                unsafe_allow_html=True)


def date_inputs(svc, key, default_day="2023-02-14", default_hour=18, with_hour=True, cols=None):
    """Date (and hour) pickers covering 1 Oct 2022 up to today."""
    from src.predict import DATASET_END, DATASET_START
    cols = cols or st.columns(2 if with_hour else 1)
    day = cols[0].date_input(
        "Date", value=pd.Timestamp(default_day).date(), min_value=DATASET_START.date(),
        max_value=svc.max_date.date(), key=f"{key}_day", format="DD/MM/YYYY",
        help=f"Up to {DATASET_END:%d %b %Y}: compared with actual demand. Later: estimate.")
    if not with_hour:
        return day, None
    hour = cols[1].selectbox("Hour", list(range(24)), index=default_hour,
                             format_func=hour_label, key=f"{key}_hour")
    return day, hour


def zone_select(svc, col=None, key=None, label="Zone"):
    zones = svc.zone_ids()
    target = col or st
    return target.selectbox(label, zones, index=zones.index(102) if 102 in zones else 0,
                            format_func=zone_label, key=key)


def csv_download(df, filename, label="⬇ Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv")


def display_table(df):
    """User-friendly column names for forecast tables."""
    names = {"time": "Time", "zone": "Zone", "hour": "Hour", "predicted_kwh": "Predicted (kWh)",
             "actual_kwh": "Actual (kWh)", "level": "Demand level",
             "utilisation_pct": "Utilisation (%)", "price_per_kwh": "Price (RMB/kWh)"}
    cols = [c for c in names if c in df.columns and not (c == "actual_kwh" and df[c].isna().all())]
    out = df[cols].rename(columns=names)
    for m in MODEL_NAMES:
        if m in df.columns:
            out[f"{m} (kWh)"] = df[m].round(2).values
    return out


def line_layout(fig, title, y_title="kWh", height=360, hours_axis=False):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=height,
                      hovermode="x unified", plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.14, x=1, xanchor="right"),
                      yaxis=dict(title=y_title, gridcolor="#e6e5e1"))
    if hours_axis:
        fig.update_xaxes(title=None, tickmode="array", tickvals=list(range(0, 24, 3)),
                         ticktext=[hour_label(h) for h in range(0, 24, 3)])
    return fig
