import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.train_models import MODEL_NAMES
from src.ui import (ACTUAL_COLOR, MODEL_COLORS, date_inputs, get_service, kpi_row, level_badge,
                    line_layout, mode_banner, setup_page, zone_label, zone_select)

setup_page("🔮 Demand Forecast")
svc = get_service()

with st.form("forecast_form", border=False):
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 0.9], vertical_alignment="bottom")
    zone = zone_select(svc, c1)
    day, hour = date_inputs(svc, "fc", cols=[c2, c3])
    submitted = c4.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    try:
        ts = pd.Timestamp(day) + pd.Timedelta(hours=int(hour))
        with st.spinner("Forecasting ..."):
            st.session_state["forecast"] = (svc.zone_hour(zone, ts), svc.zone_day(zone, day))
    except ValueError as exc:
        st.session_state.pop("forecast", None)
        st.error(str(exc))

if "forecast" not in st.session_state:
    st.info("Choose a zone, date and hour, then press **Predict**.")
    st.stop()

r, day_df = st.session_state["forecast"]
best = r["best_model"]
st.subheader(f"{zone_label(r['zone'])} · {r['time']:%a %d %b %Y, %H:00}")
mode_banner(r["mode"])

level = (level_badge(r["level"]), f"{r['utilisation_pct']:.0f}% charger utilisation",
         "Compared with this zone's usual demand")
if r["mode"] == "dataset":
    b = r["badges"][best]
    side = "over" if r["predicted_kwh"] > r["actual_kwh"] else "under"
    kpi_row([
        (f"Predicted · {best}", f"{r['predicted_kwh']:,.1f} kWh"),
        ("Actual", f"{r['actual_kwh']:,.1f} kWh"),
        ("Accuracy", f"{b['icon']} {b['label']}", f"{b['text']} ({side})"),
        ("Demand level", *level),
    ])
else:
    kpi_row([
        (f"Estimate · {best}", f"{r['predicted_kwh']:,.1f} kWh"),
        ("Typical error", f"± {r['typical_error'][best]:,.1f} kWh", "",
         "Average error of this model for this zone on held-out February 2023 data"),
        ("Price", f"{r['price_per_kwh']:.2f} RMB/kWh", "Public holiday" if r.get("is_holiday") else ""),
        ("Demand level", *level),
    ])

# ---------------------------------------------------------------- all three models
closest = (min(MODEL_NAMES, key=lambda m: abs(r["models"][m] - r["actual_kwh"]))
           if r["mode"] == "dataset" else None)
rows = []
for m in MODEL_NAMES:
    row = {"Model": m + (" ⭐" if m == best else "") + (" 🎯" if m == closest else ""),
           "Prediction (kWh)": r["models"][m]}
    if r["mode"] == "dataset":
        b = r["badges"][m]
        row.update({"Error (kWh)": r["models"][m] - r["actual_kwh"], "Accuracy": f"{b['icon']} {b['text']}"})
    else:
        row["Typical error (± kWh)"] = r["typical_error"][m]
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
             column_config={c: st.column_config.NumberColumn(format="%.1f")
                            for c in ("Prediction (kWh)", "Error (kWh)", "Typical error (± kWh)")})
legend = "⭐ best on average" + (" · 🎯 closest this hour" if closest else "")
st.markdown(f'<span class="subtle">{legend}</span>', unsafe_allow_html=True,
            help="The ⭐ model has the smallest error over all forecast hours. For a single hour "
                 "another model can be closer (🎯).")

# ---------------------------------------------------------------- whole day chart
fig = go.Figure()
if r["mode"] == "dataset":
    fig.add_trace(go.Scatter(x=day_df.hour, y=day_df.actual_kwh, name="Actual",
                             line=dict(color=ACTUAL_COLOR, width=3)))
for m in MODEL_NAMES:
    fig.add_trace(go.Scatter(x=day_df.hour, y=day_df[m], name=m,
                             line=dict(color=MODEL_COLORS[m], width=2,
                                       dash="solid" if m == best else "dot")))
fig.add_vline(x=r["time"].hour, line_dash="dot", line_color="#8a8a85")
line_layout(fig, "Whole day", hours_axis=True)
st.plotly_chart(fig, use_container_width=True)
