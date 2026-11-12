import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.train_models import MODEL_NAMES
from src.ui import (ACTUAL_COLOR, MODEL_COLORS, date_inputs, get_service, level_badge, line_layout,
                    mode_banner, period_note, setup_page, zone_label)

setup_page("🔮 Demand Forecast")
st.write("Choose a charging zone, a date and an hour to see the predicted charging demand.")

svc = get_service()
period_note()

with st.form("forecast_form"):
    zones = svc.zone_ids()
    zone = st.selectbox("Charging zone", zones, index=zones.index(102) if 102 in zones else 0,
                        format_func=zone_label)
    day, hour = date_inputs(svc, "fc")
    submitted = st.form_submit_button("Predict demand", type="primary")

if submitted:
    try:
        ts = pd.Timestamp(day) + pd.Timedelta(hours=int(hour))
        with st.spinner("Forecasting ..."):
            st.session_state["forecast"] = (svc.zone_hour(zone, ts), svc.zone_day(zone, day))
    except ValueError as exc:
        st.session_state.pop("forecast", None)
        st.error(str(exc))

if "forecast" in st.session_state:
    r, day_df = st.session_state["forecast"]
    st.subheader(f"{zone_label(r['zone'])} · {r['time']:%A %d %B %Y, %H:00}")
    mode_banner(r["mode"])
    best = r["best_model"]

    c1, c2, c3, c4 = st.columns(4)
    if r["mode"] == "dataset":
        c1.metric(f"Predicted demand ({best})", f"{r['predicted_kwh']:,.1f} kWh")
        c2.metric("Actual demand", f"{r['actual_kwh']:,.1f} kWh")
        b = r["badges"][best]
        side = "over-estimate" if r["predicted_kwh"] > r["actual_kwh"] else "under-estimate"
        c3.metric("Accuracy", f"{b['icon']} {b['label']}")
        c3.caption(f"{b['text']} ({side})")
    else:
        err = r["typical_error"][best]
        c1.metric(f"Estimated demand ({best})", f"{r['predicted_kwh']:,.1f} kWh")
        c2.metric("Typical error range", f"± {err:,.1f} kWh",
                  help="Average error of this model for this zone on held-out February 2023 data.")
        c3.metric("Charging price (typical)", f"{r['price_per_kwh']:.2f} RMB/kWh")
    with c4:
        st.markdown('<p class="note">Demand level (vs this zone\'s usual)</p>' + level_badge(r["level"]),
                    unsafe_allow_html=True)
        st.markdown(f'<p class="note">Estimated charger utilisation: {r["utilisation_pct"]:.0f}%</p>',
                    unsafe_allow_html=True)
    if r["mode"] == "future" and r.get("is_holiday"):
        st.info("This date is a public holiday in mainland China.")

    # ---------------------------------------------------------- all three models
    st.markdown("##### All three models")
    rows = []
    closest = (min(MODEL_NAMES, key=lambda m: abs(r["models"][m] - r["actual_kwh"]))
               if r["mode"] == "dataset" else None)
    for m in MODEL_NAMES:
        marks = (" ⭐" if m == best else "") + (" 🎯" if m == closest else "")
        row = {"Model": m + marks, "Prediction (kWh)": round(r["models"][m], 1)}
        if r["mode"] == "dataset":
            b = r["badges"][m]
            row.update({"Actual (kWh)": round(r["actual_kwh"], 1),
                        "Error (kWh)": round(r["models"][m] - r["actual_kwh"], 1),
                        "Accuracy": f"{b['icon']} {b['label']} ({b['text']})"})
        else:
            row["Typical error (± kWh)"] = round(r["typical_error"][m], 1)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if r["mode"] == "dataset":
        st.caption("⭐ = most accurate model **on average** over all forecast hours (Oct 2022 – Feb 2023); "
                   "it gives the headline number. 🎯 = the model that happened to be closest for **this "
                   "hour**. For a single hour any model can be closest, but over hundreds of thousands of "
                   "hours the ⭐ model makes the smallest errors.")
    else:
        st.caption("⭐ = the model with the smallest error on held-out February 2023 data for future-date "
                   "estimates; it gives the headline number.")

    # ---------------------------------------------------------- whole day chart
    fig = go.Figure()
    if r["mode"] == "dataset":
        fig.add_trace(go.Scatter(x=day_df.hour, y=day_df.actual_kwh, name="Actual",
                                 line=dict(color=ACTUAL_COLOR, width=3)))
    for m in MODEL_NAMES:
        fig.add_trace(go.Scatter(x=day_df.hour, y=day_df[m], name=m,
                                 line=dict(color=MODEL_COLORS[m], width=2,
                                           dash="solid" if m == best else "dot")))
    fig.add_vline(x=r["time"].hour, line_dash="dot", line_color="#8a8a85")
    line_layout(fig, "The whole day (selected hour marked)", hours_axis=True, height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Demand level compares the prediction with this zone's normal range: Low (bottom 25% "
               "of its hours), Moderate, High (top 25%), Very High (top 10%). Utilisation = predicted "
               "energy ÷ the zone's total charger power.")
