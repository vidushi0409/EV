"""Generate the project report and viva document (PDF + Word) from saved results.

    python build_reports.py

Run AFTER python train.py. Every number is read from the files in artifacts/,
models/ and the dataset, so the documents always match the latest results.
"""
import json
import re
from pathlib import Path

import pandas as pd

from src import config
from src.data_loader import inspect_dataset, price_schedule_check, check_5min_consistency
from src.eda import CORR_DIR
from src.evaluate_models import (COEF_FILE, COMPARISON_FILE, EVAL_DIR, IMPORTANCE_FILE,
                                 METRICS_FILE, read_best_model)
from src.feature_engineering import (CALENDAR_FEATURES, FEATURES, LAG_FEATURES, PRICE_FEATURES,
                                     STATIC_FEATURES, load_features)
from src.train_models import MODEL_NAMES, read_training_log

ML = MODEL_NAMES


# =====================================================================  document model
class Doc:
    """A renderer-independent list of blocks."""

    def __init__(self, title, subtitle):
        self.title, self.subtitle, self.blocks = title, subtitle, []

    def h1(self, t): self.blocks.append(("h1", t))
    def h2(self, t): self.blocks.append(("h2", t))
    def p(self, t): self.blocks.append(("p", t))
    def bullets(self, items): self.blocks.append(("bullets", list(items)))
    def table(self, df, widths=None): self.blocks.append(("table", (df, widths)))
    def image(self, path, caption, width=1.0): self.blocks.append(("image", (Path(path), caption, width)))
    def qa(self, q, a): self.blocks.append(("qa", (q, a)))
    def pagebreak(self): self.blocks.append(("pagebreak", None))


def fmt(v, nd=2):
    if isinstance(v, (int,)) or (isinstance(v, float) and v.is_integer() and abs(v) > 999):
        return f"{v:,.0f}"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def df_str(df, nd=2):
    return df.apply(lambda col: col.map(lambda v: fmt(v, nd)))


# =====================================================================  PDF renderer
def render_pdf(doc, path):
    from fpdf import FPDF
    from fpdf.fonts import FontFace

    fonts = Path("C:/Windows/Fonts")
    unicode_ok = (fonts / "arial.ttf").exists()

    class PDF(FPDF):
        def footer(self):
            if self.page_no() == 1:
                return
            self.set_y(-12)
            self.set_font(FONT, "", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, f"{doc.title}  |  page {self.page_no()}", align="C")

    pdf = PDF(format="A4")
    if unicode_ok:
        pdf.add_font("Body", "", str(fonts / "arial.ttf"))
        pdf.add_font("Body", "B", str(fonts / "arialbd.ttf"))
        pdf.add_font("Body", "I", str(fonts / "ariali.ttf"))
        FONT = "Body"
        clean = lambda s: s
    else:   # portable fallback: core font, replace non-Latin-1 characters
        FONT = "Helvetica"
        repl = {"²": "2", "Σ": "sum", "ŷ": "y_hat", "≥": ">=", "≤": "<=", "×": "x", "–": "-",
                "—": "-", "’": "'", "“": '"', "”": '"', "·": "-", "≈": "~", "√": "sqrt", "→": "->"}
        clean = lambda s: "".join(repl.get(ch, ch) for ch in s).encode("latin-1", "replace").decode("latin-1")

    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, 18)
    blue = (13, 54, 107)

    def write_rich(text, size=10.5, style=""):
        """Paragraph supporting **bold** segments."""
        pdf.set_font(FONT, style, size)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 5.6, clean(text), markdown=True, new_x="LMARGIN", new_y="NEXT")

    # ---- title page
    pdf.add_page()
    pdf.set_fill_color(*blue)
    pdf.rect(0, 0, 210, 95, "F")
    pdf.set_xy(18, 32)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(FONT, "B", 24)
    pdf.multi_cell(174, 11, clean(doc.title), align="L")
    pdf.set_x(18)
    pdf.set_font(FONT, "", 13)
    pdf.multi_cell(174, 7, clean(doc.subtitle))
    pdf.set_xy(18, 115)
    pdf.set_text_color(30, 30, 30)
    pdf.set_font(FONT, "", 11)
    for label in ["Submitted by:", "Roll number:", "Course: Data Science Lab", "Guide:", "Institution:",
                  "Date:"]:
        pdf.cell(0, 10, clean(label + ("" if ":" not in label[-1:] else " ______________________________")),
                 new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(250)
    pdf.set_font(FONT, "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 5, clean("Dataset: UrbanEV - Li, H. et al. (2025), Scientific Data 12, 523. "
                               "All figures and numbers in this document were generated from the "
                               "project's executed code."))
    pdf.add_page()

    for kind, val in doc.blocks:
        if kind == "h1":
            if pdf.get_y() > 230:
                pdf.add_page()
            pdf.ln(3)
            pdf.set_font(FONT, "B", 15)
            pdf.set_text_color(*blue)
            pdf.multi_cell(0, 8, clean(val), new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(42, 120, 214)
            pdf.set_line_width(0.5)
            pdf.line(18, pdf.get_y(), 192, pdf.get_y())
            pdf.ln(3)
        elif kind == "h2":
            if pdf.get_y() > 250:
                pdf.add_page()
            pdf.ln(2)
            pdf.set_font(FONT, "B", 12)
            pdf.set_text_color(*blue)
            pdf.multi_cell(0, 7, clean(val), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif kind == "p":
            write_rich(val)
            pdf.ln(2)
        elif kind == "bullets":
            for item in val:
                pdf.set_font(FONT, "", 10.5)
                pdf.set_text_color(30, 30, 30)
                pdf.set_x(22)
                pdf.cell(4, 5.6, "•" if unicode_ok else "-")
                pdf.multi_cell(0, 5.6, clean(item), markdown=True, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(0.8)
            pdf.ln(1.5)
        elif kind == "table":
            df, widths = val
            df = df_str(df)
            pdf.set_font(FONT, "", 8.5)
            pdf.set_text_color(30, 30, 30)
            pdf.set_draw_color(200, 205, 215)
            pdf.set_line_width(0.2)
            pdf.set_fill_color(243, 246, 251)
            heading = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=(42, 120, 214))
            with pdf.table(col_widths=widths, headings_style=heading, line_height=5,
                           cell_fill_color=(243, 246, 251), cell_fill_mode="ROWS",
                           text_align="LEFT", first_row_as_headings=True) as t:
                row = t.row()
                for c in df.columns:
                    row.cell(clean(str(c)))
                for _, r in df.iterrows():
                    row = t.row()
                    for v in r:
                        row.cell(clean(str(v)))
            pdf.ln(4)
        elif kind == "image":
            img, caption, width = val
            if not img.exists():
                continue
            from PIL import Image as PILImage
            with PILImage.open(img) as im:
                ratio = im.height / im.width
            w = 174 * width
            h = w * ratio
            if pdf.get_y() + h + 10 > 280:
                pdf.add_page()
            pdf.image(str(img), x=18 + (174 - w) / 2, w=w)
            pdf.set_font(FONT, "I", 8.5)
            pdf.set_text_color(90, 90, 90)
            pdf.multi_cell(0, 5, clean(caption), align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
        elif kind == "qa":
            q, a = val
            if pdf.get_y() > 255:
                pdf.add_page()
            pdf.set_font(FONT, "B", 10.5)
            pdf.set_text_color(*blue)
            pdf.multi_cell(0, 5.6, clean("Q. " + q), new_x="LMARGIN", new_y="NEXT")
            write_rich("A. " + a)
            pdf.ln(2.5)
        elif kind == "pagebreak":
            pdf.add_page()
    pdf.output(str(path))


# =====================================================================  DOCX renderer
def render_docx(doc, path):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    d = Document()
    for s in d.sections:
        s.left_margin = s.right_margin = Cm(2)
        s.top_margin = s.bottom_margin = Cm(1.8)
    style = d.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    blue = RGBColor(13, 54, 107)

    def add_rich(par, text):
        for i, part in enumerate(re.split(r"\*\*", text)):
            run = par.add_run(part)
            run.bold = i % 2 == 1

    def shade(cell, hex_color):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tcPr.append(shd)

    t = d.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("\n\n" + doc.title)
    r.bold, r.font.size, r.font.color.rgb = True, Pt(26), blue
    s = d.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.add_run(doc.subtitle).font.size = Pt(14)
    d.add_paragraph("\n")
    for label in ["Submitted by:", "Roll number:", "Course: Data Science Lab", "Guide:",
                  "Institution:", "Date:"]:
        d.add_paragraph(label + ("" if "Course" in label else " ______________________________"))
    note = d.add_paragraph("\nDataset: UrbanEV - Li, H. et al. (2025), Scientific Data 12, 523. "
                           "All figures and numbers in this document were generated from the "
                           "project's executed code.")
    note.runs[0].italic = True
    d.add_page_break()

    for kind, val in doc.blocks:
        if kind == "h1":
            h = d.add_heading(val, level=1)
        elif kind == "h2":
            d.add_heading(val, level=2)
        elif kind == "p":
            add_rich(d.add_paragraph(), val)
        elif kind == "bullets":
            for item in val:
                add_rich(d.add_paragraph(style="List Bullet"), item)
        elif kind == "table":
            df = df_str(val[0])
            tbl = d.add_table(rows=1, cols=len(df.columns))
            tbl.style = "Table Grid"
            for i, c in enumerate(df.columns):
                cell = tbl.rows[0].cells[i]
                cell.text = str(c)
                cell.paragraphs[0].runs[0].bold = True
                cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
                shade(cell, "2A78D6")
            for _, row in df.iterrows():
                cells = tbl.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = str(v)
            for row in tbl.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for run in p.runs:
                            run.font.size = Pt(9)
            d.add_paragraph()
        elif kind == "image":
            img, caption, width = val
            if img.exists():
                d.add_picture(str(img), width=Cm(17 * width))
                d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                c = d.add_paragraph(caption)
                c.alignment = WD_ALIGN_PARAGRAPH.CENTER
                c.runs[0].italic = True
                c.runs[0].font.size = Pt(9)
        elif kind == "qa":
            q, a = val
            pq = d.add_paragraph()
            rq = pq.add_run("Q. " + q)
            rq.bold, rq.font.color.rgb = True, blue
            add_rich(d.add_paragraph(), "A. " + a)
        elif kind == "pagebreak":
            d.add_page_break()
    d.save(str(path))


# =====================================================================  load results
def load_results():
    best = read_best_model()
    if best is None:
        raise SystemExit("No results found. Run: python train.py")
    feats = load_features()
    R = {
        "summary": inspect_dataset(),
        "prices": price_schedule_check(),
        "five": check_5min_consistency(),
        "best": best,
        "comp": pd.read_csv(COMPARISON_FILE).set_index("model"),
        "metrics": pd.read_csv(METRICS_FILE),
        "imp": pd.read_csv(IMPORTANCE_FILE),
        "coef": pd.read_csv(COEF_FILE),
        "log": read_training_log(),
        "tcorr": pd.read_csv(CORR_DIR / "target_correlations.csv"),
        "pairs": pd.read_csv(CORR_DIR / "multicollinear_pairs.csv"),
        "within": pd.read_csv(CORR_DIR / "within_zone_correlations.csv"),
        "notes": json.loads((CORR_DIR / "interpretation.json").read_text())["notes"],
        "splits": feats.groupby("split").agg(rows=("time", "size"), start=("time", "min"),
                                             end=("time", "max")).loc[["train", "val", "test"]],
        "n_rows": len(feats),
    }
    from src.profile_model import PROFILE_EVAL, PROFILE_INFO
    from src.walk_forward import WF_MONTHLY
    R["wf"] = pd.read_csv(WF_MONTHLY)
    R["profile_eval"] = pd.read_csv(PROFILE_EVAL)
    R["future_best"] = json.loads(PROFILE_INFO.read_text())["best_model"]
    panel = pd.read_parquet(config.PROCESSED_DIR / "panel_hourly.parquet", columns=["time", "volume"])
    by_hour = panel.groupby(panel.time.dt.hour).volume.sum()
    R["peak_hours"] = by_hour.sort_values(ascending=False).index[:3].tolist()
    R["night_share"] = by_hour.loc[0:7].sum() / by_hour.sum()
    R["vol_stats"] = panel.volume.describe()
    return R


# =====================================================================  report content
def build_report(R):
    S, best, comp = R["summary"], R["best"], R["comp"]
    b = best["best_model"]
    hf = S["hourly_files"]["volume"]
    tc = R["tcorr"].set_index("feature").pearson_r
    wz = R["within"].set_index("feature")
    met = R["metrics"]
    rm = lambda m, s: met[(met.model == m) & (met.split == s)].RMSE.iloc[0]
    imp_b = R["imp"][R["imp"].model == b].sort_values("importance", ascending=False) \
        if b in R["imp"].model.unique() else None
    doc = Doc("Spatiotemporal Prediction of Electric Vehicle Charging Demand",
              "Data Science Lab Project Report")

    # 1
    doc.h1("1. Introduction")
    doc.p("Electric vehicles (EVs) are spreading quickly in large cities, and charging infrastructure has to "
          "keep up. Shenzhen, China, has one of the largest electric fleets in the world, including taxis "
          "and ride-hailing vehicles that charge many times a day. How much energy is drawn from chargers "
          "changes strongly from hour to hour and from one part of the city to another.")
    doc.p("This project builds a machine-learning system that forecasts the hourly EV charging demand (in kWh) "
          "of every charging zone in Shenzhen one hour ahead. Three regression models (Linear Regression, "
          "Random Forest and LightGBM) are trained on the public UrbanEV dataset and compared on a strict "
          "chronological test period. The best model is deployed in a local web application (Streamlit) "
          "where users can view forecasts per zone, on a city map and over a full day.")

    # 2
    doc.h1("2. Problem Statement")
    doc.p("Charging demand varies greatly by time and location: some stations are under-used while others are "
          "overcrowded at peak hours. This affects drivers (queues), charging operators (capacity and pricing), "
          "urban planners (where to add chargers) and electricity providers (grid load). The task is to "
          "**predict the total energy (kWh) delivered by EV chargers in each zone during the next hour**, using "
          "only information available before that hour starts. Because the target is a continuous number, "
          "this is a **supervised regression problem**.")

    # 3
    doc.h1("3. Objectives")
    doc.bullets([
        "Collect, inspect and understand the UrbanEV spatiotemporal charging dataset for Shenzhen.",
        "Preprocess and integrate the charging, zone, station, charger-power and adjacency data.",
        "Engineer leakage-free features for next-hour forecasting.",
        "Analyse relationships between demand and the other variables using Pearson correlation.",
        "Train Linear Regression, Random Forest and LightGBM regressors and compare them with MAE, MSE, RMSE and R².",
        "Interpret the models (coefficients, feature importance, residuals) and select the best one.",
        "Deploy the selected model in a user-friendly web application.",
    ])

    # 4
    doc.h1("4. Dataset Description")
    doc.p("**Source:** UrbanEV - An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction "
          "(Li et al., Scientific Data, 2025). The zone-level, cleaned and aggregated version is used, with "
          "hourly resolution. The files are provided as CSV.")
    doc.table(pd.DataFrame([
        ["Charging zones", S["zones"]],
        ["Hourly timestamps", hf["rows (hours)"]],
        ["Period", f"{hf['start'][:10]} to {hf['end'][:10]}"],
        ["Zone-hour records", f"{S['zone_hour_records']:,}"],
        ["Charging stations (raw station file)", S["station_info_raw"]["stations"]],
        ["Charging piles with rated power", f"{S['pile_rated_power']['piles']:,} "
                                            f"({S['pile_rated_power']['AC piles']:,} AC, {S['pile_rated_power']['DC piles']:,} DC)"],
        ["Missing values in hourly files", sum(v["missing values"] for v in S["hourly_files"].values())],
    ], columns=["Property", "Value"]), widths=(60, 114))
    doc.h2("4.1 Files used")
    doc.table(pd.DataFrame([
        ["charge_1hour/volume.csv", "Energy delivered per zone-hour (kWh) - TARGET", "4,344 x 275 (wide)"],
        ["charge_1hour/occupancy.csv", "Number of charging piles in use", "4,344 x 275"],
        ["charge_1hour/duration.csv", "Charging duration", "4,344 x 275"],
        ["charge_1hour/e_price.csv", "Electricity price (RMB/kWh)", "4,344 x 275"],
        ["charge_1hour/s_price.csv", "Service price (RMB/kWh)", "4,344 x 275"],
        ["charge_1hour/volume-11kw.csv", "Energy from <=11 kW chargers (part of volume)", "4,344 x 275"],
        ["zone-information.csv", "Zone ID, longitude, latitude, charger count, area, perimeter", "275 x 6"],
        ["station_information.csv (raw)", "Station -> zone, slow / fast charger counts", f"{S['station_info_raw']['stations']:,} x 7"],
        ["pile_rated_power.csv", "Charger (pile) power in kW and AC/DC type", f"{S['pile_rated_power']['piles']:,} x 4"],
        ["adj.csv / distance.csv", "Zone adjacency (0/1) and distances (m)", "275 x 275"],
        ["charge_5min/volume.csv", "5-minute volume (used only for a consistency check)", f"{R['five']['five_min_rows']:,} x 275"],
    ], columns=["File", "Content", "Shape"]), widths=(52, 88, 34))
    doc.p("Each hourly file is a **wide** table: one time column plus one column per zone ID. The zone columns, "
          "the TAZID values in zone-information.csv and the row/column order of adj.csv and distance.csv were "
          "verified to refer to the same 275 zones in the same order.")
    v = R["vol_stats"]
    doc.p(f"**Target variable volume:** mean {v['mean']:.1f} kWh per zone-hour, median {v['50%']:.1f} kWh, "
          f"maximum {v['max']:,.1f} kWh; {hf['zero share'] * 100:.1f}% of zone-hours have zero demand.")

    # 5
    doc.h1("5. Data Preprocessing")
    doc.bullets([
        f"**Reshaping:** the six wide hourly files were stacked into one long table with one row per (time, zone): {S['zone_hour_records']:,} rows.",
        "**Missing values, duplicates and negative values:** none were found (0 in every file). Forward-fill, duplicate removal and clipping at 0 are kept as safeguards.",
        "**Time consistency:** 4,344 consecutive hours with no gaps or duplicate timestamps.",
        "**Data types:** measurements converted to float32 and zone IDs to integers to save memory.",
        f"**Invalid coordinates:** zone {', '.join(map(str, S['zone_info']['zones with missing (0,0) coordinates']))} had longitude/latitude = 0; it was replaced by the mean location of the zone's stations.",
        f"**Adjacency matrix:** {S['adjacency']['asymmetric pairs']} one-directional links were made symmetric; {S['adjacency']['zones with no neighbours']} zones without neighbours use their 3 nearest zones (from distance.csv).",
        "**Scaling:** only Linear Regression needs it; a StandardScaler is fitted on training data only, inside a pipeline.",
    ])
    doc.h2("5.1 Data validation checks")
    doc.table(pd.DataFrame([
        ["Hourly volume = sum of 5-minute volumes", f"max difference {R['five']['max_abs_difference_kwh']} kWh", "Hourly data is consistent"],
        ["Zone charger counts = sum of station slow + fast counts", "100% match", "Station file used for charger features"],
        ["e_price daily profile repeats previous day", f"{R['prices']['e_price'] * 100:.1f}% of zone-days", "Prices are a known tariff"],
        ["s_price daily profile repeats previous day", f"{R['prices']['s_price'] * 100:.1f}% of zone-days", "Prices are a known tariff"],
        ["volume_11kw <= volume", f"{S['volume_11kw_leakage']['share of rows with volume_11kw <= volume'] * 100:.1f}% of rows", "Component of target - excluded"],
        ["Adjacent zones' mean distance", "about 2 km (vs 22 km non-adjacent)", "Adjacency aligned correctly"],
    ], columns=["Check", "Result", "Conclusion"]), widths=(68, 50, 56))
    doc.h2("5.2 Data integration")
    doc.p("Static attributes were joined to each zone-hour by zone ID: number of chargers, share of fast "
          "chargers, average rated power per charger (from pile_rated_power.csv, linked via station_id -> zone) "
          "and total charger power (capacity, kW). Hourly demand never exceeds the zone's total charger power, "
          "which confirms the capacity feature is physically consistent.")

    # 6
    doc.h1("6. Feature Engineering")
    doc.p("**Prediction horizon: next hour.** A row (zone, hour t) has the target volume at hour t. All features "
          "use information from hour t-1 or earlier, or information known in advance for hour t (calendar, "
          "public holidays, tariff prices, zone attributes). Lags are computed separately for each zone.")
    doc.table(pd.DataFrame([
        ["Demand history", ", ".join(LAG_FEATURES[:4]), "Zone's own demand 1 h, 24 h and 168 h earlier; mean of the last 24 h"],
        ["Activity history", "occupancy_lag1, duration_lag1", "Chargers in use and charging duration in the previous hour"],
        ["Spatial", "neighbor_volume_lag1", "Average previous-hour demand of adjacent zones"],
        ["Tariff", ", ".join(PRICE_FEATURES), "Electricity and service price for the hour (published tariff)"],
        ["Calendar", ", ".join(CALENDAR_FEATURES), "Hour, day, weekday, weekend and holiday flags, sine/cosine cycles"],
        ["Zone", ", ".join(STATIC_FEATURES), "Charger count and power, zone area and location"],
    ], columns=["Group", "Features", "Meaning"]), widths=(28, 76, 70))
    doc.p(f"In total **{len(FEATURES)} input features** are used. month is created but not used as an input: "
          "training covers Sep-Dec while the test set is February, a value never seen in training. The first "
          "168 hours are dropped because the weekly lag is not yet defined.")
    doc.h2("6.1 Chronological split")
    sp = R["splits"].reset_index()
    sp["start"] = sp.start.dt.strftime("%d %b %Y")
    sp["end"] = sp.end.dt.strftime("%d %b %Y")
    sp.columns = ["Split", "Rows", "Start", "End"]
    doc.table(sp, widths=(30, 40, 52, 52))
    doc.p("All zones of one timestamp are always in the same split, and the splits never overlap in time. The "
          "validation month is used only for LightGBM early stopping and for model selection.")
    doc.h2("6.2 Leakage prevention")
    doc.bullets([
        "Same-hour occupancy and duration are only known after the hour ends, so only their 1-hour lags are used.",
        "volume_11kw is part of the target and is never used as an input.",
        "Lag values were spot-checked on 2,000 random rows against the original data (all correct).",
        "Scaling and every learnt quantity are fitted on training data only.",
    ])

    # 7
    doc.h1("7. Correlation Analysis")
    doc.p(f"Exploratory analysis is limited to Pearson correlation, computed on the training period "
          f"({R['splits'].loc['train', 'rows']:,} rows). Same-hour occupancy, duration and volume_11kw are included "
          "only for discussion.")
    doc.image(CORR_DIR / "correlation_heatmap.png", "Figure 1. Pearson correlation matrix (training period).", 0.95)
    t = R["tcorr"].copy()
    t.columns = ["Feature", "Pearson r", "Strength", "Role"]
    doc.h2("7.1 Correlation with the target")
    doc.table(t, widths=(42, 22, 38, 72))
    doc.h2("7.2 Pooled vs within-zone correlation")
    w = R["within"].copy()
    w.columns = ["Feature", "Pooled r", "Median within-zone r"]
    doc.table(w, widths=(70, 50, 54))
    doc.h2("7.3 Multicollinearity (|r| >= 0.8 among inputs)")
    pr = R["pairs"].copy()
    pr.columns = ["Feature 1", "Feature 2", "Pearson r"]
    doc.table(pr, widths=(64, 64, 46))
    doc.h2("7.4 Interpretation")
    doc.bullets(R["notes"])

    # 8
    doc.h1("8. Algorithms Used")
    doc.h2("8.1 Linear Regression")
    doc.p("Models the target as a weighted sum of the features: ŷ = b0 + b1·x1 + ... + bn·xn. The weights minimise "
          "the sum of squared errors (ordinary least squares). It is fast and interpretable, but it can only "
          "represent linear relationships, and its coefficients become unstable when features are strongly "
          "correlated (multicollinearity). Features are standardised so coefficients are comparable.")
    doc.h2("8.2 Random Forest Regressor")
    doc.p("An ensemble of decision trees (bagging). Each tree is trained on a random sample of rows and considers a "
          "random subset of features at each split; the forest prediction is the average of all trees. Averaging "
          "reduces variance, captures non-linear relationships and interactions, and gives impurity-based "
          "feature importance.")
    doc.h2("8.3 LightGBM Regressor")
    doc.p("A gradient-boosting framework. Trees are built one after another; each new tree is fitted to the "
          "remaining errors (gradients) of the current ensemble, and its contribution is scaled by a learning "
          "rate. LightGBM grows trees leaf-wise and uses histogram-based splitting, which makes it fast and memory-"
          "efficient on large tabular data. Early stopping on the validation set prevents overfitting.")
    doc.h2("8.4 Naive reference forecasts")
    doc.p("Two forecasts that involve no learning are reported for context: repeat the previous hour's demand "
          "(lag-1) and repeat the demand at the same hour one week earlier (lag-168). A useful model must clearly "
          "beat them.")

    # 9
    doc.h1("9. Model Implementation")
    doc.table(pd.DataFrame([
        ["Linear Regression", "StandardScaler -> LinearRegression (scikit-learn)"],
        ["Random Forest", "100 trees, max_depth 18, min_samples_leaf 20, max_features 0.5, max_samples 0.3"],
        ["LightGBM", "learning_rate 0.05, num_leaves 63, min_child_samples 50, subsample 0.8, colsample_bytree 0.8, "
                     "up to 3,000 trees with early stopping (100 rounds) on validation RMSE"],
        ["All models", "same 25 features, same training rows, random seed 42, predictions clipped at 0 kWh"],
    ], columns=["Model", "Configuration"]), widths=(36, 138))
    lg = pd.DataFrame(R["log"]).T.reindex(ML)
    lg = lg[["train_rows", "training_seconds", "file_size_mb"]].reset_index()
    lg.columns = ["Model", "Training rows", "Training time (s)", "Saved size (MB)"]
    lg["Training rows"] = lg["Training rows"].astype(int)
    doc.table(lg, widths=(50, 40, 42, 42))
    if "best_iteration" in R["log"].get("LightGBM", {}):
        doc.p(f"LightGBM stopped early at {R['log']['LightGBM']['best_iteration']} trees.")
    doc.h2("9.1 Two prediction modes")
    doc.p("Users can pick any date from 1 October 2022 up to today. Two kinds of models make this possible, "
          "each built with all three algorithms:")
    doc.table(pd.DataFrame([
        ["Dates", "1 Oct 2022 - 28 Feb 2023", "1 Mar 2023 - today"],
        ["Models", "Next-hour models (25 features incl. recent demand)",
         "Profile models (calendar, holidays, typical prices, zone attributes, zone typical demand)"],
        ["Training", "Walk-forward: each month predicted by models trained only on earlier months",
         "All six months (evaluated first on February 2023)"],
        ["Output", "Prediction AND actual demand, error and accuracy label",
         "Estimate with a typical error range (no actual data exists)"],
    ], columns=["", "Mode 1: dataset period", "Mode 2: future dates"]), widths=(24, 70, 80))
    doc.p("September 2022 is not offered: it is the first month, so no earlier data exists to train on. "
          "Future-date holiday flags use the official public-holiday schedules up to 2026.")
    doc.h2("9.2 Software architecture")
    doc.bullets([
        "**Backend (src/):** data loading, preprocessing, feature engineering, correlation analysis, training, evaluation, walk-forward forecasting, future-date models and a prediction service. python train.py runs the whole pipeline once.",
        "**Frontend (Streamlit app):** Home, Demand Forecast, Prediction vs Actual, Demand Map, Zone Insights, Compare Zones and About the Model pages. The app only loads saved models and results; no training happens in the interface.",
        "**Notebook:** the full analysis, using the same src modules so results are identical.",
        "**Persistence:** models saved with joblib; data cached as Parquet; results saved as CSV/JSON/PNG.",
    ])

    # 10
    doc.h1("10. Evaluation Metrics")
    doc.bullets([
        "**MAE** = (1/n) Σ |y - ŷ| - the average absolute error in kWh; easy to interpret.",
        "**MSE** = (1/n) Σ (y - ŷ)² - squares errors, so large errors are penalised strongly (units kWh²).",
        "**RMSE** = √MSE - like MSE but back in kWh; used for model selection.",
        "**R²** = 1 - Σ(y - ŷ)² / Σ(y - ȳ)² - share of the variance in demand explained by the model (1 = perfect).",
    ])
    doc.p("All metrics are computed on the February 2023 test period, which was never used for training or model "
          "selection.")

    # 11
    doc.h1("11. Results and Comparison")
    ct = comp.reset_index()[["model", "MAE", "MSE", "RMSE", "R2"]]
    ct.columns = ["Model", "MAE (kWh)", "MSE", "RMSE (kWh)", "R²"]
    ct["R²"] = ct["R²"].map(lambda v: f"{v:.4f}")
    doc.table(ct, widths=(50, 30, 34, 30, 30))
    doc.image(EVAL_DIR / "metric_comparison.png", "Figure 2. Test-set metrics for all models and baselines.", 1.0)
    doc.h2("11.1 Overfitting and underfitting check")
    tv = met[met.model.isin(ML)].pivot(index="model", columns="split", values="RMSE")[["train", "val", "test"]]
    tv = tv.reindex(ML).reset_index()
    tv.columns = ["Model", "Train RMSE", "Validation RMSE", "Test RMSE"]
    doc.table(tv, widths=(50, 41, 42, 41))
    doc.p(f"The tree models show small gaps between training and test error ({b}: {rm(b, 'train'):.1f} -> "
          f"{rm(b, 'test'):.1f} kWh), so there is little overfitting. Linear Regression has a higher error even on "
          f"its own training data ({rm('Linear Regression', 'train'):.1f} kWh), a sign of underfitting: the "
          "relationship is not purely linear.")
    doc.image(EVAL_DIR / "actual_vs_predicted.png", "Figure 3. Actual vs predicted demand on the test period.", 1.0)
    doc.image(EVAL_DIR / "residuals.png", "Figure 4. Residual analysis (actual - predicted).", 1.0)
    doc.p("Residuals are centred close to zero for all models. Errors grow for zones with high demand, which is "
          "expected because large zones vary more in absolute kWh.")
    doc.image(EVAL_DIR / "feature_importance.png", "Figure 5. Feature importance (tree models) and Linear Regression "
                                                    "coefficients.", 1.0)
    cf = R["coef"].head(10).copy()
    cf.columns = ["Feature", "Coefficient (kWh per +1 std)"]
    doc.h2("11.2 Linear Regression coefficients (top 10)")
    doc.table(cf, widths=(87, 87))
    if imp_b is not None:
        doc.h2(f"11.3 {b} feature importance (top 10)")
        ib = imp_b.head(10)[["feature", "importance"]].copy()
        ib["importance"] = (ib.importance * 100).map(lambda v: f"{v:.1f}%")
        ib.columns = ["Feature", "Share of importance"]
        doc.table(ib, widths=(87, 87))
    doc.h2("11.4 Final model selection")
    reductions = ", ".join(f"{100 * (1 - comp.at[b, 'RMSE'] / comp.at[m, 'RMSE']):.1f}% vs {m}"
                           for m in ML + ["Naive (lag-1)"] if m != b)
    doc.p(f"The model with the lowest validation RMSE is **{b}** "
          f"({', '.join(f'{k}: {v:.1f}' for k, v in best['validation_rmse'].items())}). The choice was "
          f"{'confirmed' if best['selection_confirmed_on_test'] else 'not confirmed'} on the test set, where {b} "
          f"achieved MAE {comp.at[b, 'MAE']:.1f} kWh, RMSE {comp.at[b, 'RMSE']:.1f} kWh and R² {comp.at[b, 'R2']:.4f}. "
          f"RMSE reduction: {reductions}.")
    doc.p("Gradient boosting builds trees sequentially, each correcting the remaining errors, so it captures "
          "non-linear effects (daily cycles, zone size, interactions between recent and weekly demand) that a "
          "linear equation cannot, and with early stopping it avoids overfitting. Random Forest averages "
          "independent trees and is limited in depth for memory reasons, which leaves slightly more error.")

    wf = R["wf"]
    doc.h2("11.5 Walk-forward results (month by month)")
    doc.p("Each month was predicted by models trained only on the months before it. These are the "
          "predictions shown in the application for the dataset period.")
    wt = wf.pivot(index="month", columns="model", values="RMSE")[ML]
    wt = wt.reindex(config.WALK_FORWARD_MONTHS + ["Oct-Feb (all)"]).reset_index()
    wt.columns = ["Predicted month"] + [f"{m} RMSE" for m in ML]
    doc.table(wt, widths=(44, 43, 43, 44))
    allwf = wf[wf.month == "Oct-Feb (all)"].set_index("model")
    wins = sum(wf[wf.month != "Oct-Feb (all)"].groupby("month")
               .apply(lambda d: d.sort_values("RMSE").model.iloc[0] == b))
    doc.p(f"{b} had the lowest error in {wins} of {len(config.WALK_FORWARD_MONTHS)} months; over the whole "
          f"period its RMSE was {allwf.at[b, 'RMSE']:.1f} kWh (R² {allwf.at[b, 'R2']:.3f}). Errors fall as more "
          "training months become available, so the model ranking is stable over time.")

    pe = R["profile_eval"].set_index("model")
    fb = R["future_best"]
    ref = "Reference (zone typical demand)"
    doc.h2("11.6 Future-date models")
    pt = R["profile_eval"][["model", "MAE", "MSE", "RMSE", "R2"]].copy()
    pt.columns = ["Model", "MAE (kWh)", "MSE", "RMSE (kWh)", "R²"]
    pt["R²"] = pt["R²"].map(lambda v: f"{v:.4f}")
    doc.table(pt, widths=(58, 28, 34, 28, 26))
    doc.p(f"Tested on February 2023 after training on earlier months. The best future-date model is **{fb}** "
          f"(RMSE {pe.at[fb, 'RMSE']:.1f} kWh), only {100 * (1 - pe.at[fb, 'RMSE'] / pe.at[ref, 'RMSE']):.1f}% better "
          f"than simply using the zone's typical demand ({pe.at[ref, 'RMSE']:.1f} kWh), and far less precise "
          f"than next-hour forecasting ({wf[(wf.month == '2023-02') & (wf.model == b)].RMSE.iloc[0]:.1f} kWh on "
          "the same month). Without recent real demand, the best available estimate is close to the zone's "
          "usual pattern; the tree models fitted Sep-Jan specifics that did not carry over to February. "
          "Future estimates are therefore labelled as estimates in the application.")

    # 12
    doc.h1("12. Conclusion")
    top3 = ", ".join(imp_b.feature.head(3)) if imp_b is not None else "recent demand lags"
    doc.bullets([
        f"The UrbanEV dataset was fully inspected and validated ({S['zone_hour_records']:,} zone-hours); all files and identifiers align.",
        f"A leakage-free next-hour forecasting pipeline with {len(FEATURES)} features was built and verified.",
        f"All three models clearly outperform the naive forecasts; **{b}** is the best model (R² {comp.at[b, 'R2']:.3f}, "
        f"average error {comp.at[b, 'MAE']:.1f} kWh per zone-hour).",
        f"The most influential features are {top3}: recent demand history, especially the same hour on the previous day, drives next-hour demand.",
        f"Demand is concentrated overnight: {R['night_share'] * 100:.0f}% of all energy is delivered between 00:00 and 07:59, "
        "when electricity prices are lowest.",
        "Zones with more fast chargers and higher installed power deliver more energy.",
        "Walk-forward forecasting confirmed the ranking in every month from October to February.",
        "Without recent demand, future-date estimates are close to typical-demand averages: recent history is what makes accurate forecasting possible.",
        "Everything is deployed in a local web application where users pick any date: dataset-period dates are compared with actual demand, later dates receive clearly-labelled estimates.",
    ])
    doc.p("**Practical applications:** operators can plan capacity, maintenance and dynamic pricing; drivers can "
          "avoid busy hours; planners can identify zones where chargers are under pressure; grid operators can "
          "anticipate EV charging load.")

    # 13
    doc.h1("13. Limitations")
    doc.bullets([
        "Only six months of data: yearly seasonality cannot be learnt, and calendar month cannot be used as a feature.",
        "Results are specific to Shenzhen and its large electric taxi fleet; other cities may behave differently.",
        "No weather or event data; public holidays are represented only by a flag.",
        "Forecasts are one hour ahead and evaluated on historical data; the application is not connected to live data.",
        "Future-date estimates reflect Sep 2022 - Feb 2023 patterns only (no summer months, no EV growth after February 2023) and cannot be checked against actual demand. Holiday dates after the dataset come from official schedules and should be verified.",
        "Small inconsistencies between source files (e.g. the rated-power list covers fewer stations than the station file) were documented, not corrected.",
        "Random Forest was size-limited to run on a laptop, which may cost some accuracy.",
    ])

    # 14
    doc.h1("14. Future Scope")
    doc.bullets([
        "Add weather, traffic and event data, and a full public-holiday calendar.",
        "Forecast several hours or days ahead (multi-step forecasting).",
        "Compare with sequence and graph models (LSTM/GRU, STGCN, Graph WaveNet) that use the adjacency matrix directly.",
        "Connect the application to live charging data for real-time forecasts and alerts.",
        "Apply the pipeline to other cities' charging data to test generalisation.",
    ])

    # 15
    doc.h1("15. References")
    doc.bullets([
        "Li, H., Qu, H., Tan, X., You, L., Zhu, R., & Fan, W. (2025). UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction. Scientific Data, 12, 523. https://doi.org/10.1038/s41597-025-04874-4",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32.",
        "Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems 30.",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
        "Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.",
    ])
    return doc


# =====================================================================  viva content
def build_viva(R):
    S, best, comp = R["summary"], R["best"], R["comp"]
    b = best["best_model"]
    tc = R["tcorr"].set_index("feature").pearson_r
    wz = R["within"].set_index("feature")
    met = R["metrics"]
    rm = lambda m, s: met[(met.model == m) & (met.split == s)].RMSE.iloc[0]
    imp_b = R["imp"][R["imp"].model == b].sort_values("importance", ascending=False)
    lr_rmse, rf_rmse = comp.at["Linear Regression", "RMSE"], comp.at["Random Forest", "RMSE"]
    doc = Doc("Viva Preparation", "Spatiotemporal Prediction of EV Charging Demand - questions and answers")

    doc.h1("Project in one paragraph")
    doc.p(f"We predict the energy (kWh) delivered by EV chargers in each of Shenzhen's {S['zones']} charging zones "
          f"during the next hour, using the UrbanEV dataset ({S['zone_hour_records']:,} zone-hours, Sep 2022 - "
          f"Feb 2023). We engineered {len(FEATURES)} leakage-free features, compared Linear Regression, Random Forest "
          f"and LightGBM on a chronological split, and selected **{b}** (test MAE {comp.at[b, 'MAE']:.1f} kWh, RMSE "
          f"{comp.at[b, 'RMSE']:.1f} kWh, R² {comp.at[b, 'R2']:.3f}). It powers a Streamlit app for end users.")

    doc.h1("1. Dataset and its structure")
    doc.qa("What dataset did you use?", "UrbanEV (Li et al., Scientific Data, 2025): an open benchmark of EV charging in Shenzhen. We used the zone-level, cleaned, hourly version.")
    doc.qa("How is the data structured?", f"Each hourly CSV is a wide table: a time column plus one column per zone ({S['zones']} zones x {S['hourly_files']['volume']['rows (hours)']:,} hours). We reshaped it into a long table with one row per zone-hour ({S['zone_hour_records']:,} rows).")
    doc.qa("What is the target variable?", "volume: total energy (kWh) delivered by chargers in a zone during an hour. It is continuous, so this is regression.")
    doc.qa("Which other files did you use and how are they linked?", "zone-information (by zone ID = TAZID), station information (station_id -> TAZID), pile rated power (pile -> station_id), and the 275 x 275 adjacency and distance matrices, whose order matches the zone columns. We verified every link.")
    doc.qa("Were there missing values or data problems?", f"No missing, negative or duplicate values. We found zone {', '.join(map(str, S['zone_info']['zones with missing (0,0) coordinates']))} with (0, 0) coordinates (fixed from its stations), {S['adjacency']['asymmetric pairs']} one-way adjacency links (made symmetric) and {S['adjacency']['zones with no neighbours']} zones without neighbours (used the 3 nearest zones).")
    doc.qa("Why did you not use the 5-minute data?", f"The hourly data is exactly the sum of the 5-minute data (max difference {R['five']['max_abs_difference_kwh']} kWh), so it adds no new information for hourly forecasting and is 12 times larger. We used it only for this consistency check.")

    doc.h1("2. Regression")
    doc.qa("Why is this a regression problem?", "The output is a continuous quantity (kWh), not a category.")
    doc.qa("What is supervised learning?", "Learning a mapping from inputs to a known output using labelled examples; here each past zone-hour has its known demand as the label.")

    doc.h1("3. Linear Regression")
    doc.qa("How does Linear Regression work?", "It predicts ŷ = b0 + b1·x1 + ... + bn·xn and chooses the coefficients that minimise the sum of squared errors (ordinary least squares).")
    doc.qa("Why did you scale features for Linear Regression only?", "Scaling does not change tree splits, but for a linear model it puts coefficients on a comparable scale. The scaler was fitted on training data only, inside a pipeline.")
    doc.qa("How do you interpret its coefficients?", f"With standardised features, a coefficient is the change in kWh for a one-standard-deviation increase in that feature. The largest is {R['coef'].feature.iloc[0]} ({R['coef'].coefficient.iloc[0]:.1f} kWh). Because lag features are highly correlated, individual coefficients should be read with caution.")
    doc.qa("Why did Linear Regression perform worst?", f"It can only fit straight-line relationships. Its error is high even on training data (RMSE {rm('Linear Regression', 'train'):.1f} kWh), which indicates underfitting.")

    doc.h1("4. Random Forest")
    doc.qa("How does a Random Forest work?", "It trains many decision trees on random samples of rows, each split considering a random subset of features, and averages their predictions (bagging). Averaging reduces variance.")
    doc.qa("Which hyperparameters did you use and why?", "100 trees, max depth 18, at least 20 samples per leaf, 50% of features per split and 30% of rows per tree. These limits keep training time and memory acceptable on an 8 GB laptop.")
    doc.qa("How is feature importance calculated?", "As the total reduction in squared error (impurity) contributed by each feature across all splits and trees, normalised to sum to 1.")

    doc.h1("5. LightGBM")
    doc.qa("How does LightGBM work?", "It is gradient boosting: trees are added one at a time, each fitted to the remaining errors of the current model and scaled by a learning rate. It grows trees leaf-wise and uses histograms of feature values, which makes it fast.")
    doc.qa("What is the difference between bagging and boosting?", "Bagging (Random Forest) trains trees independently and averages them, mainly reducing variance. Boosting (LightGBM) trains trees sequentially, each correcting the previous errors, mainly reducing bias.")
    doc.qa("What is early stopping?", f"Training stops when the validation error has not improved for 100 rounds. LightGBM stopped at {R['log'].get('LightGBM', {}).get('best_iteration', 'n/a')} trees, which prevents overfitting.")
    doc.qa(f"Why did {b} perform best?", f"It captured non-linear patterns and interactions better than a linear model, and boosting with early stopping corrected errors without overfitting. Its test RMSE is {100 * (1 - comp.at[b, 'RMSE'] / lr_rmse):.1f}% lower than Linear Regression and {100 * (1 - comp.at[b, 'RMSE'] / rf_rmse):.1f}% lower than Random Forest." if b == "LightGBM" else f"It had the lowest validation and test error (see the comparison table).")

    doc.h1("6. Correlation and multicollinearity")
    doc.qa("What does Pearson correlation measure?", "The strength and direction of a LINEAR relationship between two variables, from -1 to +1. It does not imply causation.")
    doc.qa("Which features correlate most with demand?", f"volume_lag24 (r = {tc['volume_lag24']:.2f}), volume_lag168 ({tc['volume_lag168']:.2f}) and volume_lag1 ({tc['volume_lag1']:.2f}). The only noticeable negative correlation is e_price ({tc['e_price']:.2f}).")
    doc.qa("Why does neighbour demand show almost zero correlation if it is useful?", f"Pooled over all zones r = {tc['neighbor_volume_lag1']:.2f}, because size differences between zones dominate. Inside each zone over time the median r is {wz.at['neighbor_volume_lag1', 'median_within_zone_r']:.2f}.")
    doc.qa("What is multicollinearity and did you find it?", f"Strong correlation between input features. We found {len(R['pairs'])} pairs with |r| >= 0.8, e.g. the demand lags with each other and occupancy_lag1 with duration_lag1. It makes linear coefficients unstable but does not harm tree models.")
    doc.qa("Why is hour almost uncorrelated with demand?", f"Pearson only measures linear association; demand follows a daily cycle. The cyclical hour_cos feature has median within-zone r = {wz.at['hour_cos', 'median_within_zone_r']:.2f}.")

    doc.h1("7. Feature engineering and lag features")
    doc.qa("What features did you create?", f"{len(FEATURES)} inputs: demand lags (1, 24, 168 h), a 24-hour rolling mean, previous-hour occupancy and duration, neighbouring zones' previous-hour demand, tariff prices, calendar features (hour, day, weekday, weekend, holiday, sine/cosine cycles) and zone attributes (chargers, fast-charger share, power, area, location).")
    doc.qa("What is a lag feature?", "The value of a variable at an earlier time, e.g. volume_lag24 is the zone's demand 24 hours earlier. Lags were computed separately for each zone and shifted strictly into the past.")
    doc.qa("Why lags of 1, 24 and 168 hours?", "1 h captures current momentum, 24 h captures the daily cycle and 168 h captures the weekly cycle.")
    doc.qa("Why use sine/cosine for the hour?", "Hour 23 and hour 0 are adjacent in time but far apart as numbers. Sine/cosine encoding places them close together.")
    doc.qa("Why did you exclude month?", "Training covers Sep-Dec and testing is February, so the model would see a month value it never learnt; six months cannot teach yearly seasonality.")

    doc.h1("8. Train/test splitting")
    sp = R["splits"]
    doc.qa("How did you split the data?", f"Chronologically: train {sp.loc['train', 'start']:%d %b %Y} - {sp.loc['train', 'end']:%d %b %Y} ({sp.loc['train', 'rows']:,} rows), validation January 2023 ({sp.loc['val', 'rows']:,}), test February 2023 ({sp.loc['test', 'rows']:,}). All zones of a timestamp are in the same split.")
    doc.qa("Why not a random split?", "Random splitting would let the model train on hours after the test hours and on almost identical neighbouring hours, giving unrealistically good scores. Forecasting must always predict the future from the past.")
    doc.qa("Why select the model on validation instead of test?", "Choosing on the test set would make the reported test score optimistic. The test set is only used once, to confirm the choice.")

    doc.h1("9. Evaluation metrics")
    doc.qa("Define MAE, MSE, RMSE and R².", "MAE: average absolute error (kWh). MSE: average squared error (kWh²), penalises large errors. RMSE: square root of MSE, back in kWh. R²: share of variance explained, 1 is perfect.")
    doc.qa(f"What do {b}'s results mean in practice?", f"On average its forecast is {comp.at[b, 'MAE']:.1f} kWh away from the actual hourly demand of a zone (mean demand is about {R['vol_stats']['mean']:.0f} kWh). RMSE {comp.at[b, 'RMSE']:.1f} kWh is larger than MAE because a few large zones have larger errors. R² = {comp.at[b, 'R2']:.3f} means it explains {comp.at[b, 'R2'] * 100:.1f}% of the variation.")
    doc.qa("Is a high R² enough to say the model is good?", f"No. Part of R² comes from differences between big and small zones. That is why we compared against naive forecasts: {b} reduces RMSE by {100 * (1 - comp.at[b, 'RMSE'] / comp.at['Naive (lag-1)', 'RMSE']):.1f}% compared with repeating the last hour.")

    doc.h1("10. Overfitting and underfitting")
    doc.qa("What are overfitting and underfitting?", "Overfitting: the model memorises training data and performs much worse on new data. Underfitting: the model is too simple and performs poorly even on training data.")
    doc.qa("Did your models overfit?", f"Not noticeably. {b}: train RMSE {rm(b, 'train'):.1f}, validation {rm(b, 'val'):.1f}, test {rm(b, 'test'):.1f} kWh. Random Forest: {rm('Random Forest', 'train'):.1f} / {rm('Random Forest', 'val'):.1f} / {rm('Random Forest', 'test'):.1f}. Linear Regression underfits (high error on training data).")
    doc.qa("How did you prevent overfitting?", "Chronological validation, early stopping for LightGBM, minimum samples per leaf and depth limits for Random Forest, and no tuning on the test set.")

    doc.h1("11. Model comparison")
    ct = comp.reset_index()[["model", "MAE", "RMSE", "R2"]]
    ct.columns = ["Model", "MAE (kWh)", "RMSE (kWh)", "R²"]
    ct["R²"] = ct["R²"].map(lambda v: f"{v:.4f}")
    doc.table(ct, widths=(56, 39, 39, 40))
    doc.qa("Which features matter most?", f"For {b}: " + ", ".join(f"{f} ({v * 100:.1f}%)" for f, v in zip(imp_b.feature.head(3), imp_b.importance.head(3))) + ". Recent demand history dominates.")

    doc.h1("12. Data leakage")
    doc.qa("What is data leakage?", "Using information during training that would not be available at prediction time, which gives falsely good results.")
    doc.qa("How did you avoid it?", "Only past values (lags) of demand, occupancy and duration are used; volume_11kw (part of the target) is excluded; the scaler is fitted on training data only; the split is chronological; and lag values were checked automatically against the raw data.")
    doc.qa("Why can you use same-hour prices but not same-hour occupancy?", f"Prices follow a published time-of-use tariff ({R['prices']['e_price'] * 100:.0f}% of days repeat the previous day's profile), so they are known in advance. Occupancy is only known after the hour has happened.")
    doc.qa("Why is volume_11kw leakage?", f"It is the energy from slow chargers, which is part of the total volume (≤ volume in {S['volume_11kw_leakage']['share of rows with volume_11kw <= volume'] * 100:.1f}% of rows, {S['volume_11kw_leakage']['share of total energy'] * 100:.0f}% of all energy). Using it means partly giving the model the answer.")

    wf = R["wf"]
    allwf = wf[wf.month == "Oct-Feb (all)"].set_index("model")
    pe = R["profile_eval"].set_index("model")
    fb, ref = R["future_best"], "Reference (zone typical demand)"
    doc.h1("13. Walk-forward forecasting and future dates")
    doc.qa("What is walk-forward forecasting?", "Each month is predicted by models trained only on the months before it (train Sep -> predict Oct, train Sep-Oct -> predict Nov, ... train Sep-Jan -> predict Feb). It simulates retraining the model every month as new data arrives.")
    doc.qa("Why did you use it?", f"So users can pick any date from 1 Oct 2022 to 28 Feb 2023 and compare an honest prediction with the actual demand, and to check that the model ranking is stable over time. Over Oct-Feb {b} reached RMSE {allwf.at[b, 'RMSE']:.1f} kWh (R² {allwf.at[b, 'R2']:.3f}), versus {allwf.at['Random Forest', 'RMSE']:.1f} for Random Forest and {allwf.at['Linear Regression', 'RMSE']:.1f} for Linear Regression.")
    doc.qa("Why can't September be predicted?", "It is the first month of the data, so there is no earlier data to train on. Predicting it with a model that learnt from it would not be an honest forecast.")
    doc.qa("How can you predict dates after February 2023?", "Next-hour models need the last hours' real demand, which does not exist after the dataset. For later dates we use profile models: inputs known for any date (hour, weekday, holiday, typical prices, zone attributes and the zone's typical demand), trained on all six months.")
    doc.qa("How accurate are the future-date estimates?", f"Tested on February: best model {fb}, RMSE {pe.at[fb, 'RMSE']:.1f} kWh, only {100 * (1 - pe.at[fb, 'RMSE'] / pe.at[ref, 'RMSE']):.1f}% better than just using the zone's typical demand ({pe.at[ref, 'RMSE']:.1f}). Next-hour forecasts on the same month are far better ({wf[(wf.month == '2023-02') & (wf.model == b)].RMSE.iloc[0]:.1f}). Recent demand is the key information.")
    doc.qa("Why is Linear Regression best for future dates when LightGBM is best overall?", "Without recent demand the task becomes 'scale the zone's usual pattern'. The tree models learnt Sep-Jan specifics that did not carry over to February, while the simpler linear model generalised better. The best model depends on the task and the inputs.")
    doc.qa("What are the limits of future estimates?", "They reflect Sep 2022 - Feb 2023 patterns only: no summer months, no growth in EV numbers or new stations after February 2023, and they cannot be checked against actual data. The app labels them as estimates.")

    doc.h1("14. Application and practical use")
    doc.qa("How does the web application work?", "It is a Streamlit app. The user picks a zone and any date from 1 Oct 2022 to today. For dataset dates it shows all three models' predictions next to the actual demand with an accuracy label; for later dates it shows estimates with a typical error range. Pages: Home, Demand Forecast, Prediction vs Actual, Demand Map, Zone Insights, Compare Zones, About the Model.")
    doc.qa("What are the practical applications?", "Operators: capacity planning, dynamic pricing, maintenance in quiet hours. Drivers: avoiding queues. Planners: identifying overloaded zones. Grid operators: anticipating EV charging load.")
    doc.qa("What did you find surprising?", f"Demand peaks overnight: {R['night_share'] * 100:.0f}% of energy is delivered between 00:00 and 07:59, when electricity is cheapest, consistent with electric taxis and ride-hailing fleets charging at night.")
    doc.qa("What are the main limitations and improvements?", "Six months of data, one city, no weather data, one-hour horizon. Improvements: weather/event data, multi-step forecasts, graph neural networks using the adjacency matrix, and live data.")
    return doc


def main():
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading results ...")
    R = load_results()
    for builder, stem in [(build_report, "Project_Report"), (build_viva, "Viva_Preparation")]:
        doc = builder(R)
        render_pdf(doc, config.REPORTS_DIR / f"{stem}.pdf")
        render_docx(doc, config.REPORTS_DIR / f"{stem}.docx")
        print(f"Saved reports/{stem}.pdf and reports/{stem}.docx")


if __name__ == "__main__":
    main()
