# ⚡ EV Charging Demand Prediction System

**Spatiotemporal Prediction of Electric Vehicle Charging Demand** — Data Science Lab project.

The system predicts the energy (kWh) that electric-vehicle chargers deliver in each of
Shenzhen's 275 charging zones per hour. Three regression models
(**Linear Regression, Random Forest, LightGBM**) are trained on the UrbanEV dataset and compared.
In the local **Streamlit** web application users pick **any date from 1 Oct 2022 up to today**:

| Dates | What the app shows |
|---|---|
| 1 Oct 2022 – 28 Feb 2023 (dataset period) | Next-hour predictions of all three models **next to the actual demand**, with error and an accuracy label. Each month was predicted by models trained only on earlier months (walk-forward). |
| 1 Mar 2023 – today | **Estimates** from future-date (profile) models trained on all six months, with a typical error range. No actual data exists for these dates. |

September 2022 is training-only data (no earlier month to learn from).

## Features

- Leakage-free feature pipeline: demand lags (1 h, 24 h, 168 h), rolling mean, previous-hour
  activity, neighbouring-zone demand, tariff prices, calendar and holiday flags, and charger
  capacity per zone.
- Correlation analysis (the only EDA), model training, evaluation (MAE, MSE, RMSE, R²),
  feature importance and residual analysis.
- Walk-forward forecasting (Oct 2022 – Feb 2023) and future-date models for any later date.
- A web app for end users: demand forecast, prediction vs actual over a date range, city demand
  map, zone insights, zone comparison, and an "About the Model" page with month-by-month accuracy.
- A Jupyter notebook, a project report and a viva preparation document (PDF + Word), all built
  from the saved results.

## Folder structure

```
EV/
├── app.py                     # Streamlit entry point  ->  streamlit run app.py
├── app_pages/                 # the 7 app pages (Home, Forecast, Prediction vs Actual, Map, Insights, Compare, About)
├── train.py                   # runs the full ML pipeline once (data -> features -> models -> results)
├── build_reports.py           # regenerates the report + viva documents from saved results
├── requirements.txt
├── src/
│   ├── config.py              # paths, split dates, holidays, seed
│   ├── data_loader.py         # loading, inspection, zone attributes, 5-min consistency check
│   ├── preprocessing.py       # validation, cleaning, chronological split
│   ├── feature_engineering.py # all model features + leakage checks
│   ├── eda.py                 # Pearson correlation analysis + heatmap
│   ├── train_models.py        # Linear Regression, Random Forest, LightGBM
│   ├── evaluate_models.py     # metrics, comparison, best-model selection, charts
│   ├── walk_forward.py        # month-by-month forecasts for the dataset period
│   ├── profile_model.py       # future-date models (any date after Feb 2023)
│   ├── predict.py             # prediction service used by the app (both modes)
│   ├── plot_style.py, ui.py   # shared chart colours / app helpers
├── notebooks/UrbanEV_Analysis.ipynb
├── tests/                     # pytest data, leakage and model tests
├── data/
│   ├── Dataset/UrbanEVDataset/UrbanEVDataset/   # <- put the dataset here (not in Git)
│   └── processed/             # cached Parquet tables (created automatically)
├── models/                    # saved models + best_model.json (created by train.py)
├── artifacts/                 # correlation + evaluation tables and charts
└── reports/                   # Project_Report.pdf/.docx, Viva_Preparation.pdf/.docx
```

## Dataset placement

The dataset is too large for Git (about 6 GB) and must be placed locally:

```
data/Dataset/UrbanEVDataset/UrbanEVDataset/
├── 20220901-20230228_zone-cleaned-aggregated/
│   ├── charge_1hour/   (volume.csv, volume-11kw.csv, occupancy.csv, duration.csv, e_price.csv, s_price.csv)
│   ├── charge_5min/    (only volume.csv is used, for a consistency check)
│   ├── adj.csv, distance.csv, zone-information.csv, station_information.csv
└── 20220901-20230228_station-raw/
    ├── station_information.csv
    └── pile_rated_power.csv
```

The large `station-raw/charge_5min/` folder is **not** used and may be deleted.
To keep the data somewhere else, set the environment variable `EV_DATA_DIR` to the folder that
contains the two sub-folders above.

Source: Li, H. et al. (2025). *UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle
Charging Demand Prediction.* Scientific Data, 12, 523. https://doi.org/10.1038/s41597-025-04874-4

## Installation (Windows)

Requires Python 3.9–3.11. Run these commands in the project folder:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the project

1. **Train the models (once, about 20 minutes on a laptop):**
   ```bat
   python train.py
   ```
   This builds the cached tables, runs the correlation analysis, trains and compares the three
   models, computes the walk-forward forecasts (the slowest step, about 15 minutes) and trains the
   future-date models. Saved results are reused on later runs (then it takes about 1 minute);
   use `python train.py --force` to retrain everything.

2. **Start the web application:**
   ```bat
   streamlit run app.py
   ```
   It opens at **http://localhost:8501**. The map background needs an internet connection;
   everything else works offline.

3. **Optional: regenerate the documents** (after retraining):
   ```bat
   python build_reports.py
   ```

4. **Optional: run the tests:**
   ```bat
   python -m pytest -q
   ```

5. **Notebook:** open `notebooks/UrbanEV_Analysis.ipynb` in VS Code or Jupyter and choose the
   **"Python (EV venv)"** kernel (or the `venv` interpreter), then *Run All*. To register the
   kernel on a new machine, run:
   `python -m ipykernel install --sys-prefix --name ev-venv --display-name "Python (EV venv)"`

## Method summary

| Step | Details |
|---|---|
| Target | `volume`: kWh delivered per zone per hour |
| Horizon | next hour. Features use only data from hour t−1 or earlier, plus values known in advance |
| Split | train 8 Sep–31 Dec 2022 · validation Jan 2023 · test Feb 2023 (chronological) |
| Models | Linear Regression (scaled), Random Forest, LightGBM, plus two naive reference forecasts |
| Selection | lowest validation RMSE, confirmed on the test set |
| Dataset-period app forecasts | walk-forward: each month Oct–Feb predicted by models trained only on earlier months |
| Future-date estimates | profile models (calendar, holidays, typical prices, zone typical demand) trained on all six months; holiday dates after Feb 2023 are listed in `src/config.py` and should be verified |
| Excluded inputs | `volume_11kw` (part of the target), same-hour occupancy and duration (unknown in advance), `month` (test month unseen in training) |

Results (metrics, best model, charts) are in `artifacts/evaluation/`, on the app's
**About the Model** page and in `reports/Project_Report.pdf`.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Dataset files not found" | Check the folder layout above or set `EV_DATA_DIR` |
| "The forecasting model is not ready" | Run `python train.py` |
| Port 8501 already in use | Close the other Streamlit window or run `streamlit run app.py --server.port 8502` |
| Notebook says pyarrow is missing | Select the "Python (EV venv)" kernel, not the system Python |
