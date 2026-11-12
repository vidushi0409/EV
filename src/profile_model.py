"""Demand-profile models for dates AFTER the dataset ends (Mode 2).

Next-hour models need the previous hours' real demand, which does not exist
after 28 Feb 2023. Profile models instead use only inputs that are known for
ANY date:
  * calendar: hour, weekday, weekend, public holiday, sine/cosine cycles;
  * tariff prices: each zone's typical price for that hour (median of the last
    28 days of data, used for dates after the dataset);
  * static zone attributes (chargers, power, area, location);
  * zone demand profile learnt from TRAINING data only:
      zone_hour_mean     - the zone's average demand at that hour of day,
      zone_dow_mean      - the zone's average demand on that weekday,
      zone_wk_hour_mean  - average demand at that hour on weekdays / weekends.

The models predict demand RELATIVE to the zone's typical demand
(volume / (zone_wk_hour_mean + 1)) and the result is scaled back to kWh. This
lets weekday, holiday and price effects be learnt once for all zones.
A reference forecast "zone_wk_hour_mean" (no learning) is evaluated too.

Evaluation: trained on 1 Sep 2022 - 31 Jan 2023, tested on February 2023.
Final models: retrained on all six months for future-date estimates.
Limitations: patterns of Sep 2022 - Feb 2023 only (no summer months, no EV
growth after Feb 2023).
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_panel, load_zone_info
from src.evaluate_models import predict, regression_metrics
from src.feature_engineering import STATIC_FEATURES, add_calendar_features
from src.preprocessing import clean_panel
from src.train_models import MODEL_NAMES, build_model

PROFILE_DIR = config.MODELS_DIR / "profile"
PROFILE_EVAL = config.ARTIFACTS_DIR / "profile_model" / "evaluation.csv"
PROFILE_ZONE_MAE = config.ARTIFACTS_DIR / "profile_model" / "zone_mae.csv"
PROFILE_STATS = PROFILE_DIR / "zone_profiles.joblib"
PROFILE_PRICES = PROFILE_DIR / "future_prices.parquet"
PROFILE_INFO = PROFILE_DIR / "info.json"
PROFILE_FILES = {n: PROFILE_DIR / f"{n.lower().replace(' ', '_')}.joblib" for n in MODEL_NAMES}

PROFILE_FEATURES = (["hour", "dayofweek", "is_weekend", "is_holiday", "hour_sin", "hour_cos",
                     "dow_sin", "dow_cos", "e_price", "s_price"] + STATIC_FEATURES
                    + ["zone_hour_mean", "zone_dow_mean", "zone_wk_hour_mean"])
REFERENCE = "Reference (zone typical demand)"
TEST_START = pd.Timestamp("2023-02-01")


def profile_models_ready():
    """True only if EVERY file the estimator needs exists (model files are not in Git)."""
    needed = list(PROFILE_FILES.values()) + [PROFILE_STATS, PROFILE_PRICES, PROFILE_INFO,
                                             PROFILE_EVAL, PROFILE_ZONE_MAE]
    return all(p.exists() for p in needed)


# ---------------------------------------------------------------- feature building
def zone_profiles(history):
    """Zone x hour and zone x weekday average demand from the given history."""
    h = history.assign(hour=history.time.dt.hour, dayofweek=history.time.dt.dayofweek)
    by_hour = h.groupby(["zone", "hour"]).volume.mean().rename("zone_hour_mean").reset_index()
    by_dow = h.groupby(["zone", "dayofweek"]).volume.mean().rename("zone_dow_mean").reset_index()
    h["is_weekend"] = (h.dayofweek >= 5).astype("int8")
    by_wk = (h.groupby(["zone", "is_weekend", "hour"]).volume.mean()
             .rename("zone_wk_hour_mean").reset_index())
    return by_hour, by_dow, by_wk


def build_profile_rows(frame, profiles, zones):
    """Add profile features to rows that have time, zone, e_price, s_price."""
    df = add_calendar_features(frame.copy())
    by_hour, by_dow, by_wk = profiles
    df = (df.merge(by_hour, on=["zone", "hour"], how="left")
            .merge(by_dow, on=["zone", "dayofweek"], how="left")
            .merge(by_wk, on=["zone", "is_weekend", "hour"], how="left")
            .merge(zones[["zone"] + STATIC_FEATURES], on="zone", how="left"))
    return df


def typical_prices(panel, days=28):
    """Median price per zone and hour over the last `days` days of data."""
    recent = panel[panel.time > panel.time.max() - pd.Timedelta(days=days)]
    return (recent.assign(hour=recent.time.dt.hour)
            .groupby(["zone", "hour"])[["e_price", "s_price"]].median().reset_index())


def _xy(df):
    """Inputs and RELATIVE target (demand / typical demand)."""
    scale = (df["zone_wk_hour_mean"] + 1).values
    return df[PROFILE_FEATURES].astype("float32"), (df[config.TARGET].values / scale).astype("float32")


def profile_predict(model, df):
    """Model output (relative) scaled back to kWh, clipped at 0."""
    scale = (df["zone_wk_hour_mean"] + 1).values
    return predict(model, df[PROFILE_FEATURES].astype("float32")) * scale


def _new_model(name):
    model = build_model(name)
    if name == "LightGBM":   # fixed size: no lag features -> no early-stopping set needed
        model.set_params(n_estimators=600)
    return model


# ---------------------------------------------------------------- training
def train_profile_models(log=print):
    """Evaluate on Feb 2023, then retrain on all data and save everything."""
    panel = clean_panel(load_panel())
    zones = load_zone_info()
    base = panel[["time", "zone", config.TARGET, "e_price", "s_price"]]

    # 1) honest evaluation: profiles + models from data before February only
    train_hist = base[base.time < TEST_START]
    profiles = zone_profiles(train_hist)
    train_df = build_profile_rows(train_hist, profiles, zones)
    test_df = build_profile_rows(base[base.time >= TEST_START], profiles, zones)
    X_tr, y_tr = _xy(train_df)
    y_te = test_df[config.TARGET].values
    rows, zone_mae = [], test_df[["zone"]].copy()
    ref = test_df["zone_wk_hour_mean"].values
    rows.append({"model": REFERENCE, **regression_metrics(y_te, ref), "seconds": 0.0})
    for name in MODEL_NAMES:
        t0 = time.time()
        model = _new_model(name)
        model.fit(X_tr, y_tr)
        p = profile_predict(model, test_df)
        rows.append({"model": name, **regression_metrics(y_te, p), "seconds": round(time.time() - t0, 1)})
        zone_mae[name] = np.abs(y_te - p)
        log(f"[profile] eval {name}: RMSE {rows[-1]['RMSE']:.1f}, R2 {rows[-1]['R2']:.3f}")
    evaluation = pd.DataFrame(rows)
    PROFILE_EVAL.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(PROFILE_EVAL, index=False)
    zone_mae.groupby("zone").mean().reset_index().to_csv(PROFILE_ZONE_MAE, index=False)

    # 2) final models on all six months
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profiles = zone_profiles(base)
    full_df = build_profile_rows(base, profiles, zones)
    X_all, y_all = _xy(full_df)
    for name in MODEL_NAMES:
        model = _new_model(name)
        model.fit(X_all, y_all)
        joblib.dump({"model": model, "features": PROFILE_FEATURES}, PROFILE_FILES[name], compress=3)
        log(f"[profile] final {name} saved")
    joblib.dump(profiles, PROFILE_STATS)
    typical_prices(base).to_parquet(PROFILE_PRICES, index=False)
    best = evaluation[evaluation.model.isin(MODEL_NAMES)].sort_values("RMSE").iloc[0]["model"]
    PROFILE_INFO.write_text(json.dumps({
        "best_model": best, "trained_on": f"{base.time.min():%Y-%m-%d} to {base.time.max():%Y-%m-%d}",
        "evaluated_on": "2023-02 (trained on data before February)", "features": PROFILE_FEATURES,
    }, indent=2))
    return evaluation


# ---------------------------------------------------------------- inference
class ProfileEstimator:
    """Loads the final profile models and produces estimates for any date."""

    def __init__(self):
        if not profile_models_ready():
            raise FileNotFoundError("Future-date models not found. Run: python train.py")
        self.info = json.loads(PROFILE_INFO.read_text())
        self.models = {}
        for name, path in PROFILE_FILES.items():
            bundle = joblib.load(path)
            if bundle["features"] != PROFILE_FEATURES:
                raise ValueError("Profile models are out of date. Run: python train.py --force")
            self.models[name] = bundle["model"]
        self.profiles = joblib.load(PROFILE_STATS)
        self.prices = pd.read_parquet(PROFILE_PRICES)
        self.zones = load_zone_info()
        self.evaluation = pd.read_csv(PROFILE_EVAL).set_index("model")
        self.zone_mae = pd.read_csv(PROFILE_ZONE_MAE).set_index("zone")

    def estimate(self, times, zones):
        """Estimates for every combination of the given timestamps and zones."""
        grid = pd.MultiIndex.from_product([pd.to_datetime(times), list(zones)],
                                          names=["time", "zone"]).to_frame(index=False)
        grid["hour"] = grid.time.dt.hour
        grid = grid.merge(self.prices, on=["zone", "hour"], how="left").drop(columns="hour")
        df = build_profile_rows(grid, self.profiles, self.zones)
        out = df[["time", "zone", "hour", "is_holiday", "e_price", "s_price", "capacity_kw"]].copy()
        for name, model in self.models.items():
            out[name] = profile_predict(model, df)
            out[f"{name} typical error"] = self.zone_mae.loc[out.zone, name].values
        return out
