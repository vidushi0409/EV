"""Prediction service used by the user application.

The app never trains or builds features itself; it only calls the functions
below. Forecasts use the BEST model (chosen in evaluation) and the exact same
feature pipeline as training. Historical inputs (lags, neighbour demand) are
taken from the real dataset, so no future information is used.

Forecasts are offered for the held-out test period (February 2023): data the
models never saw during training, so each forecast can be shown next to what
actually happened.
"""
import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_zone_info
from src.evaluate_models import predict as clipped_predict
from src.evaluate_models import read_best_model
from src.feature_engineering import FEATURE_CACHE, FEATURES, load_features
from src.train_models import load_model

LEVELS = ["Low", "Moderate", "High", "Very High"]
LEVELS_FILE = config.ARTIFACTS_DIR / "zone_demand_levels.csv"
FORECAST_SPLIT = "test"


class ModelNotReadyError(RuntimeError):
    """Raised when models or features have not been produced yet."""


# ---------------------------------------------------------------- loading
def load_best_model():
    info = read_best_model()
    if info is None:
        raise ModelNotReadyError("No trained model found. Run: python train.py")
    return info["best_model"], load_model(info["best_model"])


def load_forecast_table():
    """Feature rows of the forecast period only (keeps app memory small)."""
    if not FEATURE_CACHE.exists():
        raise ModelNotReadyError("Feature table not found. Run: python train.py")
    cols = ["time", "zone", "split", config.TARGET] + FEATURES
    df = pd.read_parquet(FEATURE_CACHE, columns=cols,
                         filters=[("split", "==", FORECAST_SPLIT)])
    return df.drop(columns="split").sort_values(["time", "zone"], ignore_index=True)


def zone_table():
    """Static zone info for display (location, piles, capacity)."""
    return load_zone_info()[["zone", "longitude", "latitude", "num_piles", "fast_count",
                             "slow_count", "capacity_kw"]]


def zone_demand_levels():
    """Per-zone thresholds (25th / 75th / 90th percentile of TRAINING demand).

    A forecast is labelled relative to what is normal for that zone:
    Low < p25 <= Moderate < p75 <= High < p90 <= Very High."""
    if LEVELS_FILE.exists():
        return pd.read_csv(LEVELS_FILE)
    train = load_features()
    train = train[train.split == "train"]
    q = train.groupby("zone")[config.TARGET].quantile([0.25, 0.75, 0.90]).unstack()
    q.columns = ["p25", "p75", "p90"]
    q = q.reset_index()
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    q.to_csv(LEVELS_FILE, index=False)
    return q


def demand_level(kwh, p25, p75, p90):
    """Vectorised label for predicted kWh given a zone's thresholds."""
    kwh = np.asarray(kwh, dtype=float)
    out = np.where(kwh < p25, 0, np.where(kwh < p75, 1, np.where(kwh < p90, 2, 3)))
    return [LEVELS[i] for i in np.atleast_1d(out)]


# ---------------------------------------------------------------- forecasting
class Forecaster:
    """Bundles the best model with the forecast-period data. Create once, reuse."""

    def __init__(self):
        self.model_name, self.model = load_best_model()
        self.table = load_forecast_table()
        self.zones = zone_table()
        self.levels = zone_demand_levels().set_index("zone")
        self.start, self.end = self.table.time.min(), self.table.time.max()

    # ---- helpers
    def zone_ids(self):
        return sorted(self.table.zone.unique().tolist())

    def _check_time(self, ts):
        ts = pd.Timestamp(ts)
        if not (self.start <= ts <= self.end):
            raise ValueError(f"Choose a time between {self.start:%d %b %Y %H:%M} "
                             f"and {self.end:%d %b %Y %H:%M}.")
        return ts

    def _check_zone(self, zone):
        if int(zone) not in self.levels.index:
            raise ValueError(f"Unknown zone: {zone}")
        return int(zone)

    def _enrich(self, rows):
        """Run the model and add level, utilisation and price columns."""
        out = rows[["time", "zone", config.TARGET, "e_price", "s_price",
                    "volume_lag1", "volume_lag24", "capacity_kw"]].copy()
        out["predicted_kwh"] = clipped_predict(self.model, rows[FEATURES].astype("float32"))
        out = out.rename(columns={config.TARGET: "actual_kwh"})
        lv = self.levels.loc[out.zone]
        out["level"] = demand_level(out.predicted_kwh.values, lv.p25.values,
                                    lv.p75.values, lv.p90.values)
        out["utilisation_pct"] = 100 * out.predicted_kwh / out.capacity_kw
        out["price_per_kwh"] = out.e_price + out.s_price
        return out.reset_index(drop=True)

    # ---- public API
    def zone_hour(self, zone, ts):
        """Forecast for one zone and hour -> dict."""
        zone, ts = self._check_zone(zone), self._check_time(ts)
        rows = self.table[(self.table.zone == zone) & (self.table.time == ts)]
        r = self._enrich(rows).iloc[0]
        return {
            "zone": zone, "time": ts, "predicted_kwh": float(r.predicted_kwh),
            "actual_kwh": float(r.actual_kwh), "level": r.level,
            "utilisation_pct": float(r.utilisation_pct), "price_per_kwh": float(r.price_per_kwh),
            "previous_hour_kwh": float(r.volume_lag1), "same_hour_yesterday_kwh": float(r.volume_lag24),
        }

    def zone_day(self, zone, date):
        """24 hourly forecasts for one zone on one date."""
        zone = self._check_zone(zone)
        day = pd.Timestamp(date).normalize()
        self._check_time(day)
        rows = self.table[(self.table.zone == zone) & (self.table.time >= day)
                          & (self.table.time < day + pd.Timedelta(days=1))]
        out = self._enrich(rows)
        out["hour"] = out.time.dt.hour
        return out

    def city_hour(self, ts):
        """Forecast for every zone at one hour (with coordinates for the map)."""
        ts = self._check_time(ts)
        out = self._enrich(self.table[self.table.time == ts])
        return out.merge(self.zones, on="zone", how="left", suffixes=("", "_z"))

    def period(self):
        """Forecasts for the whole forecast period (for summary cards)."""
        return self._enrich(self.table)
