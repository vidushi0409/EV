"""Prediction service used by the user application (two modes).

Mode 1 - DATASET PERIOD (1 Oct 2022 - 28 Feb 2023)
    Next-hour forecasts from walk-forward models: each month was predicted by
    models trained only on earlier months, so every forecast is for unseen data
    and can be compared with the real value from the dataset.

Mode 2 - FUTURE DATES (after 28 Feb 2023, up to today)
    Pattern-based estimates from the profile models trained on all six months.
    No actual demand exists for these dates, so no comparison is possible;
    a typical error range (measured on held-out February data) is shown instead.

September 2022 is training-only data: it cannot be forecast honestly because
no earlier data exists to learn from.
"""
from datetime import date

import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_zone_info
from src.evaluate_models import read_best_model
from src.feature_engineering import FEATURE_CACHE, load_features
from src.profile_model import ProfileEstimator
from src.train_models import MODEL_NAMES
from src.walk_forward import load_walk_forward

LEVELS = ["Low", "Moderate", "High", "Very High"]
LEVELS_FILE = config.ARTIFACTS_DIR / "zone_demand_levels.csv"
DATASET_START = pd.Timestamp(config.WALK_FORWARD_MONTHS[0] + "-01")
DATASET_END = pd.Timestamp(config.DATASET_END)


class ModelNotReadyError(RuntimeError):
    """Raised when models or features have not been produced yet."""


def zone_demand_levels():
    """Per-zone thresholds (25th / 75th / 90th percentile of TRAINING demand)."""
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
    kwh = np.asarray(kwh, dtype=float)
    idx = np.where(kwh < p25, 0, np.where(kwh < p75, 1, np.where(kwh < p90, 2, 3)))
    return [LEVELS[i] for i in np.atleast_1d(idx)]


def accuracy_badge(predicted, actual):
    """Plain-language accuracy of one prediction against the real value."""
    err = abs(predicted - actual)
    pct = err / actual * 100 if actual >= 10 else None
    if (pct is not None and pct <= 10) or (pct is None and err <= 5):
        label, icon = "Very accurate", "✅"
    elif (pct is not None and pct <= 25) or (pct is None and err <= 15):
        label, icon = "Accurate", "✅"
    elif (pct is not None and pct <= 50) or (pct is None and err <= 30):
        label, icon = "Fair", "⚠️"
    else:
        label, icon = "Off", "❌"
    text = f"{pct:.0f}% from actual" if pct is not None else f"{err:.1f} kWh from actual"
    return {"label": label, "icon": icon, "text": text, "error_kwh": err, "error_pct": pct}


class DemandService:
    """Single entry point for every page. Create once and reuse."""

    def __init__(self):
        best = read_best_model()
        if best is None or not FEATURE_CACHE.exists():
            raise ModelNotReadyError("Models not trained yet. Run: python train.py")
        self.dataset_best = best["best_model"]
        preds, self.monthly = load_walk_forward()
        context = pd.read_parquet(
            FEATURE_CACHE, columns=["time", "zone", "volume_lag1", "volume_lag24", "e_price",
                                    "s_price", "capacity_kw"],
            filters=[("time", ">=", DATASET_START)])
        self.history = preds.merge(context, on=["time", "zone"], how="left")
        self.profile = ProfileEstimator()
        self.future_best = self.profile.info["best_model"]
        self.zones = load_zone_info()[["zone", "longitude", "latitude", "num_piles", "fast_count",
                                       "slow_count", "capacity_kw"]]
        self.levels = zone_demand_levels().set_index("zone")
        self.max_date = max(pd.Timestamp(date.today()), DATASET_END.normalize())

    # ---------------------------------------------------------------- validation
    def zone_ids(self):
        return sorted(self.zones.zone.tolist())

    def mode_of(self, ts):
        ts = pd.Timestamp(ts)
        if ts < DATASET_START:
            raise ValueError("September 2022 is the first month of the dataset and is used only "
                             "for training, so it cannot be forecast honestly. Please choose "
                             "1 October 2022 or later.")
        if ts > self.max_date + pd.Timedelta(hours=23):
            raise ValueError(f"Please choose a date up to {self.max_date:%d %b %Y}.")
        return "dataset" if ts <= DATASET_END else "future"

    def _check_zone(self, zone):
        if int(zone) not in self.levels.index:
            raise ValueError(f"Unknown zone: {zone}")
        return int(zone)

    # ---------------------------------------------------------------- core
    def _finish(self, df, mode):
        """Common columns: best prediction, level, utilisation, price."""
        best = self.dataset_best if mode == "dataset" else self.future_best
        df["mode"] = mode
        df["best_model"] = best
        df["predicted_kwh"] = df[best]
        lv = self.levels.loc[df.zone]
        df["level"] = demand_level(df.predicted_kwh.values, lv.p25.values, lv.p75.values, lv.p90.values)
        df["utilisation_pct"] = 100 * df.predicted_kwh / df.capacity_kw
        df["price_per_kwh"] = df.e_price + df.s_price
        df["hour"] = df.time.dt.hour
        if mode == "future":
            df["actual_kwh"] = np.nan
        return df.reset_index(drop=True)

    def _rows(self, times, zones, mode):
        if mode == "dataset":
            h = self.history
            df = h[h.time.isin(times) & h.zone.isin(zones)].rename(columns={"actual": "actual_kwh"})
            return self._finish(df.copy(), mode)
        return self._finish(self.profile.estimate(times, zones), mode)

    # ---------------------------------------------------------------- public API
    def zone_hour(self, zone, ts):
        zone, ts = self._check_zone(zone), pd.Timestamp(ts)
        mode = self.mode_of(ts)
        r = self._rows([ts], [zone], mode).iloc[0]
        out = {"zone": zone, "time": ts, "mode": mode, "best_model": r.best_model,
               "predicted_kwh": float(r.predicted_kwh), "level": r.level,
               "utilisation_pct": float(r.utilisation_pct), "price_per_kwh": float(r.price_per_kwh),
               "models": {m: float(r[m]) for m in MODEL_NAMES}}
        if mode == "dataset":
            out["actual_kwh"] = float(r.actual_kwh)
            out["previous_hour_kwh"] = float(r.volume_lag1)
            out["badges"] = {m: accuracy_badge(out["models"][m], out["actual_kwh"]) for m in MODEL_NAMES}
        else:
            out["typical_error"] = {m: float(r[f"{m} typical error"]) for m in MODEL_NAMES}
            out["is_holiday"] = bool(r.is_holiday)
        return out

    def zone_day(self, zone, day):
        zone, day = self._check_zone(zone), pd.Timestamp(day).normalize()
        mode = self.mode_of(day)
        times = pd.date_range(day, periods=24, freq="h")
        return self._rows(times, [zone], mode)

    def city_hour(self, ts):
        ts = pd.Timestamp(ts)
        mode = self.mode_of(ts)
        return self._rows([ts], self.zone_ids(), mode).merge(
            self.zones.drop(columns="capacity_kw"), on="zone", how="left")

    def compare_range(self, start, end, zone=None):
        """Dataset-period comparison over a date range for one zone or the whole city.

        Returns (hourly DataFrame, metrics DataFrame)."""
        start, end = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23)
        if self.mode_of(start) != "dataset" or end > DATASET_END:
            raise ValueError("Comparison with actual demand is only possible between "
                             f"{DATASET_START:%d %b %Y} and {DATASET_END:%d %b %Y}.")
        if end < start:
            raise ValueError("The end date must be after the start date.")
        h = self.history[(self.history.time >= start) & (self.history.time <= end)]
        if zone is not None:
            h = h[h.zone == self._check_zone(zone)]
        hourly = h.groupby("time")[["actual"] + MODEL_NAMES].sum().reset_index()
        from src.evaluate_models import regression_metrics
        metrics = pd.DataFrame([{"model": m, **regression_metrics(hourly.actual, hourly[m]),
                                 "total_error_pct": 100 * abs(hourly[m].sum() - hourly.actual.sum())
                                 / max(hourly.actual.sum(), 1e-9)} for m in MODEL_NAMES])
        return hourly.rename(columns={"actual": "actual_kwh"}), metrics

    def dataset_summary(self):
        """Hourly city totals over the whole forecastable dataset period."""
        return self.history.groupby("time")[["actual"] + MODEL_NAMES].sum()
