"""Feature engineering for next-hour EV charging-demand forecasting.

Forecasting set-up
------------------
Each row is (zone, hour t). The target is `volume` at hour t (kWh).
Features use ONLY information available before hour t starts:
  * historical values from hour t-1 or earlier (lags, rolling mean,
    lagged occupancy/duration, neighbours' previous-hour demand);
  * values known in advance for hour t: calendar, public holidays,
    the time-of-use tariff prices (verified to follow a fixed daily schedule)
    and static zone attributes.
Same-hour occupancy/duration and volume_11kw are NEVER used (leakage).

Lags are computed on the (time x zone) matrix, so each shift stays inside its
own zone column and moves strictly back in time.
"""
import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_adjacency, load_distance, load_panel, load_zone_info
from src.preprocessing import clean_panel, split_label

# `month` is created (for display) but NOT used as a model input: training covers
# Sep-Dec and the test set is February, a month value never seen in training, and
# six months of data cannot teach yearly seasonality.
CALENDAR_FEATURES = ["hour", "day", "dayofweek", "is_weekend", "is_holiday",
                     "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
PRICE_FEATURES = ["e_price", "s_price"]
LAG_FEATURES = ["volume_lag1", "volume_lag24", "volume_lag168", "volume_roll24_mean",
                "occupancy_lag1", "duration_lag1", "neighbor_volume_lag1"]
STATIC_FEATURES = ["num_piles", "fast_share", "avg_pile_kw", "capacity_kw", "area",
                   "longitude", "latitude"]
FEATURES = LAG_FEATURES + PRICE_FEATURES + CALENDAR_FEATURES + STATIC_FEATURES

FEATURE_CACHE = config.PROCESSED_DIR / "features.parquet"
MAX_LAG = 168  # the first 168 hours have incomplete history and are dropped


# ---------------------------------------------------------------- spatial weights
def neighbor_weights(n_nearest_fallback=3):
    """Row-normalised neighbour matrix W (zone x zone), self excluded.

    * adj.csv is made symmetric (a link in either direction counts).
    * Zones without any neighbour use their 3 nearest zones from distance.csv.
    """
    adj = load_adjacency()
    a = adj.values.astype(float)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0)
    dist = load_distance().values.astype(float)
    for i in np.where(a.sum(1) == 0)[0]:
        d = dist[i].copy()
        d[i] = np.inf
        a[i, np.argsort(d)[:n_nearest_fallback]] = 1
    w = a / a.sum(1, keepdims=True)
    return pd.DataFrame(w, index=adj.index, columns=adj.columns)


# ---------------------------------------------------------------- calendar
def add_calendar_features(df):
    t = df["time"].dt
    df["hour"] = t.hour.astype("int8")
    df["day"] = t.day.astype("int8")
    df["dayofweek"] = t.dayofweek.astype("int8")
    df["month"] = t.month.astype("int8")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    holidays = pd.to_datetime(config.HOLIDAYS)
    df["is_holiday"] = df["time"].dt.normalize().isin(holidays).astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype("float32")
    return df


# ---------------------------------------------------------------- main builder
def build_features(panel=None):
    """Create the full model table (one row per zone-hour, history-complete rows only)."""
    panel = clean_panel(load_panel() if panel is None else panel)

    # wide (time x zone) matrices make per-zone shifting simple and fast
    vol = panel.pivot(index="time", columns="zone", values="volume")
    occ = panel.pivot(index="time", columns="zone", values="occupancy")
    dur = panel.pivot(index="time", columns="zone", values="duration")
    w = neighbor_weights().loc[vol.columns, vol.columns]

    lagged = {
        "volume_lag1": vol.shift(1),
        "volume_lag24": vol.shift(24),
        "volume_lag168": vol.shift(168),
        # mean of hours t-24 ... t-1 (shift first, so hour t is never included)
        "volume_roll24_mean": vol.shift(1).rolling(24).mean(),
        "occupancy_lag1": occ.shift(1),
        "duration_lag1": dur.shift(1),
        # average previous-hour demand of neighbouring zones
        "neighbor_volume_lag1": vol.shift(1) @ w.T.values,
    }
    # panel sorted by (time, zone) has exactly the row order of the flattened matrix
    panel = panel.sort_values(["time", "zone"], ignore_index=True)
    assert (panel["zone"].values[: vol.shape[1]] == vol.columns.values).all()
    for name, frame in lagged.items():
        panel[name] = np.asarray(frame, dtype="float32").ravel()

    panel = add_calendar_features(panel)
    zones = load_zone_info()[["zone"] + STATIC_FEATURES]
    panel = panel.merge(zones, on="zone", how="left")
    panel[STATIC_FEATURES] = panel[STATIC_FEATURES].astype("float32")

    # drop the warm-up period where lag-168 is not yet defined
    first_valid = panel["time"].min() + pd.Timedelta(hours=MAX_LAG)
    panel = panel[panel["time"] >= first_valid].reset_index(drop=True)
    panel["split"] = split_label(panel["time"])

    keep = ["time", "zone", "split", "month", config.TARGET, "volume_11kw", "occupancy",
            "duration"] + [
        f for f in FEATURES if f not in ("e_price", "s_price")] + ["e_price", "s_price"]
    return panel[keep].sort_values(["time", "zone"], ignore_index=True)


def load_features(use_cache=True):
    """Feature table, cached as Parquet after the first build."""
    if use_cache and FEATURE_CACHE.exists():
        return pd.read_parquet(FEATURE_CACHE)
    feats = build_features()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATURE_CACHE, index=False)
    return feats


def leakage_checks(feats, panel=None):
    """Verify that lag features really come from the past. Returns a dict of booleans."""
    panel = clean_panel(load_panel() if panel is None else panel)
    vol = panel.pivot(index="time", columns="zone", values="volume")
    sample = feats.sample(2000, random_state=config.RANDOM_SEED)
    t, z = sample["time"], sample["zone"]
    lag1 = np.array([vol.at[ti - pd.Timedelta(hours=1), zi] for ti, zi in zip(t, z)])
    lag24 = np.array([vol.at[ti - pd.Timedelta(hours=24), zi] for ti, zi in zip(t, z)])
    lag168 = np.array([vol.at[ti - pd.Timedelta(hours=168), zi] for ti, zi in zip(t, z)])
    roll = np.array([vol.loc[ti - pd.Timedelta(hours=24): ti - pd.Timedelta(hours=1), zi].mean()
                     for ti, zi in zip(t, z)])
    tr, va, te = (feats.loc[feats.split == s, "time"] for s in ("train", "val", "test"))
    return {
        "lag1 == volume at t-1": bool(np.allclose(sample.volume_lag1, lag1, atol=1e-3)),
        "lag24 == volume at t-24": bool(np.allclose(sample.volume_lag24, lag24, atol=1e-3)),
        "lag168 == volume at t-168": bool(np.allclose(sample.volume_lag168, lag168, atol=1e-3)),
        "roll24 uses t-24..t-1 only": bool(np.allclose(sample.volume_roll24_mean, roll, rtol=1e-4, atol=1e-2)),
        "no feature equals same-hour target column": not any(
            np.allclose(feats[f].values, feats[config.TARGET].values) for f in FEATURES),
        "leaky columns excluded from FEATURES": not {"volume_11kw", "occupancy", "duration",
                                                    config.TARGET} & set(FEATURES),
        "train < val < test (no time overlap)": bool(tr.max() < va.min() and va.max() < te.min()),
        "each timestamp in exactly one split": bool(feats.groupby("time").split.nunique().max() == 1),
        "no missing feature values": bool(feats[FEATURES].notna().all().all()),
    }
