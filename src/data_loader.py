"""Loading and inspection of the UrbanEV dataset.

Verified structure (see notebook Section 1):
* Each hourly file in charge_1hour/ is WIDE: a `time` column + one column per
  zone ID (275 zones, 4,344 hourly rows, 2022-09-01 00:00 -> 2023-02-28 23:00).
* zone-information.csv holds static zone attributes keyed by TAZID (= zone ID).
* station_information.csv files map station_id -> TAZID.
* pile_rated_power.csv maps pile_id -> station_id with power (kW) and AC/DC type.
* adj.csv / distance.csv are 275 x 275 matrices whose column order equals the
  zone column order of the hourly files (row i corresponds to column i).
"""
import json

import numpy as np
import pandas as pd

from src import config


class DatasetNotFoundError(FileNotFoundError):
    """Raised when required dataset files are missing."""


def missing_files():
    """Return the list of required dataset files that do not exist."""
    return [str(p) for p in config.REQUIRED_FILES if not p.exists()]


def check_dataset():
    """Raise a helpful error if the dataset is not where the config expects it."""
    missing = missing_files()
    if missing:
        raise DatasetNotFoundError(
            "Dataset files not found. Place the UrbanEV dataset in "
            f"'{config.DATA_DIR}' (or set EV_DATA_DIR). Missing:\n  "
            + "\n  ".join(missing))


# ---------------------------------------------------------------- raw readers
def load_hourly_wide(stem):
    """Read one hourly file as a wide DataFrame (index=time, columns=int zone IDs)."""
    df = pd.read_csv(config.HOURLY_DIR / f"{stem}.csv", parse_dates=["time"])
    df = df.set_index("time")
    df.columns = df.columns.astype(int)
    return df.astype("float32")


def zone_order():
    """Zone IDs in the column order used by hourly files and the matrices."""
    header = pd.read_csv(config.HOURLY_DIR / "volume.csv", nrows=0)
    return [int(c) for c in header.columns if c != "time"]


def load_matrix(path):
    """Read adj.csv / distance.csv as a DataFrame indexed and labelled by zone ID."""
    m = pd.read_csv(path)
    m.columns = m.columns.astype(int)
    m.index = m.columns          # row i belongs to the same zone as column i
    return m


def load_adjacency():
    return load_matrix(config.ADJ_FILE)


def load_distance():
    return load_matrix(config.DISTANCE_FILE)


# ---------------------------------------------------------------- zone attributes
def load_zone_info():
    """Static zone attributes, enriched with charger-capacity features.

    Preprocessing decisions (documented assumptions):
    * Zone 348 has longitude/latitude = 0 (missing). It is replaced by the mean
      coordinates of the stations located in that zone.
    * slow/fast pile counts come from the raw station file; their per-zone sums
      equal zone-information's charge_count exactly (verified).
    * avg_pile_kw = mean rated power of the zone's piles from pile_rated_power.csv
      (piles with 0 kW ignored). The file covers 274 of 275 zones; the missing
      zone gets the median across zones.
    """
    zones = pd.read_csv(config.ZONE_INFO_FILE).rename(columns={"TAZID": "zone"})
    stations = pd.read_csv(config.RAW_STATION_INFO_FILE)

    # 1) repair missing coordinates from the zone's stations
    bad = (zones.longitude < 100) | (zones.latitude < 10)
    centroid = stations.groupby("TAZID")[["longitude", "latitude"]].mean()
    for idx in zones.index[bad]:
        z = zones.at[idx, "zone"]
        zones.loc[idx, ["longitude", "latitude"]] = centroid.loc[z].values

    # 2) slow / fast pile counts
    counts = stations.groupby("TAZID")[["slow_count", "fast_count"]].sum()
    zones = zones.merge(counts, left_on="zone", right_index=True, how="left")
    zones[["slow_count", "fast_count"]] = zones[["slow_count", "fast_count"]].fillna(0)
    zones["fast_share"] = zones.fast_count / zones.charge_count

    # 3) average rated power per pile
    piles = pd.read_csv(config.PILE_POWER_FILE)
    piles = piles[piles.power > 0].merge(
        stations[["station_id", "TAZID"]], on="station_id")
    avg_kw = piles.groupby("TAZID").power.mean().rename("avg_pile_kw")
    zones = zones.merge(avg_kw, left_on="zone", right_index=True, how="left")
    zones["avg_pile_kw"] = zones.avg_pile_kw.fillna(zones.avg_pile_kw.median())
    zones["capacity_kw"] = zones.charge_count * zones.avg_pile_kw

    return zones.rename(columns={"charge_count": "num_piles"})


# ---------------------------------------------------------------- long panel
PANEL_CACHE = config.PROCESSED_DIR / "panel_hourly.parquet"


def build_long_panel():
    """Combine the six hourly wide files into one long table (time, zone, variables)."""
    check_dataset()
    parts = []
    for stem, col in config.HOURLY_FILES.items():
        wide = load_hourly_wide(stem)
        parts.append(wide.stack().rename(col))
    panel = pd.concat(parts, axis=1)
    panel.index.names = ["time", "zone"]
    panel = panel.reset_index().sort_values(["zone", "time"], ignore_index=True)
    panel["zone"] = panel["zone"].astype("int32")
    return panel


def load_panel(use_cache=True):
    """Long panel of all zone-hours; cached as Parquet after the first build."""
    if use_cache and PANEL_CACHE.exists():
        return pd.read_parquet(PANEL_CACHE)
    panel = build_long_panel()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_CACHE, index=False)
    return panel


# ---------------------------------------------------------------- inspection
def inspect_dataset():
    """Return a summary dict describing every relevant file (used by notebook/report)."""
    check_dataset()
    summary = {"hourly_files": {}}
    for stem in config.HOURLY_FILES:
        w = load_hourly_wide(stem)
        steps = w.index.to_series().diff().dropna()
        summary["hourly_files"][stem] = {
            "rows (hours)": int(w.shape[0]), "zone columns": int(w.shape[1]),
            "start": str(w.index.min()), "end": str(w.index.max()),
            "missing values": int(w.isna().sum().sum()),
            "negative values": int((w < 0).sum().sum()),
            "duplicate timestamps": int(w.index.duplicated().sum()),
            "non-hourly gaps": int((steps != pd.Timedelta(hours=1)).sum()),
            "zero share": round(float((w == 0).mean().mean()), 4),
            "min": round(float(w.min().min()), 3), "max": round(float(w.max().max()), 3),
            "mean": round(float(np.nanmean(w.values)), 3),
        }
    order = zone_order()
    raw_zone = pd.read_csv(config.ZONE_INFO_FILE)
    adj, dist = load_adjacency(), load_distance()
    a = adj.values
    stations = pd.read_csv(config.STATION_INFO_FILE)
    raw_st = pd.read_csv(config.RAW_STATION_INFO_FILE)
    piles = pd.read_csv(config.PILE_POWER_FILE)

    v, k = load_hourly_wide("volume").values, load_hourly_wide("volume-11kw").values
    summary.update({
        "zones": len(order),
        "zone_hour_records": int(len(order) * summary["hourly_files"]["volume"]["rows (hours)"]),
        "zone_info": {"rows": int(len(raw_zone)), "columns": list(raw_zone.columns),
                      "zones match hourly columns": sorted(raw_zone.TAZID) == sorted(order),
                      "zones with missing (0,0) coordinates": raw_zone.loc[
                          raw_zone.longitude < 100, "TAZID"].tolist()},
        "station_info_cleaned": {"stations": int(len(stations)),
                                 "columns": list(stations.columns)},
        "station_info_raw": {"stations": int(len(raw_st)), "columns": list(raw_st.columns),
                             "slow+fast == charge_count": bool(
                                 (raw_st.slow_count + raw_st.fast_count == raw_st.charge_count).all())},
        "pile_rated_power": {"piles": int(len(piles)), "stations covered": int(piles.station_id.nunique()),
                             "AC piles": int((piles.pileType == "AC").sum()),
                             "DC piles": int((piles.pileType == "DC").sum())},
        "adjacency": {"shape": list(a.shape), "columns match hourly order": list(adj.columns) == order,
                      "values": sorted(np.unique(a).tolist()), "self-loops on diagonal": bool((np.diag(a) == 1).all()),
                      "asymmetric pairs": int((a != a.T).sum() // 2),
                      "zones with no neighbours": int(((a.sum(1) - np.diag(a)) == 0).sum()),
                      "mean neighbours": round(float((a.sum(1) - np.diag(a)).mean()), 2)},
        "distance": {"shape": list(dist.shape), "columns match hourly order": list(dist.columns) == order,
                     "max distance (m)": round(float(dist.values.max()), 1)},
        "volume_11kw_leakage": {"share of rows with volume_11kw <= volume": round(float((k <= v + 1e-9).mean()), 4),
                                "share of total energy": round(float(k.sum() / v.sum()), 4)},
    })
    return summary


def price_schedule_check():
    """Share of zone-days whose 24-hour price profile equals the previous day.

    A high share means prices follow a published time-of-use tariff and are
    therefore known in advance (safe to use at prediction time)."""
    out = {}
    for stem in ("e_price", "s_price"):
        p = load_hourly_wide(stem).values
        out[stem] = round(float((p[24:] == p[:-24]).mean()), 4)
    return out


CONSISTENCY_FILE = config.ARTIFACTS_DIR / "consistency_5min_vs_1hour.json"


def check_5min_consistency(force=False):
    """Verify that hourly volume == sum of the twelve 5-minute volumes in the hour.

    Reads the 230 MB 5-minute file once; the result is cached as JSON."""
    if CONSISTENCY_FILE.exists() and not force:
        return json.loads(CONSISTENCY_FILE.read_text())
    hourly = load_hourly_wide("volume")
    five = pd.read_csv(config.FIVE_MIN_DIR / "volume.csv", parse_dates=["time"]).set_index("time")
    five.columns = five.columns.astype(int)
    summed = five.resample("1h").sum().reindex(hourly.index)
    diff = (summed - hourly).abs()
    result = {
        "five_min_rows": int(five.shape[0]), "five_min_zones": int(five.shape[1]),
        "same zone columns": list(five.columns) == list(hourly.columns),
        "max_abs_difference_kwh": round(float(diff.max().max()), 6),
        "mean_abs_difference_kwh": round(float(np.nanmean(diff.values)), 6),
        "hourly_equals_sum_of_5min": bool(diff.max().max() < 1e-3),
    }
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CONSISTENCY_FILE.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(inspect_dataset(), indent=2, default=str))
    print("price schedule:", price_schedule_check())
    print("5-min consistency:", check_5min_consistency())
