"""Central configuration: paths, file names, split dates and constants.

All paths are relative to the project root, so the project runs on any machine.
The dataset location can be overridden with the environment variable EV_DATA_DIR.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get(
    "EV_DATA_DIR", ROOT / "data" / "Dataset" / "UrbanEVDataset" / "UrbanEVDataset"))
ZONE_DIR = DATA_DIR / "20220901-20230228_zone-cleaned-aggregated"
HOURLY_DIR = ZONE_DIR / "charge_1hour"
FIVE_MIN_DIR = ZONE_DIR / "charge_5min"
RAW_DIR = DATA_DIR / "20220901-20230228_station-raw"

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

# ---------------------------------------------------------------- files
# Hourly dynamic variables: file stem -> column name used in the project
HOURLY_FILES = {
    "volume": "volume",            # TARGET: energy delivered (kWh) per zone-hour
    "volume-11kw": "volume_11kw",  # part of volume -> leakage, never a predictor
    "occupancy": "occupancy",
    "duration": "duration",
    "e_price": "e_price",
    "s_price": "s_price",
}
ZONE_INFO_FILE = ZONE_DIR / "zone-information.csv"
STATION_INFO_FILE = ZONE_DIR / "station_information.csv"
ADJ_FILE = ZONE_DIR / "adj.csv"
DISTANCE_FILE = ZONE_DIR / "distance.csv"
RAW_STATION_INFO_FILE = RAW_DIR / "station_information.csv"
PILE_POWER_FILE = RAW_DIR / "pile_rated_power.csv"

REQUIRED_FILES = [HOURLY_DIR / f"{stem}.csv" for stem in HOURLY_FILES] + [
    ZONE_INFO_FILE, STATION_INFO_FILE, ADJ_FILE, DISTANCE_FILE,
    RAW_STATION_INFO_FILE, PILE_POWER_FILE,
]

# ---------------------------------------------------------------- modelling
TARGET = "volume"
RANDOM_SEED = 42

# Chronological split (all zones of a timestamp stay in the same split)
TRAIN_END = "2022-12-31 23:00:00"
VAL_END = "2023-01-31 23:00:00"   # test = Feb 2023

def _days(start, end):
    import pandas as pd
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end)]


# Mainland-China public holiday periods (State Council schedules).
# 2022-09 .. 2023-02 lie inside the dataset. Later years are used only for
# future-date estimates - PLEASE VERIFY them against the official notices.
# Dates after the last listed year are treated as non-holidays.
HOLIDAYS = (
    _days("2022-09-10", "2022-09-12") + _days("2022-10-01", "2022-10-07")
    + _days("2022-12-31", "2023-01-02") + _days("2023-01-21", "2023-01-27")
    # 2023
    + _days("2023-04-05", "2023-04-05") + _days("2023-04-29", "2023-05-03")
    + _days("2023-06-22", "2023-06-24") + _days("2023-09-29", "2023-10-06")
    + _days("2023-12-30", "2024-01-01")
    # 2024
    + _days("2024-02-10", "2024-02-17") + _days("2024-04-04", "2024-04-06")
    + _days("2024-05-01", "2024-05-05") + _days("2024-06-08", "2024-06-10")
    + _days("2024-09-15", "2024-09-17") + _days("2024-10-01", "2024-10-07")
    # 2025
    + _days("2025-01-01", "2025-01-01") + _days("2025-01-28", "2025-02-04")
    + _days("2025-04-04", "2025-04-06") + _days("2025-05-01", "2025-05-05")
    + _days("2025-05-31", "2025-06-02") + _days("2025-10-01", "2025-10-08")
    # 2026
    + _days("2026-01-01", "2026-01-03") + _days("2026-02-15", "2026-02-23")
    + _days("2026-04-04", "2026-04-06") + _days("2026-05-01", "2026-05-05")
    + _days("2026-06-19", "2026-06-21") + _days("2026-09-25", "2026-09-27")
    + _days("2026-10-01", "2026-10-07")
)
HOLIDAY_CALENDAR_END = "2026-12-31"

# Walk-forward: each month is predicted by models trained on all earlier data
WALK_FORWARD_MONTHS = ["2022-10", "2022-11", "2022-12", "2023-01", "2023-02"]
DATASET_END = "2023-02-28 23:00:00"
