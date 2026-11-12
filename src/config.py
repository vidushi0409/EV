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

# Official mainland-China public holidays inside the data period
HOLIDAYS = (
    ["2022-09-10", "2022-09-11", "2022-09-12"]                      # Mid-Autumn
    + [f"2022-10-0{d}" for d in range(1, 8)]                         # National Day
    + ["2022-12-31", "2023-01-01", "2023-01-02"]                     # New Year
    + [f"2023-01-{d}" for d in range(21, 28)]                        # Spring Festival
)
