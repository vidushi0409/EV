"""Data preprocessing: validation, cleaning and chronological splitting."""
import pandas as pd

from src import config


def validate_panel(panel):
    """Run data-quality checks on the long panel and return a report dict.

    The UrbanEV zone files are already cleaned by the dataset authors; these
    checks confirm that and fail loudly if the files were altered."""
    key_dups = int(panel.duplicated(["time", "zone"]).sum())
    value_cols = list(config.HOURLY_FILES.values())
    report = {
        "rows": int(len(panel)),
        "zones": int(panel.zone.nunique()),
        "timestamps": int(panel.time.nunique()),
        "duplicate (time, zone) rows": key_dups,
        "missing values": {c: int(panel[c].isna().sum()) for c in value_cols},
        "negative values": {c: int((panel[c] < 0).sum()) for c in value_cols},
        "complete grid": int(len(panel)) == panel.zone.nunique() * panel.time.nunique(),
    }
    if key_dups or not report["complete grid"]:
        raise ValueError(f"Panel failed validation: {report}")
    return report


def clean_panel(panel):
    """Apply the (minimal) cleaning steps needed.

    * Remove exact duplicate (time, zone) rows if any (none in the original data).
    * Missing values: forward-fill within each zone, then 0 (none in the original data).
    * Negative values are impossible for energy/price/occupancy -> clipped at 0.
    * Data types: time -> datetime64, zone -> int32, measurements -> float32.
    """
    panel = panel.drop_duplicates(["time", "zone"]).copy()
    panel["time"] = pd.to_datetime(panel["time"])
    panel["zone"] = panel["zone"].astype("int32")
    value_cols = list(config.HOURLY_FILES.values())
    panel = panel.sort_values(["zone", "time"], ignore_index=True)
    panel[value_cols] = (panel.groupby("zone")[value_cols].ffill().fillna(0)
                         .clip(lower=0).astype("float32"))
    return panel


def split_label(times):
    """Return 'train' / 'val' / 'test' for each timestamp (chronological split)."""
    times = pd.to_datetime(pd.Series(times))
    labels = pd.Series("test", index=times.index)
    labels[times <= config.VAL_END] = "val"
    labels[times <= config.TRAIN_END] = "train"
    return labels.values


def chronological_split(df):
    """Split a feature table by time. All zones of one timestamp stay together."""
    labels = split_label(df["time"])
    return (df[labels == "train"], df[labels == "val"], df[labels == "test"])
