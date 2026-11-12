"""Data and leakage tests. Run with:  python -m pytest -q"""
import pytest

from src import config
from src.data_loader import missing_files

pytestmark = pytest.mark.skipif(bool(missing_files()), reason="dataset not available")


@pytest.fixture(scope="module")
def feats():
    from src.feature_engineering import load_features
    return load_features()


def test_panel_is_complete_and_clean():
    from src.data_loader import load_panel
    from src.preprocessing import validate_panel
    report = validate_panel(load_panel())
    assert report["zones"] == 275 and report["timestamps"] == 4344
    assert report["complete grid"]
    assert sum(report["missing values"].values()) == 0


def test_no_leakage(feats):
    from src.feature_engineering import leakage_checks
    checks = leakage_checks(feats)
    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, failed


def test_chronological_split(feats):
    assert feats.loc[feats.split == "train", "time"].max() <= __import__("pandas").Timestamp(config.TRAIN_END)
    assert feats.loc[feats.split == "test", "time"].min().month == 2
