"""Streamlit app and prediction-service tests (headless). Run with:  python -m pytest -q"""
import datetime as dt

import pytest
from streamlit.testing.v1 import AppTest

import src.ui as ui
from src.profile_model import PROFILE_INFO
from src.walk_forward import WF_PREDICTIONS

pytestmark = pytest.mark.skipif(not (WF_PREDICTIONS.exists() and PROFILE_INFO.exists()),
                                reason="models not trained")

PAGES = ["home", "demand_forecast", "prediction_vs_actual", "demand_map", "zone_insights",
         "compare_zones", "about_model"]
DATE_PAGES = ["demand_map", "zone_insights", "compare_zones"]


def run_page(name, day=None):
    at = AppTest.from_file(f"app_pages/{name}.py", default_timeout=180)
    at.run()
    if day is not None:
        at.date_input[0].set_value(day).run()
    return at


def no_problems(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_errors(page):
    no_problems(run_page(page))


@pytest.mark.parametrize("page", DATE_PAGES)
def test_page_runs_for_future_date(page):
    at = run_page(page, dt.date(2025, 8, 15))
    no_problems(at)
    assert any("Typical-pattern estimate" in m.value for m in at.markdown)


def test_app_entry_point_runs():
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    no_problems(at)


def test_home_shows_summary():
    assert any("Energy delivered" in m.value for m in run_page("home").markdown)


@pytest.mark.parametrize("day,expect_actual", [(dt.date(2023, 2, 14), True), (dt.date(2025, 8, 15), False)])
def test_forecast_button(day, expect_actual):
    at = run_page("demand_forecast")
    at.date_input[0].set_value(day).run()
    at.button[0].click().run()
    no_problems(at)
    cards = " ".join(m.value for m in at.markdown if 'class="kpi"' in m.value)
    assert "Predicted · " in cards or "Estimate · " in cards
    assert ('<div class="l">Actual</div>' in cards) == expect_actual


@pytest.mark.parametrize("page", ["prediction_vs_actual", "demand_map", "zone_insights",
                                  "compare_zones", "about_model"])
def test_download_buttons_present(page):
    assert len(run_page(page).get("download_button")) >= 1


def test_missing_dataset_shows_message(monkeypatch):
    monkeypatch.setattr(ui, "missing_files", lambda: ["data/.../volume.csv"])
    at = run_page("demand_forecast")
    assert not at.exception
    assert any("dataset was not found" in e.value for e in at.error)


def test_untrained_model_shows_message(monkeypatch):
    def fail():
        raise FileNotFoundError("Models not trained yet. Run: python train.py")
    monkeypatch.setattr(ui, "_load_service", fail)
    at = run_page("zone_insights")
    assert not at.exception
    assert any("not ready" in e.value for e in at.error)


def test_service_modes_and_validation():
    svc = ui._load_service()
    r = svc.zone_hour(102, "2023-02-14 18:00")
    assert r["mode"] == "dataset" and "actual_kwh" in r and set(r["badges"]) == set(r["models"])
    f = svc.zone_hour(102, "2025-08-15 18:00")
    assert f["mode"] == "future" and "actual_kwh" not in f and f["predicted_kwh"] >= 0
    for bad in ("2022-09-15 10:00", "2099-01-01 10:00"):
        with pytest.raises(ValueError):
            svc.zone_hour(102, bad)
    with pytest.raises(ValueError):
        svc.compare_range("2023-02-20", "2023-03-05")


def test_walk_forward_never_trains_on_predicted_month():
    import pandas as pd
    from src import config
    monthly = pd.read_csv(config.ARTIFACTS_DIR / "walk_forward" / "monthly_metrics.csv")
    per_month = monthly[monthly.month != "Oct-Feb (all)"].drop_duplicates("month")
    from src.feature_engineering import load_features
    feats = load_features()
    for _, row in per_month.iterrows():
        start = pd.Timestamp(row.month + "-01")
        assert int(row.train_rows) == int((feats.time < start).sum())
