"""Streamlit app tests (headless). Run with:  python -m pytest -q"""
import pytest
from streamlit.testing.v1 import AppTest

import src.ui as ui
from src.evaluate_models import read_best_model

pytestmark = pytest.mark.skipif(read_best_model() is None, reason="models not trained")

PAGES = ["home", "demand_forecast", "demand_map", "zone_insights", "compare_zones", "about_model"]


def run_page(name):
    at = AppTest.from_file(f"app_pages/{name}.py", default_timeout=180)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_errors(page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_app_entry_point_runs():
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    assert not at.exception


def test_forecast_button_returns_prediction():
    at = run_page("demand_forecast")
    at.button[0].click().run()
    labels = {m.label: m.value for m in at.metric}
    assert "Predicted demand" in labels and labels["Predicted demand"].endswith("kWh")
    assert "Actual demand" in labels


@pytest.mark.parametrize("page", ["demand_map", "zone_insights", "compare_zones", "about_model"])
def test_download_buttons_present(page):
    at = run_page(page)
    assert len(at.get("download_button")) >= 1


def test_missing_dataset_shows_message(monkeypatch):
    monkeypatch.setattr(ui, "missing_files", lambda: ["data/.../volume.csv"])
    at = run_page("demand_forecast")
    assert not at.exception
    assert any("dataset was not found" in e.value for e in at.error)


def test_untrained_model_shows_message(monkeypatch):
    def fail():
        raise FileNotFoundError("LightGBM has not been trained yet. Run: python train.py")
    monkeypatch.setattr(ui, "_load_forecaster", fail)
    at = run_page("zone_insights")
    assert not at.exception
    assert any("not ready" in e.value for e in at.error)


def test_out_of_period_time_is_rejected():
    fc = ui._load_forecaster()
    with pytest.raises(ValueError):
        fc.zone_hour(102, "2022-10-01 10:00")
    with pytest.raises(ValueError):
        fc.zone_day(999999, "2023-02-10")
