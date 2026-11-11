import pytest

from src.ev.main import estimate_range_km


def test_estimate_range():
    assert estimate_range_km(60, 15) == 400


def test_invalid_efficiency():
    with pytest.raises(ValueError):
        estimate_range_km(60, 0)
