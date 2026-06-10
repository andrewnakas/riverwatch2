"""v15.9: nwm_per_h preference order — measured archived-forecast MAE
(holdout_stats.json) first, hindcast×decay second, 0.9×persistence last."""
import pytest

from app import forecast


@pytest.fixture(autouse=True)
def _reset_stats_cache():
    forecast._NWM_HOLDOUT_STATS_CACHED = None
    yield
    forecast._NWM_HOLDOUT_STATS_CACHED = None


def test_measured_stats_preferred_over_formula():
    forecast._NWM_HOLDOUT_STATS_CACHED = {
        "stations": {
            "01234567": {
                "1": {"mae_nwm_corrected": 12.5, "mae_persistence": 15.0, "n": 21},
                "2": {"mae_nwm_corrected": 20.0, "n": 4},  # below _NWM_STATS_MIN_N
            }
        }
    }
    out = forecast._nwm_per_h_estimates("01234567", 3, 100.0, {})
    assert out[1] == 12.5  # measured wins over the formula
    assert out[2] == pytest.approx(100.0 * 1.08)  # too few samples → formula
    assert out[3] == pytest.approx(100.0 * 1.12)


def test_fallback_to_persistence_when_no_hindcast():
    forecast._NWM_HOLDOUT_STATS_CACHED = {}
    out = forecast._nwm_per_h_estimates("x", 2, None, {1: 10.0, 2: 20.0})
    assert out[1] == pytest.approx(9.0)
    assert out[2] == pytest.approx(18.0)


def test_unknown_station_and_no_signals_yields_empty():
    forecast._NWM_HOLDOUT_STATS_CACHED = {}
    assert forecast._nwm_per_h_estimates("x", 3, None, {}) == {}
