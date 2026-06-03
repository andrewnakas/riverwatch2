"""End-to-end tests for the AUDIT Phase 1 guardrails on `forecast_station`:
staleness flag, member-composition transparency, and the NaN forecast floor.

We build a synthetic, network-free `StationInputs` (sinusoidal seasonal flow +
mild noise) and drive `forecast_station` directly. Foundation models are
monkeypatched off so the test is fast and hermetic — the ridge + persistence
members alone are enough to exercise the guardrail code paths."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

import app.forecast as fc
from app.forecast import (
    StationInputs,
    _build_features,
    _feature_columns,
    _q_scale,
    forecast_station,
)


@pytest.fixture(autouse=True)
def _disable_foundation_models(monkeypatch):
    """Keep tests hermetic/fast: no Chronos/TTM/TimesFM weight downloads."""
    monkeypatch.setattr(fc, "chronos_forecast", lambda *a, **k: None)
    monkeypatch.setattr(fc, "ttm_forecast", lambda *a, **k: None)
    monkeypatch.setattr(fc, "timesfm_forecast", lambda *a, **k: None)
    # Ensure the optional members stay gated off regardless of the host env.
    for var in ("RW2_ENABLE_NWM", "RW2_ENABLE_NWM_RESIDUAL",
                "RW2_ENABLE_TIMESFM_XREG", "RW2_ENABLE_SNOTEL"):
        monkeypatch.delenv(var, raising=False)


def _make_inputs(last_obs_date: dt.date, *, n_days: int = 900, horizon: int = 14):
    """Synthesize a StationInputs ending on `last_obs_date`."""
    dates = pd.date_range(end=pd.Timestamp(last_obs_date), periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy()
    rng = np.random.default_rng(42)
    # Seasonal flow: spring-melt hump, never negative.
    q = 200.0 + 150.0 * np.sin(2 * np.pi * (doy - 90) / 365.0)
    q = np.clip(q + rng.normal(0, 10.0, size=n_days), 1.0, None)
    q_hist = pd.DataFrame({"date": dates.date, "q_cfs": q})

    wx_cols = ["date"] + fc.weather.DAILY_VARS
    wx_hist = pd.DataFrame(columns=wx_cols)
    wx_fcst = pd.DataFrame(columns=wx_cols)

    feats = _build_features(q_hist, pd.concat([wx_hist, wx_fcst], ignore_index=True))
    cols = _feature_columns(feats)
    qs = float(feats.attrs.get("q_scale", _q_scale(q_hist["q_cfs"])))
    return StationInputs(
        station_id="00000001",
        lat=45.0,
        lon=-110.0,
        horizon=horizon,
        today=dt.date.today(),
        q_hist=q_hist,
        wx_hist=wx_hist,
        wx_fcst=wx_fcst,
        snotel_df=None,
        snotel_meta=None,
        feats=feats,
        cols=cols,
        qs=qs,
        has_clim=bool(feats.attrs.get("has_climatology")),
        notes=[],
        attrs=None,
    )


def test_fresh_gauge_not_stale():
    inp = _make_inputs(dt.date.today())
    f = forecast_station("00000001", 45.0, -110.0, inputs=inp)
    assert f.stale is False
    assert f.data_age_days is not None and f.data_age_days <= 1
    assert not f.degraded


def test_stale_gauge_flagged():
    old = dt.date.today() - dt.timedelta(days=10)
    inp = _make_inputs(old)
    f = forecast_station("00000001", 45.0, -110.0, inputs=inp)
    assert f.stale is True
    assert f.data_age_days >= 9
    assert any("stale" in n for n in f.notes)


def test_stale_threshold_env_override(monkeypatch):
    monkeypatch.setenv("RW2_STALE_AFTER_DAYS", "30")
    inp = _make_inputs(dt.date.today() - dt.timedelta(days=10))
    f = forecast_station("00000001", 45.0, -110.0, inputs=inp)
    assert f.stale is False  # 10d age is under the 30d override


def test_members_used_reports_finite_contributors():
    inp = _make_inputs(dt.date.today())
    f = forecast_station("00000001", 45.0, -110.0, inputs=inp)
    # Persistence is always present and finite; foundation models are off.
    assert "persistence_lag1" in f.members_used
    for dropped_off in ("chronos_bolt", "ttm", "timesfm"):
        assert dropped_off not in f.members_used


def test_blend_never_nan_at_h1():
    inp = _make_inputs(dt.date.today())
    f = forecast_station("00000001", 45.0, -110.0, inputs=inp)
    v = f.blend[0]["q_cfs"]
    assert v is not None and np.isfinite(v)
