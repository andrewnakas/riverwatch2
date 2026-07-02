"""Unit tests for the ECMWF IFS ENS ensemble-mean forecast path (v17.1).

`weather.fetch_forecast_ecmwf_ens` feeds the MB-LSTM decoder when
RW2_MBLSTM_FCST=ecmwf_ens. Everything here is hermetic: `_http_json` is
monkeypatched (no network) and CACHE_DIR is redirected to tmp_path so the
real SD-card-backed cache is never touched. Covered:

  * member averaging (control + memberNN columns collapse to the mean)
  * request shape (Ensemble API URL, models=ecmwf_ifs025, 5 daily vars)
  * cache write/reuse per fetch_forecast conventions
  * units passthrough sanity (daily aggregates arrive in corpus units:
    °C / mm / MJ/m² — no conversion should be applied)
  * forecast.py wiring: env flag routes the ens frame to mblstm.forecast,
    and any fetch exception falls back to the shared wx_fcst with a note
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

import app.weather as weather


def _payload(n_days=15, tmean=(10.0, 20.0, 30.0), precip=(0.0, 1.5, 3.0),
             srad=(10.0, 20.0, 30.0)):
    """Fake Ensemble API response: control column + two member columns per
    var, constant across days so the expected mean is trivial."""
    start = dt.date.today()
    times = [(start + dt.timedelta(days=i)).isoformat() for i in range(n_days)]
    daily = {"time": times}
    for var, vals in [
        ("temperature_2m_mean", tmean),
        ("temperature_2m_max", tuple(v + 5.0 for v in tmean)),
        ("temperature_2m_min", tuple(v - 5.0 for v in tmean)),
        ("precipitation_sum", precip),
        ("shortwave_radiation_sum", srad),
    ]:
        daily[var] = [vals[0]] * n_days
        daily[f"{var}_member01"] = [vals[1]] * n_days
        daily[f"{var}_member02"] = [vals[2]] * n_days
    return {"daily": daily}


@pytest.fixture()
def _isolated(monkeypatch, tmp_path):
    """No network, no real cache dir."""
    monkeypatch.setattr(weather, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(weather, "NO_FETCH", False)
    return tmp_path


def test_ens_mean_across_members(_isolated, monkeypatch):
    calls = []

    def fake_http(url, timeout=60):
        calls.append(url)
        return _payload()

    monkeypatch.setattr(weather, "_http_json", fake_http)
    df = weather.fetch_forecast_ecmwf_ens(46.7, -68.5, days=14)

    assert len(df) == 15
    assert list(df.columns) == ["date"] + weather.ENS_DAILY_VARS
    # (10 + 20 + 30) / 3 across control + 2 members, no unit conversion.
    assert np.allclose(df["temperature_2m_mean"], 20.0)
    assert np.allclose(df["temperature_2m_max"], 25.0)
    assert np.allclose(df["temperature_2m_min"], 15.0)
    assert np.allclose(df["precipitation_sum"], 1.5)
    assert np.allclose(df["shortwave_radiation_sum"], 20.0)
    # Dates are python dates starting today, consecutive daily.
    assert df["date"].iloc[0] == dt.date.today()
    deltas = pd.to_datetime(df["date"]).diff().dropna()
    assert (deltas == pd.Timedelta(days=1)).all()

    # Request shape: Ensemble API endpoint, pinned model, the 5 daily vars.
    assert len(calls) == 1
    url = calls[0]
    assert url.startswith(weather.ENSEMBLE_URL)
    assert "models=ecmwf_ifs025" in url
    for var in weather.ENS_DAILY_VARS:
        assert var in url


def test_ens_mean_skips_nan_members(_isolated, monkeypatch):
    payload = _payload(n_days=3)
    # Knock one member out on day 0 for tmean: mean of remaining {10, 30}.
    payload["daily"]["temperature_2m_mean_member01"][0] = None
    monkeypatch.setattr(weather, "_http_json", lambda url, timeout=60: payload)
    df = weather.fetch_forecast_ecmwf_ens(46.7, -68.5, days=14)
    assert df["temperature_2m_mean"].iloc[0] == pytest.approx(20.0)
    assert df["temperature_2m_mean"].iloc[1] == pytest.approx(20.0)


def test_ens_cache_reused_within_max_age(_isolated, monkeypatch):
    calls = []

    def fake_http(url, timeout=60):
        calls.append(url)
        return _payload()

    monkeypatch.setattr(weather, "_http_json", fake_http)
    df1 = weather.fetch_forecast_ecmwf_ens(46.7, -68.5, days=14)
    df2 = weather.fetch_forecast_ecmwf_ens(46.7, -68.5, days=14)
    assert len(calls) == 1  # second call served from the ensfcst cache file
    pd.testing.assert_frame_equal(df1, df2)
    assert list(_isolated.glob("ensfcst_46.700_-68.500_*.json"))


def test_ens_empty_payload_gives_empty_frame(_isolated, monkeypatch):
    monkeypatch.setattr(weather, "_http_json", lambda url, timeout=60: {"daily": {}})
    df = weather.fetch_forecast_ecmwf_ens(46.7, -68.5, days=14)
    assert df.empty
    assert list(df.columns) == ["date"] + weather.ENS_DAILY_VARS


# ---------------------------------------------------------------------------
# forecast.py wiring: RW2_MBLSTM_FCST=ecmwf_ens routing + graceful fallback.
# Reuses the hermetic StationInputs pattern from test_forecast_guardrails.
# ---------------------------------------------------------------------------

import app.forecast as fc  # noqa: E402
import app.mblstm as mblstm_mod  # noqa: E402
from app.forecast import StationInputs, _build_features, _feature_columns, _q_scale  # noqa: E402


@pytest.fixture(autouse=True)
def _hermetic_members(monkeypatch):
    monkeypatch.setattr(fc, "chronos_forecast", lambda *a, **k: None)
    monkeypatch.setattr(fc, "ttm_forecast", lambda *a, **k: None)
    monkeypatch.setattr(fc, "timesfm_forecast", lambda *a, **k: None)
    for var in ("RW2_ENABLE_NWM", "RW2_ENABLE_NWM_RESIDUAL",
                "RW2_ENABLE_TIMESFM_XREG", "RW2_ENABLE_SNOTEL",
                "RW2_MBLSTM_FCST"):
        monkeypatch.delenv(var, raising=False)


def _make_inputs(horizon=14, n_days=900):
    dates = pd.date_range(end=pd.Timestamp(dt.date.today()), periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy()
    rng = np.random.default_rng(7)
    q = np.clip(200.0 + 150.0 * np.sin(2 * np.pi * (doy - 90) / 365.0)
                + rng.normal(0, 10.0, size=n_days), 1.0, None)
    q_hist = pd.DataFrame({"date": dates.date, "q_cfs": q})
    wx_cols = ["date"] + fc.weather.DAILY_VARS
    wx_hist = pd.DataFrame(columns=wx_cols)
    wx_fcst = pd.DataFrame(columns=wx_cols)
    feats = _build_features(q_hist, pd.concat([wx_hist, wx_fcst], ignore_index=True))
    cols = _feature_columns(feats)
    qs = float(feats.attrs.get("q_scale", _q_scale(q_hist["q_cfs"])))
    return StationInputs(
        station_id="00000001", lat=46.7, lon=-68.5, horizon=horizon,
        today=dt.date.today(), q_hist=q_hist, wx_hist=wx_hist, wx_fcst=wx_fcst,
        snotel_df=None, snotel_meta=None, feats=feats, cols=cols, qs=qs,
        has_clim=bool(feats.attrs.get("has_climatology")), notes=[], attrs=None,
    )


def test_flag_routes_ens_frame_to_mblstm(monkeypatch):
    monkeypatch.setenv("RW2_MBLSTM_FCST", "ecmwf_ens")
    sentinel = pd.DataFrame({
        "date": [dt.date.today() + dt.timedelta(days=i) for i in range(1, 16)],
        **{v: [1.0] * 15 for v in weather.ENS_DAILY_VARS},
    })
    monkeypatch.setattr(weather, "fetch_forecast_ecmwf_ens",
                        lambda lat, lon, days=14, **k: sentinel)
    seen = {}

    def fake_mblstm(q_hist, wx_hist, wx_fcst, static_attrs, horizon):
        seen["wx_fcst"] = wx_fcst
        return None

    monkeypatch.setattr(mblstm_mod, "forecast", fake_mblstm)
    inp = _make_inputs()
    fc.forecast_station("00000001", 46.7, -68.5, inputs=inp)
    assert seen["wx_fcst"] is sentinel


def test_flag_unset_keeps_shared_wx_fcst(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ensemble fetch must not run when flag is unset")

    monkeypatch.setattr(weather, "fetch_forecast_ecmwf_ens", boom)
    seen = {}

    def fake_mblstm(q_hist, wx_hist, wx_fcst, static_attrs, horizon):
        seen["wx_fcst"] = wx_fcst
        return None

    monkeypatch.setattr(mblstm_mod, "forecast", fake_mblstm)
    inp = _make_inputs()
    fc.forecast_station("00000001", 46.7, -68.5, inputs=inp)
    assert seen["wx_fcst"] is inp.wx_fcst


def test_fetch_failure_falls_back_to_shared_wx_fcst(monkeypatch):
    monkeypatch.setenv("RW2_MBLSTM_FCST", "ecmwf_ens")

    def boom(*a, **k):
        raise RuntimeError("ensemble api down")

    monkeypatch.setattr(weather, "fetch_forecast_ecmwf_ens", boom)
    seen = {}

    def fake_mblstm(q_hist, wx_hist, wx_fcst, static_attrs, horizon):
        seen["wx_fcst"] = wx_fcst
        return None

    monkeypatch.setattr(mblstm_mod, "forecast", fake_mblstm)
    inp = _make_inputs()
    f = fc.forecast_station("00000001", 46.7, -68.5, inputs=inp)
    assert seen["wx_fcst"] is inp.wx_fcst
    assert any("ecmwf_ens forcing failed" in n for n in f.notes)


def test_empty_ens_frame_falls_back_with_note(monkeypatch):
    monkeypatch.setenv("RW2_MBLSTM_FCST", "ecmwf_ens")
    empty = pd.DataFrame(columns=["date"] + weather.ENS_DAILY_VARS)
    monkeypatch.setattr(weather, "fetch_forecast_ecmwf_ens",
                        lambda lat, lon, days=14, **k: empty)
    seen = {}

    def fake_mblstm(q_hist, wx_hist, wx_fcst, static_attrs, horizon):
        seen["wx_fcst"] = wx_fcst
        return None

    monkeypatch.setattr(mblstm_mod, "forecast", fake_mblstm)
    inp = _make_inputs()
    f = fc.forecast_station("00000001", 46.7, -68.5, inputs=inp)
    assert seen["wx_fcst"] is inp.wx_fcst
    assert any("ecmwf_ens forcing empty" in n for n in f.notes)
