"""PREREG §V16-1: forward-only weather refresh under RW2_NO_FETCH=1.

Off by default (the served pipeline must be byte-for-byte what it was until
the flag is flipped). On: today's forecast and the missing recent history are
fetched, nothing older; a failed fetch falls back to the cache; repeated
failures stop the process from trying; null-precipitation days are not merged.
"""
import json
from datetime import date, timedelta

import pytest

from app import weather

LAT, LON = 45.0, -110.0
TODAY = date.today()


def _payload(days, precip=1.0):
    return {"daily": {"time": [d.isoformat() for d in days],
                      **{v: [precip] * len(days) for v in weather.DAILY_VARS}}}


@pytest.fixture()
def wx(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "CACHE_DIR", tmp_path / "c")
    monkeypatch.setattr(weather, "RECORDS_DIR", tmp_path / "r")
    (tmp_path / "c").mkdir()
    (tmp_path / "r").mkdir()
    monkeypatch.setattr(weather, "NO_FETCH", True)
    monkeypatch.setattr(weather, "WX_REFRESH", True)
    monkeypatch.setattr(weather, "_refresh_budget", 100)
    monkeypatch.setattr(weather, "_refresh_fail_streak", 0)
    calls = []
    monkeypatch.setattr(weather, "_calls", calls, raising=False)
    return calls


def _stale_fcst():
    old = TODAY - timedelta(days=30)
    p = weather._cache_path(LAT, LON, old, old + timedelta(days=16), "fcst")
    p.write_text(json.dumps(_payload([old + timedelta(days=i) for i in range(16)])))


def test_off_by_default_serves_the_stale_cache(wx, monkeypatch):
    monkeypatch.setattr(weather, "WX_REFRESH", False)
    monkeypatch.setattr(weather, "_http_json", lambda *a, **k: pytest.fail("fetched with the flag off"))
    _stale_fcst()
    df = weather.fetch_forecast(LAT, LON, days=16)
    assert df["date"].min() == TODAY - timedelta(days=30)


def test_on_fetches_todays_forecast_and_caches_it(wx, monkeypatch):
    _stale_fcst()
    n = []
    monkeypatch.setattr(weather, "_http_json",
                        lambda url, timeout=60: n.append(url) or _payload([TODAY + timedelta(days=i) for i in range(16)]))
    assert weather.fetch_forecast(LAT, LON, days=16)["date"].min() == TODAY
    assert weather.fetch_forecast(LAT, LON, days=16)["date"].min() == TODAY
    assert len(n) == 1, "second call must hit today's cache"


def test_failure_falls_back_then_breaker_stops_trying(wx, monkeypatch):
    _stale_fcst()
    n = []

    def boom(url, timeout=60):
        n.append(url)
        raise OSError("429")
    monkeypatch.setattr(weather, "_http_json", boom)
    for _ in range(9):
        assert weather.fetch_forecast(LAT, LON, days=16)["date"].min() == TODAY - timedelta(days=30)
    assert len(n) == weather._REFRESH_MAX_FAIL_STREAK and not weather.refresh_allowed()


def test_history_forward_delta_only_and_null_days_not_merged(wx, monkeypatch):
    start = TODAY - timedelta(days=400)
    last = TODAY - timedelta(days=40)
    have = {(start + timedelta(days=i)).isoformat(): {v: 0.0 for v in weather.DAILY_VARS}
            for i in range((last - start).days + 1)}
    weather._record_path(LAT, LON).write_text(json.dumps(
        {"rows": have, "first_known": start.isoformat(), "last_known": last.isoformat(),
         "fetched_at": 1.0, "schema_version": weather.SCHEMA_VERSION}))
    urls = []

    def fake(url, timeout=60):
        urls.append(url)
        days = [last - timedelta(days=7) + timedelta(days=i) for i in range(47)]
        p = _payload(days, precip=2.0)
        p["daily"]["precipitation_sum"][-3:] = [None, None, None]   # archive not published yet
        return p
    monkeypatch.setattr(weather, "_http_json", fake)
    end = TODAY - timedelta(days=1)
    df = weather.fetch_history(LAT, LON, start - timedelta(days=3000), end)
    assert len(urls) == 1 and f"start_date={(last - timedelta(days=7)).isoformat()}" in urls[0], \
        "must never issue the backward backfill"
    assert df["date"].max() == end - timedelta(days=3)
    rec = json.loads(weather._record_path(LAT, LON).read_text())
    assert rec["last_known"] == (end - timedelta(days=3)).isoformat()


def test_budget_bounds_calls(wx, monkeypatch):
    monkeypatch.setattr(weather, "_refresh_budget", 1)
    monkeypatch.setattr(weather, "_http_json", lambda url, timeout=60: _payload([TODAY]))
    assert weather.refresh_json("u") is not None and weather.refresh_json("u") is None
