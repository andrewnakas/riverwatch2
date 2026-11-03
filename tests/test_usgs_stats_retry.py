"""USGS daily statistics: transient failures are retried, real 4xx answers are not."""
import io
from urllib.error import HTTPError

from app import usgs_stats

URL = "https://waterservices.usgs.gov/nwis/stat/?sites=1"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch(monkeypatch, outcomes):
    calls = []

    def fake(req, timeout=60):
        calls.append(1)
        o = outcomes[len(calls) - 1]
        if isinstance(o, Exception):
            raise o
        return _Resp(o.encode())

    monkeypatch.setattr(usgs_stats, "urlopen", fake)
    monkeypatch.setattr(usgs_stats.time, "sleep", lambda s: None)
    return calls


def test_retries_transient_then_succeeds(monkeypatch):
    calls = _patch(monkeypatch, [TimeoutError(), HTTPError("u", 503, "x", {}, None), "ok"])
    assert usgs_stats._fetch_with_retry(URL) == "ok"
    assert len(calls) == 3


def test_gives_up_after_budget(monkeypatch):
    calls = _patch(monkeypatch, [TimeoutError()] * 5)
    assert usgs_stats._fetch_with_retry(URL) is None
    assert len(calls) == 3


def test_does_not_retry_a_real_404(monkeypatch):
    calls = _patch(monkeypatch, [HTTPError("u", 404, "x", {}, None), "ok"])
    assert usgs_stats._fetch_with_retry(URL) is None
    assert len(calls) == 1


def _cold(monkeypatch, tmp_path, budget=60):
    monkeypatch.setattr(usgs_stats, "NO_FETCH", True)
    monkeypatch.setattr(usgs_stats, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(usgs_stats, "_cold_budget", budget)
    monkeypatch.setattr(usgs_stats, "_cold_fail_streak", 0)


RDB = ("agency_cd\tsite_no\tparameter_cd\tts_id\tloc_web_ds\tmonth_nu\tday_nu\tbegin_yr\tend_yr\tcount_nu"
       "\tmax_va_yr\tmax_va\tmin_va_yr\tmin_va\tmean_va\tp25_va\tp50_va\tp75_va\n"
       "5s\t15s\t5s\t10n\t15s\t3n\t3n\t6n\t6n\t8n\t5n\t12s\t5n\t12s\t12s\t12s\t12s\t12s\n"
       "USGS\t01\t00060\t1\t\t1\t1\t1931\t2026\t96\t1949\t2200\t1932\t38.0\t264\t113\t192\t335\n")


def test_cache_only_build_fetches_a_cold_site_and_caches_it(monkeypatch, tmp_path):
    """Scheduled builds run RW2_NO_FETCH=1; a never-cached site must still fetch."""
    _cold(monkeypatch, tmp_path)
    calls = _patch(monkeypatch, [RDB])
    d = usgs_stats.fetch_daily_stats("01")
    assert d and d["rows"] and (tmp_path / "01.json").exists() and len(calls) == 1
    assert usgs_stats.fetch_daily_stats("01")["rows"]          # now from cache
    assert len(calls) == 1


def test_circuit_breaker_stops_cold_fetching(monkeypatch, tmp_path):
    _cold(monkeypatch, tmp_path)
    calls = _patch(monkeypatch, [TimeoutError()] * 100)
    for i in range(10):
        assert usgs_stats.fetch_daily_stats(f"s{i}") is None
    assert len(calls) == 5 * 3                                 # 5 sites x 3 attempts, then stop
