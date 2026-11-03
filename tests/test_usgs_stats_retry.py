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
