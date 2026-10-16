"""Concurrent fetching must not change what gets written, or when it stops.

The per-basin cost of a refresh is HTTP latency -- measured ~13 s per basin
from a GitHub runner with zero failures, so it is not quota and not CPU. Served
serially, a 35-minute budget reached only about 160 of 490 basins, which meant
the published table could never be fully fresh however the freshness test was
written. Overlapping the fetches is the fix.

What must survive it, because each one was earned:
  * the time budget stops the pass cleanly, so the publish steps still run (a
    run killed by the job timeout publishes nothing and repeats the cold start);
  * the consecutive-failure bail still returns 2, which is the registered
    "budget gone, come back next window" code -- without it, ~352 doomed
    requests per window kept the quota pinned and starved the NEXT window too;
  * freshness is still read per basin from the sidecar;
  * and every table written is byte-for-byte what the serial path would write.

Writes and the sidecar stay on the main thread; only the fetches overlap.
"""
import gzip
import importlib.util
import json
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_m1conc", ROOT / "scripts" / "build_m1_forcing.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m1conc"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Tracker:
    """Counts how many fetches are in flight at once."""

    def __init__(self, delay=0.02, fail_ids=()):
        self.lock = threading.Lock()
        self.cur = 0
        self.peak = 0
        self.calls = []
        self.delay = delay
        self.fail_ids = set(fail_ids)

    def make(self, mod, latlon):
        by_coord = {v: k for k, v in latlon.items()}

        def fake(lat, lon, start, end, *, horizon_days, past_days, retries, sleep_s):
            sid = by_coord[(lat, lon)]
            with self.lock:
                self.cur += 1
                self.peak = max(self.peak, self.cur)
                self.calls.append(sid)
            try:
                time.sleep(self.delay)
                if sid in self.fail_ids:
                    return None
                idx = pd.date_range(start, end)
                return pd.DataFrame({
                    "date": [d.date() for d in idx],
                    "temperature_2m_mean": 10.0, "temperature_2m_max": 15.0,
                    "temperature_2m_min": 5.0, "precipitation_sum": 1.0,
                    "shortwave_radiation_sum": 12.0, "vapor_pressure": 900.0,
                })
            finally:
                with self.lock:
                    self.cur -= 1
        return fake


def _sandbox(tmp_path, mod, ids):
    td = tmp_path / "m1_forcing"
    td.mkdir(parents=True, exist_ok=True)
    latlon = {sid: (40.0 + i * 0.01, -100.0 - i * 0.01) for i, sid in enumerate(ids)}
    return td, latlon


def _run(mod, monkeypatch, tmp_path, ids, *, workers, tracker, extra=()):
    td, latlon = _sandbox(tmp_path, mod, ids)
    monkeypatch.setattr(mod.mf, "TABLE_DIR", td, raising=True)
    monkeypatch.setattr(mod.mf, "table_path",
                        lambda sid: td / f"{sid}.csv.gz", raising=True)
    monkeypatch.setattr(mod, "load_cohort", lambda p: list(ids), raising=True)
    monkeypatch.setattr(mod, "load_latlon", lambda p: dict(latlon), raising=True)
    monkeypatch.setattr(mod, "fetch_basin", tracker.make(mod, latlon), raising=True)
    argv = ["build_m1_forcing.py", "--workers", str(workers), "--sleep", "0",
            "--keep-days", "60", "--horizon-days", "16", *extra]
    monkeypatch.setattr(sys, "argv", argv, raising=True)
    rc = mod.main()
    return rc, td


IDS = [f"{i:08d}" for i in range(1, 13)]


def test_serial_and_parallel_write_identical_tables(tmp_path, monkeypatch):
    """The ONLY thing that may differ is wall-clock."""
    mod = _load()
    t1 = _Tracker()
    rc1, d1 = _run(mod, monkeypatch, tmp_path / "a", IDS, workers=1, tracker=t1)
    t4 = _Tracker()
    rc4, d4 = _run(mod, monkeypatch, tmp_path / "b", IDS, workers=4, tracker=t4)
    assert rc1 == 0 and rc4 == 0
    assert t1.peak == 1, f"workers=1 must stay serial, saw {t1.peak} in flight"
    assert 1 < t4.peak <= 4, f"workers=4 should overlap, peak was {t4.peak}"
    assert sorted(t1.calls) == sorted(t4.calls) == IDS
    for sid in IDS:
        a = gzip.open(d1 / f"{sid}.csv.gz", "rt").read()
        b = gzip.open(d4 / f"{sid}.csv.gz", "rt").read()
        assert a == b, f"{sid}: parallel output differs from serial"


def test_never_more_in_flight_than_workers(tmp_path, monkeypatch):
    mod = _load()
    t = _Tracker(delay=0.03)
    rc, _ = _run(mod, monkeypatch, tmp_path, IDS, workers=3, tracker=t)
    assert rc == 0
    assert t.peak <= 3, f"peak in flight {t.peak} exceeded workers=3"


def test_the_quota_bail_still_returns_2_under_concurrency(tmp_path, monkeypatch):
    """Exit 2 is the registered "budget gone, come back next window" code, and
    the whole point of the bail is to stop spending a dead quota."""
    mod = _load()
    t = _Tracker(fail_ids=IDS)          # every basin fails
    rc, _ = _run(mod, monkeypatch, tmp_path, IDS, workers=4, tracker=t,
                 extra=["--max-consecutive-failures", "4"])
    assert rc == 2, f"expected the registered bail code 2, got {rc}"
    assert len(t.calls) < len(IDS), \
        f"bailed after {len(t.calls)} of {len(IDS)} basins -- it must stop early"


def test_the_sidecar_records_every_basin_written(tmp_path, monkeypatch):
    mod = _load()
    t = _Tracker()
    rc, td = _run(mod, monkeypatch, tmp_path, IDS, workers=4, tracker=t)
    assert rc == 0
    built = json.loads((td / "_built.json").read_text())["built"]
    assert sorted(built) == IDS
    assert set(built.values()) == {date.today().isoformat()}


def test_a_basin_built_today_is_not_refetched(tmp_path, monkeypatch):
    mod = _load()
    t1 = _Tracker()
    rc, td = _run(mod, monkeypatch, tmp_path, IDS, workers=4, tracker=t1)
    assert rc == 0 and sorted(t1.calls) == IDS
    # Second pass, same day: the sidecar says every basin is fresh.
    t2 = _Tracker()
    monkeypatch.setattr(mod.mf, "TABLE_DIR", td, raising=True)
    monkeypatch.setattr(mod.mf, "table_path",
                        lambda sid: td / f"{sid}.csv.gz", raising=True)
    latlon = {sid: (40.0 + i * 0.01, -100.0 - i * 0.01) for i, sid in enumerate(IDS)}
    monkeypatch.setattr(mod, "load_cohort", lambda p: list(IDS), raising=True)
    monkeypatch.setattr(mod, "load_latlon", lambda p: dict(latlon), raising=True)
    monkeypatch.setattr(mod, "fetch_basin", t2.make(mod, latlon), raising=True)
    monkeypatch.setattr(sys, "argv",
                        ["build_m1_forcing.py", "--workers", "4", "--sleep", "0"],
                        raising=True)
    assert mod.main() == 0
    assert t2.calls == [], f"refetched {len(t2.calls)} already-fresh basins"
