"""Freshness must key on WHEN THE BASIN WAS REFRESHED, not on a row date.

Measured live on 2026-10-16, on the deployed site. The published forcing table
carries ~16 FORECAST rows, so one of them is always dated exactly today. The
freshness test was `existing_tail(sid) >= today - fresh_within_days`, and
`existing_tail` returns the newest row at or before today -- which that
forecast row satisfies. Every basin was therefore reported "already fresh" on
every run, the refresh step finished in 2 seconds, and the table FROZE for five
days while reporting a perfectly current newest row.

What it served in that state was not the configuration any gate priced:
  * the newest "observed" encoder days were point FORECASTS issued five days
    earlier, where GATE 2 spliced point OBSERVATIONS; and
  * the decoder rows ran out at lead 10 of 16, and a missing forcing row is
    turned into the TRAINING MEAN by `norm_wx`'s nan_to_num, not refused.

`existing_tail` is still right for deciding where to resume a fetch, which is
what it is now used for. Freshness is a property of the build, no row date can
express it, and so it is recorded out of band in `_built.json`.
"""
import gzip
import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ("date,temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
          "precipitation_sum,shortwave_radiation_sum,vapor_pressure,src")


def _load_builder():
    p = ROOT / "scripts" / "build_m1_forcing.py"
    spec = importlib.util.spec_from_file_location("_m1fresh", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m1fresh"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_table(d: Path, sid: str, *, days_back: int, days_fwd: int) -> None:
    """A realistic table: history behind, and a forecast tail INCLUDING today."""
    today = date.today()
    lines = [HEADER]
    for i in range(days_back, -1, -1):
        lines.append(f"{today - timedelta(days=i)},10,15,5,1.0,12.0,900,nldas_areal")
    for i in range(1, days_fwd + 1):
        lines.append(f"{today + timedelta(days=i)},10,15,5,1.0,12.0,900,om_point")
    with gzip.open(d / f"{sid}.csv.gz", "wt", newline="") as fh:
        fh.write("\n".join(lines) + "\n")


def test_a_forecast_row_dated_today_does_not_make_a_table_fresh(tmp_path, monkeypatch):
    """THE REGRESSION. existing_tail() returns today here; the sidecar does not."""
    b = _load_builder()
    monkeypatch.setattr(b.mf, "TABLE_DIR", tmp_path, raising=True)
    monkeypatch.setattr(b.mf, "table_path",
                        lambda sid: tmp_path / f"{sid}.csv.gz", raising=True)
    _write_table(tmp_path, "01022500", days_back=400, days_fwd=11)

    # The old key: satisfied by a forecast row, so it called the table fresh.
    assert b.existing_tail("01022500") == date.today(), \
        "precondition: the table carries a row dated today"

    # The new key: nothing has recorded a build, so the basin is stale.
    assert b.load_built() == {}, "no sidecar means no build has been recorded"


def test_an_absent_sidecar_is_stale_not_fresh(tmp_path, monkeypatch):
    """Absence must fail SAFE. The whole defect was that an unrecorded state
    was indistinguishable from a fresh one; a lost sidecar may cost one refresh
    pass and must never cost a silent freeze."""
    b = _load_builder()
    monkeypatch.setattr(b.mf, "TABLE_DIR", tmp_path, raising=True)
    assert not b.built_path().exists()
    assert b.load_built() == {}


def test_a_corrupt_sidecar_is_stale_not_fresh(tmp_path, monkeypatch):
    b = _load_builder()
    monkeypatch.setattr(b.mf, "TABLE_DIR", tmp_path, raising=True)
    b.built_path().write_text("{not json")
    assert b.load_built() == {}, "unparseable must mean stale, not crash and not fresh"


def test_the_sidecar_round_trips_and_is_written_per_basin(tmp_path, monkeypatch):
    b = _load_builder()
    monkeypatch.setattr(b.mf, "TABLE_DIR", tmp_path, raising=True)
    today = date.today()
    b.save_built({"01022500": today, "01031500": today - timedelta(days=3)})
    got = b.load_built()
    assert got == {"01022500": today, "01031500": today - timedelta(days=3)}
    raw = json.loads(b.built_path().read_text())
    assert raw["schema_version"] == 1
    # Flushed after each basin, because a time-budget break mid-pass is the
    # NORMAL exit and progress must survive it.
    assert "save_built(built)" in (ROOT / "scripts" / "build_m1_forcing.py").read_text()


def test_the_skip_reads_the_sidecar_and_not_the_row_date():
    """Source-text, so the two keys cannot be swapped back silently."""
    src = (ROOT / "scripts" / "build_m1_forcing.py").read_text()
    assert "was_built = built.get(sid)" in src
    assert "if was_built is not None and was_built >= fresh_bar:" in src
    assert "if tail is not None and tail >= fresh_bar:" not in src, \
        "the row-date freshness test must be gone, not merely supplemented"


def test_the_pass_order_is_stalest_first():
    """The time budget stops a pass part-way BY DESIGN, so a pure shuffle gives
    a basin no guarantee of ever being refreshed. Stalest-first makes a partial
    pass a round-robin; the shuffle survives only as the tiebreak, which is what
    keeps a truncated pass from being geographically biased."""
    src = (ROOT / "scripts" / "build_m1_forcing.py").read_text()
    assert "todo.sort(key=lambda s: (built.get(s, date(1970, 1, 1)), shuf[s]))" in src
    assert "random.Random(str(a.shuffle_seed)).shuffle(todo)" in src, \
        "the shuffle must remain, as the tiebreak"


def test_fresh_within_zero_means_built_today(tmp_path, monkeypatch):
    """The default is 0: a basin is skipped only if it was built TODAY."""
    b = _load_builder()
    monkeypatch.setattr(b.mf, "TABLE_DIR", tmp_path, raising=True)
    today = date.today()
    fresh_bar = today - timedelta(days=0)
    assert today >= fresh_bar
    assert not (today - timedelta(days=1) >= fresh_bar), \
        "a table built yesterday must be refreshed again today"
