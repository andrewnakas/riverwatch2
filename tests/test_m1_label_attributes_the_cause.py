"""A gap the label cannot explain sends the reader to the wrong place.

Measured live, week of 2026-10-12: the published label said 7 encoder days were
substituted against the 4 GATE 2 priced, and ended its warning with "(refresh
the basin-average provider)". The provider was perfectly current. NLDAS-2
upstream had stopped publishing complete days -- 2026-10-12 had 13 of 24 hours
and 2026-10-13 had none -- and the appender was correctly refusing to average a
partial day into a daily value. The authenticated HDF5 preflight passed in the
same run, so this was not a credentials problem either.

The remedies are opposite: one is "go fix our pipeline", the other is "wait for
NASA". Naming the wrong one is expensive -- the identical misattribution cost
three days in this project when a script missing from a branch presented as an
Earthdata auth failure.

Only the appender sees the per-hour counts, so it records them and the label
reports them. Same division of labour as `_built.json`, which records a build
time no row date can express.
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


def _load():
    spec = importlib.util.spec_from_file_location(
        "_m1cause", ROOT / "scripts" / "build_m1_label.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m1cause"] = mod
    spec.loader.exec_module(mod)
    return mod


def _table(d: Path, sid: str, gap: int, dec_leads: int = 14):
    today = date.today()
    t0 = today - timedelta(days=1)
    newest_areal = t0 - timedelta(days=gap)
    lines = [HEADER]
    cur = today - timedelta(days=400)
    while cur <= t0:
        src = "nldas_areal" if cur <= newest_areal else "om_point"
        lines.append(f"{cur},10,15,5,1.0,12.0,900,{src}")
        cur += timedelta(days=1)
    for i in range(dec_leads):
        lines.append(f"{today + timedelta(days=i)},10,15,5,1.0,12.0,900,om_point")
    with gzip.open(d / f"{sid}.csv.gz", "wt", newline="") as fh:
        fh.write("\n".join(lines) + "\n")


def _warn(tmp_path, monkeypatch, *, gap, upstream_state):
    lab = _load()
    td = tmp_path / "m1_forcing"
    td.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(lab, "TABLES", td, raising=True)
    sid = sorted(json.loads((ROOT / "data" / "m1_cohort.json").read_text())["cohort"])[0]
    _table(td, sid, gap)
    (td / "_built.json").write_text(json.dumps(
        {"schema_version": 1, "built": {sid: date.today().isoformat()}}))
    if upstream_state is not None:
        (td / "_areal_state.json").write_text(json.dumps(upstream_state))
    mix = lab.measure_forcing()
    g1 = json.loads(lab.GATE1.read_text())
    g2 = json.loads(lab.GATE2.read_text())
    return mix, (lab.build(g1, g2, mix).get("extrapolation_warning") or "")


UPSTREAM_STALLED = {
    "schema_version": 1,
    "newest_complete_upstream_day": "2026-10-11",
    "upstream_frontier_day": "2026-10-12",
    "hours_available_on_skipped_days": {"2026-10-12": 13},
    "upstream_is_incomplete": True,
}
UPSTREAM_FINE = {
    "schema_version": 1,
    "newest_complete_upstream_day": "2026-10-11",
    "upstream_frontier_day": None,
    "hours_available_on_skipped_days": {},
    "upstream_is_incomplete": False,
}


def test_an_upstream_stall_is_named_as_upstream(tmp_path, monkeypatch):
    """THE REGRESSION. The live warning told readers to refresh a provider that
    was current."""
    mix, w = _warn(tmp_path, monkeypatch, gap=7, upstream_state=UPSTREAM_STALLED)
    assert mix["substituted_encoder_days_median"] == 7
    assert "UPSTREAM, not this pipeline" in w
    assert "13 of 24 hours" in w
    assert "2026-10-11" in w and "2026-10-12" in w
    assert "Nothing here needs refreshing" in w
    assert "should be refreshed" not in w, \
        "it must not also tell the reader to fix our provider"


def test_our_own_lag_is_still_named_as_ours(tmp_path, monkeypatch):
    """The opposite branch must survive: when upstream is complete and we are
    behind, the remedy really is to refresh the provider."""
    mix, w = _warn(tmp_path, monkeypatch, gap=7, upstream_state=UPSTREAM_FINE)
    assert "areal provider is behind" in w
    assert "should be refreshed" in w
    assert "UPSTREAM, not this pipeline" not in w


def test_an_unrecorded_upstream_state_says_so(tmp_path, monkeypatch):
    """Absent must read as unknown, not as either answer. Guessing here is how
    a label starts asserting a cause it never measured."""
    mix, w = _warn(tmp_path, monkeypatch, gap=7, upstream_state=None)
    assert "unrecorded" in w
    assert "UPSTREAM, not this pipeline" not in w
    assert "should be refreshed" not in w


def test_a_healthy_gap_raises_nothing_whatever_upstream_says(tmp_path, monkeypatch):
    """The cause only matters when there IS a gap. At the gap GATE 2 measured,
    the label must stay silent even mid-stall."""
    mix, w = _warn(tmp_path, monkeypatch, gap=4, upstream_state=UPSTREAM_STALLED)
    assert mix["substituted_encoder_days_median"] == 4
    assert w == "", f"a healthy configuration must not warn, got: {w}"


def test_the_appender_records_what_only_it_can_see():
    src = (ROOT / "scripts_modern" / "nldas_areal_append.py").read_text()
    assert "_areal_state.json" in src
    assert "newest_complete_upstream_day" in src
    assert "hours_available_on_skipped_days" in src
    assert "write_state(arch, wanted, todo, done_days, skipped)" in src
    # It must be written even when NO day was produced -- that is precisely the
    # stalled case the label needs to explain.
    i = src.index("write_state(arch, wanted, todo, done_days, skipped)")
    j = src.index('print("no days produced")')
    assert i < j, "the state must be written BEFORE the no-days early return"
