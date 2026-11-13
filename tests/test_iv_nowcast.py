"""PREREG §V16 H-LAG: IV-derived daily means close the daily-value gap.

Only complete local days; contiguous from the daily record's last day; never
today; the frame is returned untouched when there is nothing to add.
"""
from datetime import date

import pandas as pd

from app import iv_nowcast as iv


def _rdb(rows):
    head = "# x\nagency_cd\tsite_no\tdatetime\ttz_cd\t1_00060\t1_00060_cd\n5s\t15s\t20d\t6s\t14n\t10s\n"
    return head + "".join(f"USGS\t{s}\t{t}\tEST\t{q}\tP\n" for s, t, q in rows)


def _day(site, d, q, hours=range(24)):
    return [(site, f"{d} {h:02d}:{m:02d}", q) for h in hours for m in (0, 30)]


def test_complete_days_only():
    txt = _rdb(_day("01", "2026-11-11", 10) + _day("01", "2026-11-12", 20)
               + _day("01", "2026-11-13", 99, hours=range(8))        # today, partial
               + _day("02", "2026-11-12", 5, hours=range(6, 20)))    # outage: 14 h
    got = iv.daily_means_from_rdb(txt)
    assert got == {"01": {"2026-11-11": 10.0, "2026-11-12": 20.0}}


def test_ice_and_negative_values_dropped():
    rows = _day("01", "2026-11-12", 20)
    rows[3] = ("01", rows[3][1], "Ice")
    rows[4] = ("01", rows[4][1], "-999999")
    assert iv.daily_means_from_rdb(_rdb(rows)) == {"01": {"2026-11-12": 20.0}}


def _hist(last="2026-11-11"):
    return pd.DataFrame({"date": pd.to_datetime(["2026-11-10", last]), "q_cfs": [1.0, 2.0]})


def test_extend_appends_contiguous_days_before_today():
    t = {"01": {"2026-11-12": 20.0, "2026-11-13": 30.0}}
    out = iv.extend("01", _hist(), today=date(2026, 11, 13), table=t)
    assert out["q_cfs"].tolist() == [1.0, 2.0, 20.0] and out.attrs["iv_days"] == 1
    assert str(out["date"].iloc[-1].date()) == "2026-11-12" and out["date"].dtype == _hist()["date"].dtype


def test_extend_needs_contiguity_and_leaves_frame_alone_otherwise():
    h = _hist(last="2026-11-10")
    assert iv.extend("01", h, today=date(2026, 11, 13), table={"01": {"2026-11-12": 20.0}}) is h
    assert iv.extend("zz", h, today=date(2026, 11, 13), table={}) is h
    assert iv.extend("01", h.iloc[0:0], today=date(2026, 11, 13), table={"01": {}}) is not None


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("RW2_IV_NOWCAST", raising=False)
    assert not iv.enabled()
