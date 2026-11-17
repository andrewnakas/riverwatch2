"""current.json rows: the map's marker colours depend on these classes."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import current_conditions as cc  # noqa: E402

ROW = {"month_day": "11-01", "min_va": 100, "p25_va": 200, "p50_va": 300,
       "p75_va": 400, "max_va": 900}


def test_classify_bands():
    assert [cc.classify(q, ROW) for q in (50, 150, 300, 500, 1000)] == [1, 2, 3, 4, 5]
    assert cc.classify(200, ROW) == 3 and cc.classify(400, ROW) == 3
    assert cc.classify(None, ROW) == 0
    assert cc.classify(300, None) == 0
    assert cc.classify(300, {"p25_va": None, "p75_va": 4}) == 0


def test_record_needs_enough_years():
    short = dict(ROW, count=4)
    assert cc.classify(50, short) == 2 and cc.classify(1000, short) == 4
    assert cc.classify(1000, dict(ROW, count=10)) == 5


def test_summarise_uses_last_obs_and_blend_peak(tmp_path):
    payload = {
        "history": [{"date": "2026-10-31", "q_cfs": 120}, {"date": "2026-11-01", "q_cfs": 250}],
        "blend": [{"date": "2026-11-02", "q_cfs": 260}, {"date": "2026-11-03", "q_cfs": 500},
                  {"date": "2026-11-04", "q_cfs": None}],
        "daily_stats": {"rows": [ROW, dict(ROW, month_day="11-03")]},
    }
    (tmp_path / "01010000.json").write_text(json.dumps(payload))
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "empty.json").write_text(json.dumps({"history": []}))
    out = cc.build(tmp_path, flood_file=tmp_path / "absent.json", today=__import__("datetime").date(2026, 11, 2))
    assert list(out["sites"]) == ["01010000"]
    row = dict(zip(out["fields"], out["sites"]["01010000"]))
    assert row == {"q": 250, "date": "2026-11-01", "cls": 3, "peak_q": 500,
                   "peak_date": "2026-11-03", "peak_cls": 4, "ratio": 2.0,
                   "near_q": 500, "near_date": "2026-11-03", "near_cls": 4, "near_ratio": 2.0,
                   "flood": None, "near_flood": None}


def test_near_peak_ignores_days_past_the_window(tmp_path):
    blend = [{"date": f"2026-11-{d:02d}", "q_cfs": q}
             for d, q in [(2, 260), (6, 300), (7, 900), (14, 5000)]]
    payload = {"history": [{"date": "2026-11-01", "q_cfs": 250}], "blend": blend,
               "daily_stats": {"rows": [dict(ROW, month_day=f"11-{d:02d}") for d in range(1, 15)]}}
    (tmp_path / "01010000.json").write_text(json.dumps(payload))
    row = dict(zip(*(lambda o: (o["fields"], o["sites"]["01010000"]))(cc.build(tmp_path, flood_file=tmp_path / "absent.json", today=__import__("datetime").date(2026, 11, 2)))))
    assert (row["peak_q"], row["peak_cls"]) == (5000, 5)          # day 13: record
    assert (row["near_q"], row["near_date"], row["near_cls"]) == (300, "2026-11-06", 3)
    assert row["near_ratio"] == 1.2


def test_flood_categories_from_nws_thresholds(tmp_path):
    fl = tmp_path / "nws_flood.json"
    fl.write_text(json.dumps({"fields": ["lid", "action", "minor", "moderate", "major"],
                              "sites": {"01010000": ["XXXM1", 240, 400, None, 900],
                                        "01020000": ["YYYM1", None, None, None, None]}}))
    blend = [{"date": "2026-11-02", "q_cfs": 260}, {"date": "2026-11-04", "q_cfs": 450},
             {"date": "2026-11-12", "q_cfs": 2000}]       # day 11: outside the near window
    for sid in ("01010000", "01020000", "01030000"):
        (tmp_path / f"{sid}.json").write_text(json.dumps(
            {"history": [{"date": "2026-11-01", "q_cfs": 250}], "blend": blend}))
    import datetime as _dt
    rows_today = {}
    out = {"fields": cc.FIELDS, "sites": {}}
    for sid in ("01010000", "01020000", "01030000"):
        thr = cc.load_flood(fl).get(sid)
        out["sites"][sid] = cc.summarise(json.loads((tmp_path / f"{sid}.json").read_text()), thr,
                                         today=_dt.date(2026, 11, 2))
    rows = {k: dict(zip(out["fields"], v)) for k, v in out["sites"].items()}
    assert (rows["01010000"]["flood"], rows["01010000"]["near_flood"]) == (1, 2)   # action now, minor by day 3
    # NWS point without thresholds, and no NWS point: unknown, never "none".
    assert (rows["01020000"]["flood"], rows["01020000"]["near_flood"]) == (None, None)
    assert rows["01030000"]["near_flood"] is None
    assert cc.flood_cat(100, [240, 400, None, 900]) == 0
    assert cc.flood_cat(1000, [240, 400, None, 900]) == 4


def test_flood_needs_fresh_obs_and_plausible_thresholds():
    import datetime as _dt
    payload = {"history": [{"date": "2026-11-01", "q_cfs": 250}],
               "blend": [{"date": "2026-11-02", "q_cfs": 260}],
               "daily_stats": {"rows": [dict(ROW, month_day="11-01", p75_va=300)]}}
    thr = [240, 400, None, 900]
    f = lambda **k: dict(zip(cc.FIELDS, cc.summarise(payload, thr, **k)))["flood"]
    # action threshold 240 sits below p75 (300) -> dropped; 250 cfs is below minor.
    assert f(today=_dt.date(2026, 11, 2)) == 0
    # a last observation 30 days old gets no flood category at all.
    assert f(today=_dt.date(2026, 12, 1)) is None
    assert cc.plausible_thresholds([0.0, 13, 5000], {"p75_va": 100}) == [None, None, 5000]


def test_stale_observation_is_unknown_class():
    import datetime as _dt
    payload = {"history": [{"date": "1986-09-29", "q_cfs": 2050}],
               "blend": [{"date": "1986-09-30", "q_cfs": 9000}],
               "daily_stats": {"rows": [dict(ROW, month_day="09-29"), dict(ROW, month_day="09-30")]}}
    r = dict(zip(cc.FIELDS, cc.summarise(payload, today=_dt.date(2026, 11, 16))))
    assert (r["cls"], r["peak_cls"], r["near_cls"]) == (0, 0, 0)
    assert r["q"] == 2050                      # the value itself is still shown
    fresh = dict(zip(cc.FIELDS, cc.summarise(payload, today=_dt.date(1986, 10, 1))))
    assert fresh["peak_cls"] == 5


def test_zero_record_high_is_not_a_record():
    from current_conditions import classify
    row = {"min_va": 0, "p25_va": 0, "p75_va": 0, "max_va": 0, "count": 40}
    assert classify(5.0, row) == 4
    assert classify(5.0, dict(row, max_va=2.0, p75_va=1.0)) == 5
