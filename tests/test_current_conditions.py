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
    out = cc.build(tmp_path)
    assert list(out["sites"]) == ["01010000"]
    row = dict(zip(out["fields"], out["sites"]["01010000"]))
    assert row == {"q": 250, "date": "2026-11-01", "cls": 3, "peak_q": 500,
                   "peak_date": "2026-11-03", "peak_cls": 4, "ratio": 2.0}
