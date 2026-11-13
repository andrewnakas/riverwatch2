"""Fleet scoreboard (PREREG §V16-0): known-value controls.

An arm that IS the observation must score ratio 0; an arm identical to the
reference must score exactly 1 with a CI of [1, 1]; partial-day targets must
be dropped; and the §LIVE-1 window must not be scoreable from this script.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import score_fleet_live as sf  # noqa: E402


def _frame():
    rows = []
    for s, base in (("A", 100.0), ("B", 1000.0), ("C", 10.0)):
        for i in range(5):
            for h in (1, 2):
                rows.append({"issued_date": f"2026-11-{18 + i:02d}", "station_id": s,
                             "target_date": f"2026-11-{17 + i + h:02d}", "horizon_day": h,
                             "blend": base * 1.1, "persistence_lag1": base * 1.2,
                             "perfect": base, "band_lo": base * 0.9, "band_hi": base * 1.3})
    df = pd.DataFrame(rows)
    obs = df[["station_id", "target_date"]].drop_duplicates().rename(columns={"target_date": "date"})
    obs["obs"] = obs.station_id.map({"A": 100.0, "B": 1000.0, "C": 10.0})
    return df, obs


def test_known_values():
    df, obs = _frame()
    r = sf.score(df, obs, [("blend", "persistence_lag1"), ("perfect", "persistence_lag1"),
                           ("blend", "blend")], today="2026-12-31")
    t = r["pairs"]["blend|persistence_lag1"]["horizon_day"][1]
    assert abs(t["mae_ratio"] - 0.5) < 1e-3 and t["frac_basins_better"] == 1.0
    assert abs(t["bias_arm"] - 0.1) < 1e-3 and abs(t["bias_ref"] - 0.2) < 1e-3
    assert t["band90_coverage"] == 1.0 and t["basins"] == 3
    assert r["pairs"]["perfect|persistence_lag1"]["horizon_day"][2]["mae_ratio"] == 0.0
    same = r["pairs"]["blend|blend"]["horizon_day"][1]
    assert (same["mae_ratio"], same["ci_lo"], same["ci_hi"]) == (1.0, 1.0, 1.0)


def test_calendar_lead_differs_from_horizon_day():
    df, obs = _frame()
    r = sf.score(df, obs, [("blend", "persistence_lag1")], today="2026-12-31")
    # horizon 1 targets the issue day itself here (the H-LAG situation)
    assert set(r["pairs"]["blend|persistence_lag1"]["horizon_day"]) == {1, 2}
    assert set(r["pairs"]["blend|persistence_lag1"]["cal_lead"]) == {0, 1}


def test_partial_day_targets_dropped():
    df, obs = _frame()
    r = sf.score(df, obs, [("blend", "persistence_lag1")], today="2026-11-19")
    assert r["targets"][1] < "2026-11-19"


def test_rdb_parse_two_blocks():
    txt = ("# a\nagency_cd\tsite_no\tdatetime\t1_00060_00003\t1_00060_00003_cd\n5s\t15s\t20d\t14n\t10s\n"
           "USGS\t01010000\t2026-11-12\t1540\tP\nUSGS\t01010000\t2026-11-13\tIce\tP\n"
           "# b\nagency_cd\tsite_no\tdatetime\t9_00065_00003\t9_00065_00003_cd\t7_00060_00003\t7_00060_00003_cd\n"
           "5s\t15s\t20d\t14n\t10s\t14n\t10s\nUSGS\t02020000\t2026-11-12\t3.1\tP\t77\tP\n")
    o = sf.parse_rdb(txt)
    assert o.values.tolist() == [["01010000", "2026-11-12", 1540.0], ["02020000", "2026-11-12", 77.0]]


def test_input_age():
    m = pd.DataFrame({"issued_date": ["2026-11-14"] * 4, "wx_fcst_start": ["2026-11-14", "2026-10-11", "2026-11-01", None],
                      "wx_hist_end": [None] * 4, "nbm_start": ["2026-11-14"] * 4, "last_obs_date": ["2026-11-12"] * 4})
    r = sf.input_age(m)
    assert r["wx_fcst_start"]["median_d"] == 13.0 and r["wx_fcst_start"]["missing_frac"] == 0.25
    assert r["wx_hist_end"] == {"missing_frac": 1.0} and r["last_obs_date"]["frac_ge_2d"] == 1.0


def test_live1_window_guard_is_a_constant_the_cli_enforces():
    assert sf.LIVE1_GUARD == "2026-11-18"
    assert "max(a.issued_from or LIVE1_GUARD, LIVE1_GUARD)" in Path(sf.__file__).read_text()
