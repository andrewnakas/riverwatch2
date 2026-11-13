"""PREREG §V16 H-LEAK: the `q_obs_today_log` stacker feature.

The served champion fills it, in training, with `ytrue_h[0]` -- the lead-1
target. The challenger (`q_obs_at_offset` passed) must fill it with the flow
observed on the holdout's ISSUE day, and the champion's rows must stay exactly
as they were, because the pair may differ in one thing only.
"""
import math
from types import SimpleNamespace

import numpy as np

from app.forecast import challenger_line
from app.stacker import MEMBER_ORDER, STATIC_NAMES, StackerTrainer

QCOL = len(MEMBER_ORDER) + STATIC_NAMES.index("q_obs_today_log")
ATTRS = {"drain_area_sqmi": 100.0, "lat": 45.0, "lon": -110.0, "alt_ft": 1000.0}
QS = 50.0
# one holdout at offset 30: issue-day flow 80, then the river DOUBLES on day 1
PREDS = {"persistence_lag1": [(30, [80.0, 80.0, 80.0], [160.0, 170.0, 180.0])]}


def _rows(**kw):
    t = StackerTrainer(horizon=3)
    t.add_station("X", ATTRS, QS, PREDS, {30: 200}, **kw)
    return {h: t._rows[h][0] for h in (1, 2, 3)}


def test_champion_feature_is_the_lead1_target():
    r = _rows()
    for h in (1, 2, 3):
        assert math.isclose(r[h][0][0, QCOL], math.asinh(160.0 / QS), rel_tol=1e-6)
    # ...which at lead 1 IS the label: the leak, pinned so it cannot change silently.
    assert math.isclose(r[1][0][0, QCOL], r[1][1][0], rel_tol=1e-6)


def test_challenger_feature_is_the_issue_day_flow():
    r = _rows(q_obs_at_offset={30: 80.0})
    for h in (1, 2, 3):
        assert math.isclose(r[h][0][0, QCOL], math.asinh(80.0 / QS), rel_tol=1e-6)
    assert not math.isclose(r[1][0][0, QCOL], r[1][1][0], rel_tol=1e-3)


def test_arms_differ_in_that_one_column_only():
    a, b = _rows(), _rows(q_obs_at_offset={30: 80.0})
    for h in (1, 2, 3):
        diff = ~np.isclose(a[h][0], b[h][0], equal_nan=True)
        assert diff.sum() == 1 and diff[0, QCOL]
        assert np.array_equal(a[h][1], b[h][1])


def test_missing_issue_day_flow_is_nan_not_the_target():
    r = _rows(q_obs_at_offset={})
    assert math.isnan(r[1][0][0, QCOL])


class _FakeStacker:
    fitted = True

    def predict_blend(self, **kw):
        return [10.0, None, float("nan")]


def test_challenger_line_falls_back_to_champion_value_and_never_raises():
    panel = {"attrs": ATTRS, "qs": QS, "members_panel": {}, "q_obs_today": 80.0,
             "issued_doy": 200, "target_doys": [201, 202, 203]}
    f = SimpleNamespace(station_id="X", _v14_4_panel=panel,
                        blend_strategy_per_h={1: "lgbm_stacker", 2: "lgbm_stacker", 3: "median_asinh"},
                        blend=[{"date": f"2026-11-{14 + i}", "q_cfs": 100.0 + i} for i in range(3)])
    assert challenger_line(f, _FakeStacker()) == [
        {"date": "2026-11-14", "q_cfs": 10.0}, {"date": "2026-11-15", "q_cfs": 101.0},
        {"date": "2026-11-16", "q_cfs": 102.0}]
    assert challenger_line(SimpleNamespace(_v14_4_panel=None), _FakeStacker()) == []
    assert challenger_line(f, object()) == []


def test_challenger_is_silent_where_the_champion_did_not_use_the_stacker():
    class S:
        fitted = True

        def predict_blend(self, **kw):
            return [10.0, 20.0, 30.0]
    panel = {"attrs": ATTRS, "qs": QS, "members_panel": {}, "q_obs_today": 80.0,
             "issued_doy": 200, "target_doys": [201, 202, 203]}
    f = SimpleNamespace(station_id="X", _v14_4_panel=panel,
                        blend_strategy_per_h={1: "lgbm_stacker", 2: "mean", 3: "lgbm_stacker"},
                        blend=[{"date": str(i), "q_cfs": 100.0} for i in range(3)])
    assert [p["q_cfs"] for p in challenger_line(f, S())] == [10.0, 100.0, 30.0]
