"""The stacker must predict from the same member values it was trained on.

It is fit on RAW holdout predictions; serving it the bias-scaled, anchored
members inflated the live blend (median relative bias +0.31 at lead 7 against
persistence's +0.09, 610 basins, 2026-10-13..11-02).
"""
from app.forecast import _stacker_members_panel


def test_panel_uses_raw_member_output_not_served():
    raw = {"runoff_ridge": ([100.0, 110.0, 120.0], 7)}
    served = {
        "runoff_ridge": [{"q_cfs": 200.0}, {"q_cfs": 220.0}, {"q_cfs": 240.0}],  # x2 bias scale
        "persistence_lag1": [{"q_cfs": 90.0}] * 3,                               # no raw entry
    }
    p = _stacker_members_panel(raw, served, 3)
    assert p["runoff_ridge"] == [100.0, 110.0, 120.0]
    assert p["persistence_lag1"] == [90.0, 90.0, 90.0]


def test_panel_pads_short_series_with_none():
    p = _stacker_members_panel({"ttm": ([5.0], 4)}, {"ttm": [{"q_cfs": 9.0}]}, 3)
    assert p["ttm"] == [5.0, None, None]
