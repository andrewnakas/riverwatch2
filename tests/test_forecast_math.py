"""Unit tests for the pure forecast math: anchored bias correction, the asinh
transform pair, q-scale, and inverse-MAE^2 blend weighting.

These functions have no I/O and drive every member's output, so they are the
highest-value things to pin against regression (AUDIT Phase 2)."""
import math
import re

import numpy as np
import pandas as pd
import pytest

from app.forecast import (
    _anchor_to_observed,
    _blend_weights,
    _q_inverse,
    _q_scale,
    _q_transform,
)


# ---------------------------------------------------------------------------
# _anchor_to_observed: join the h=1 forecast to last observed flow with a
# linearly-decaying correction back to the raw forecast by `decay_h`.
# ---------------------------------------------------------------------------
def test_anchor_noop_when_h1_matches_observed():
    pred = [100.0, 110.0, 120.0]
    out = _anchor_to_observed(pred, q_obs=100.0, decay_h=7)
    # delta == 0 -> returned unchanged.
    assert out == pred


def test_anchor_pulls_h1_exactly_to_observed():
    pred = [130.0, 140.0, 150.0]
    out = _anchor_to_observed(pred, q_obs=100.0, decay_h=7)
    # h=1 weight is 1.0, so the full delta is subtracted -> lands on q_obs.
    assert out[0] == pytest.approx(100.0)


def test_anchor_correction_decays_to_zero_by_decay_h():
    decay_h = 4
    raw = [120.0, 120.0, 120.0, 120.0, 120.0]
    out = _anchor_to_observed(raw, q_obs=100.0, decay_h=decay_h)
    # At h = decay_h + 1 the weight hits 0, so the raw forecast is restored.
    assert out[decay_h] == pytest.approx(120.0)
    # Correction magnitude is monotonically non-increasing across horizons.
    corrections = [raw[i] - out[i] for i in range(len(raw))]
    assert all(corrections[i] >= corrections[i + 1] - 1e-9 for i in range(len(corrections) - 1))


def test_anchor_clamps_at_zero():
    # A forecast far above a near-zero observation must never go negative.
    pred = [50.0, 5.0, 1.0]
    out = _anchor_to_observed(pred, q_obs=0.0, decay_h=7)
    assert all(v >= 0.0 for v in out)


def test_anchor_handles_empty_and_nonfinite():
    assert _anchor_to_observed([], q_obs=10.0, decay_h=7) == []
    pred = [float("nan"), 10.0]
    # delta non-finite -> returned unchanged (no crash).
    out = _anchor_to_observed(pred, q_obs=5.0, decay_h=7)
    assert math.isnan(out[0])


# ---------------------------------------------------------------------------
# asinh transform pair + q-scale.
# ---------------------------------------------------------------------------
def test_q_transform_inverse_roundtrip():
    scale = 25.0
    q = np.array([0.0, 1.0, 25.0, 1000.0, 50000.0])
    z = _q_transform(q, scale)
    back = _q_inverse(z, scale)
    np.testing.assert_allclose(back, q, rtol=1e-6, atol=1e-6)


def test_q_transform_well_behaved_at_zero():
    assert _q_transform(0.0, 10.0) == pytest.approx(0.0)


def test_q_inverse_clamps_negative_to_zero():
    # Negative asinh inputs invert to negative flow; must clamp at 0.
    out = _q_inverse(np.array([-5.0, -0.1]), 10.0)
    assert np.all(out >= 0.0)


def test_q_scale_uses_positive_median_floored_at_one():
    s = _q_scale(pd.Series([0.0, 0.0, 10.0, 30.0]))
    assert s == pytest.approx(20.0)  # median of [10, 30]


def test_q_scale_all_zero_returns_one():
    assert _q_scale(pd.Series([0.0, 0.0, 0.0])) == 1.0


def test_q_scale_floor_at_one_for_tiny_flows():
    assert _q_scale(pd.Series([0.1, 0.2, 0.3])) == 1.0


# ---------------------------------------------------------------------------
# _blend_weights: inverse-MAE^2, normalized, with a tiny floor for bad members.
# ---------------------------------------------------------------------------
def test_blend_weights_sum_to_one():
    w = _blend_weights({"a": 10.0, "b": 20.0}, ["a", "b"])
    assert sum(w.values()) == pytest.approx(1.0)


def test_blend_weights_inverse_square_ratio():
    # Half the MAE -> 4x the (pre-normalization) weight; preserved after norm.
    w = _blend_weights({"a": 10.0, "b": 20.0}, ["a", "b"])
    assert w["a"] / w["b"] == pytest.approx(4.0)


def test_blend_weights_floors_invalid_members():
    # Zero / NaN / missing MAE -> tiny 0.01 weight, never dominates.
    w = _blend_weights({"good": 5.0, "bad": 0.0, "nan": float("nan")}, ["good", "bad", "nan"])
    assert w["good"] > w["bad"]
    assert w["good"] > w["nan"]
    assert sum(w.values()) == pytest.approx(1.0)


def test_anchor_decay_zero_disables_anchoring():
    """decay_h=0 must be a strict no-op (v17.1: self-anchoring members like
    mblstm forecast better unanchored — measured on the 2025 harness sweep)."""
    pred = [140.0, 150.0, 160.0]
    assert _anchor_to_observed(pred, q_obs=100.0, decay_h=0) == pred
    assert _anchor_to_observed(pred, q_obs=100.0, decay_h=-1) == pred


# ---------------------------------------------------------------------------
# The snap-to-winner branch must not hand weight to an unmeasured member.
#
# `rolling_mae[...]` is assigned for exactly nine members. `mblstm`, `ealstm`
# and `mblstm_modern` write into `members` but never get an entry, so when the
# snap branch spread its 10% over `members.keys()` they each collected
# 0.1/len(others) of the headline blend with no measured skill behind them.
# Verified live before the fix: forecasts/01052500.json carried
# `weights_strategy: "snap_to:lgbm_pooled"`.
#
# The branch is inline in `forecast_station`, so this is a source contract --
# the same mechanism tests/test_mblstm_modern.py uses to pin its hook.
# ---------------------------------------------------------------------------

def _snap_branch_source() -> str:
    """Comment lines are dropped, so prose that merely NAMES the old broken
    expression cannot trip the check (the mistake this helper was written to
    survive) -- mirroring tests/test_mblstm_modern.py's `code_only`."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "forecast.py").read_text()
    i = src.index("# Decisive winner: 90% on it")
    j = src.index("weights_strategy = f\"snap_to:", i)
    block = src[i:j]
    return "\n".join(l for l in block.splitlines() if not l.lstrip().startswith("#"))


def test_snap_to_winner_spreads_only_over_measured_members():
    branch = _snap_branch_source()
    assert "valid_mae" in branch, \
        "the 10% must be spread over members with a measured MAE"
    assert not re.search(r"members\.keys\(\)", branch), \
        "spreading over members.keys() gives weight to unmeasured members"


def test_valid_mae_requires_a_finite_positive_measured_mae():
    """The set the snap branch now draws from must itself be the measured one,
    or the fix above is cosmetic."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "forecast.py").read_text()
    i = src.index("valid_mae = {")
    decl = src[i:src.index("if len(valid_mae)", i)]
    assert "math.isfinite" in decl and "> 0" in decl and "in members" in decl
