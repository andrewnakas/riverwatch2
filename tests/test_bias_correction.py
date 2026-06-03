"""Unit tests for AUDIT Phase 4 per-horizon bias correction
(app.forecast._clamp_bias and _per_horizon_bias_scales)."""
import numpy as np
import pytest

from app.forecast import (
    _BIAS_SCALE_MAX,
    _BIAS_SCALE_MIN,
    _clamp_bias,
    _per_horizon_bias_scales,
)


def test_clamp_bias_basic_ratio():
    # obs twice the pred -> scale 2.0 (also the clamp ceiling).
    assert _clamp_bias(200.0, 100.0) == pytest.approx(2.0)


def test_clamp_bias_respects_bounds():
    assert _clamp_bias(1000.0, 100.0) == pytest.approx(_BIAS_SCALE_MAX)  # would be 10
    assert _clamp_bias(10.0, 1000.0) == pytest.approx(_BIAS_SCALE_MIN)   # would be 0.01


def test_clamp_bias_none_when_degenerate():
    assert _clamp_bias(0.0, 100.0) is None
    assert _clamp_bias(100.0, 0.0) is None


def _tuple(offset, yhat, ytrue):
    return (offset, np.array(yhat, dtype=float), np.array(ytrue, dtype=float))


def test_per_horizon_distinct_scales():
    # 3 offsets, 2 horizons. h1: pred too low (obs/pred=2). h2: pred too high
    # (obs/pred=0.5). A pooled scale would smear these together; per-horizon
    # must recover both independently.
    preds = [
        _tuple(o, [50.0, 200.0], [100.0, 100.0])
        for o in (0, 30, 60)
    ]
    pooled, scales_h = _per_horizon_bias_scales(preds, horizon=2)
    assert scales_h[1] == pytest.approx(2.0)   # 100/50
    assert scales_h[2] == pytest.approx(0.5)    # 100/200
    # Pooled (mean obs / mean pred = 200/250 = 0.8) sits between — the exact
    # mediocre compromise per-horizon avoids.
    assert pooled == pytest.approx(0.8)


def test_sparse_horizon_falls_back_to_pooled():
    # h1 has 3 samples (enough); h2 has only 1 (below _BIAS_MIN_SAMPLES=3) so
    # it must inherit the pooled scale rather than a noisy single-sample ratio.
    preds = [
        _tuple(0, [50.0, 999.0], [100.0, 100.0]),
        _tuple(30, [50.0], [100.0]),   # h2 missing
        _tuple(60, [50.0], [100.0]),   # h2 missing
    ]
    pooled, scales_h = _per_horizon_bias_scales(preds, horizon=2)
    assert scales_h[1] == pytest.approx(2.0)
    assert scales_h[2] == pytest.approx(pooled)


def test_empty_preds_returns_none():
    pooled, scales_h = _per_horizon_bias_scales([], horizon=14)
    assert pooled is None
    assert scales_h == {}


def test_nonfinite_pairs_ignored():
    preds = [
        _tuple(0, [50.0, float("nan")], [100.0, 100.0]),
        _tuple(30, [50.0, float("inf")], [100.0, 100.0]),
        _tuple(60, [50.0, 50.0], [100.0, 100.0]),
    ]
    pooled, scales_h = _per_horizon_bias_scales(preds, horizon=2)
    assert scales_h[1] == pytest.approx(2.0)
    # h2 has only one finite pair -> below threshold -> pooled fallback.
    assert scales_h[2] == pytest.approx(pooled)
