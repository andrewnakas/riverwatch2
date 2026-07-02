"""SOTA-metric module sanity tests (app/metrics.py)."""
import numpy as np
import pytest

from app import metrics as M


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _series(rng, n=400):
    # A realistic-ish positive flow series with spread (not flat).
    t = np.arange(n)
    base = 200 + 150 * np.sin(2 * np.pi * t / 365.25)
    return np.clip(base + rng.normal(0, 20, n), 1, None)


def test_perfect_prediction(rng):
    o = _series(rng)
    assert M.nse(o, o) == pytest.approx(1.0)
    assert M.log_nse(o, o) == pytest.approx(1.0)
    assert M.kge(o, o) == pytest.approx(1.0)
    assert M.pearson_r(o, o) == pytest.approx(1.0)
    assert M.pct_bias(o, o) == pytest.approx(0.0, abs=1e-9)
    assert M.fhv(o, o) == pytest.approx(0.0, abs=1e-9)
    assert abs(M.flv(o, o)) < 1e-6


def test_kge_components_bounds(rng):
    o = _series(rng)
    s = o * 1.1 + rng.normal(0, 10, len(o))  # 10% high bias + noise
    kge, r, alpha, beta = M.kge_components(o, s)
    assert kge <= 1.0
    assert -1.0 <= r <= 1.0
    assert alpha > 0 and beta > 0
    assert beta == pytest.approx(1.1, abs=0.05)  # bias ratio recovers


def test_pct_bias_sign(rng):
    o = _series(rng)
    assert M.pct_bias(o, o * 1.2) == pytest.approx(20.0, abs=1e-6)   # over
    assert M.pct_bias(o, o * 0.8) == pytest.approx(-20.0, abs=1e-6)  # under


def test_flat_flow_returns_nan():
    o = np.full(400, 5.0)  # zero variance
    assert np.isnan(M.nse(o, o))
    assert np.isnan(M.kge(o, o))


def test_too_few_points_nan(rng):
    o = _series(rng, n=5)
    assert np.isnan(M.nse(o, o))
    assert np.isnan(M.kge(o, o))


def test_approx_crps_floor(rng):
    # approx-CRPS (mean pinball) must be >= 0.5 * median-pinball-at-0.5,
    # and >= 0; with sim=median=obs at tau=0.5 the 0.5 term is 0.
    o = _series(rng)
    levels = [0.1, 0.5, 0.9]
    qvals = np.vstack([o * 0.8, o, o * 1.2])  # lo, med=obs, hi
    crps = M.crps_from_quantiles(o, levels, qvals)
    assert crps >= 0.0
    # median term is 0 here (q=obs), so CRPS is driven by the 0.1/0.9 legs > 0
    assert crps > 0.0
    # perfect (all quantiles = obs) → 0
    assert M.crps_from_quantiles(o, levels, np.vstack([o, o, o])) == pytest.approx(0.0, abs=1e-9)


def test_tercile_masks_partition(rng):
    o = _series(rng)
    masks = M.tercile_masks(o)
    total = masks["low"].sum() + masks["mid"].sum() + masks["high"].sum()
    assert total == np.isfinite(o).sum()  # exact partition of finite obs
    assert masks["low"].sum() > 0 and masks["high"].sum() > 0


def test_aggregate_ignores_nan():
    per_station = {
        "a": {"nse": 0.8, "kge": 0.7},
        "b": {"nse": 0.6, "kge": float("nan")},
        "c": {"nse": float("nan"), "kge": 0.9},
    }
    agg = M.aggregate(per_station)
    assert agg["nse"]["scorable"] == 2
    assert agg["nse"]["median"] == pytest.approx(0.7)
    assert agg["kge"]["scorable"] == 2
    assert "frac_gt_0.5" in agg["nse"]
