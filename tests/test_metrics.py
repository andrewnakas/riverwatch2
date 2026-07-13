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


# ---------------------------- flood-event family (Nearing 2024 protocol)

def test_annual_maxima_drops_gappy_years(rng):
    # 3 full years (365 d) + 1 year with only 100 finite days.
    year = np.concatenate([np.full(365, y) for y in (2001, 2002, 2003)]
                          + [np.full(365, 2004)])
    vals = rng.uniform(1, 100, len(year))
    vals[year == 2002] = np.arange(365)          # known max = 364
    vals[year == 2004] = np.nan
    vals[np.flatnonzero(year == 2004)[:100]] = 5.0
    am = M.annual_maxima(vals, year)
    assert len(am) == 3                          # 2004 dropped
    assert am[1] == pytest.approx(364.0)


def test_return_period_thresholds_recover_gumbel(rng):
    # Gumbel(loc=100, scale=10) via inverse CDF; under F = exp(-1/T) the
    # T-year threshold is exactly loc + scale*ln(T).
    am = 100.0 - 10.0 * np.log(-np.log(rng.uniform(size=200)))
    thr = M.return_period_thresholds(am, years=(1.0, 2.0, 5.0, 10.0))
    for t in (1.0, 2.0, 5.0, 10.0):
        assert thr[t] == pytest.approx(100.0 + 10.0 * np.log(t), abs=5.0)
    assert thr[1.0] < thr[2.0] < thr[5.0] < thr[10.0]


def test_return_period_short_record_nan():
    thr = M.return_period_thresholds(np.arange(5.0), years=(2.0, 10.0))
    assert all(np.isnan(v) for v in thr.values())


def test_flood_event_scores_window_matching():
    n = 100
    obs = np.ones(n)
    obs[10:13] = 10.0   # one event starting day 10 (contiguous run = 1 event)
    obs[50] = 10.0      # second event, day 50
    sim = np.ones(n)
    sim[11] = 10.0      # 1 day off the first obs event
    sim[80] = 10.0      # false positive
    s = M.flood_event_scores(obs, sim, obs_thr=5.0, sim_thr=5.0, window_days=2)
    assert s["n_obs_events"] == 2 and s["n_sim_events"] == 2
    assert s["recall"] == pytest.approx(0.5)     # day-50 event missed
    assert s["precision"] == pytest.approx(0.5)  # day-80 event spurious
    assert s["f1"] == pytest.approx(0.5)
    # Same-day variant: the ±1-day hit no longer counts.
    s0 = M.flood_event_scores(obs, sim, obs_thr=5.0, sim_thr=5.0, window_days=0)
    assert s0["precision"] == 0.0 and s0["recall"] == 0.0 and s0["f1"] == 0.0


def test_flood_event_scores_degenerate_sides():
    n = 60
    obs = np.ones(n)
    obs[30] = 10.0
    flat = np.ones(n)
    s = M.flood_event_scores(obs, flat, obs_thr=5.0, sim_thr=5.0)
    assert s["recall"] == 0.0 and np.isnan(s["precision"]) and np.isnan(s["f1"])
    s = M.flood_event_scores(flat, obs, obs_thr=5.0, sim_thr=5.0)
    assert s["precision"] == 0.0 and np.isnan(s["recall"]) and np.isnan(s["f1"])
    s = M.flood_event_scores(obs, obs, obs_thr=5.0, sim_thr=float("nan"))
    assert np.isnan(s["f1"]) and s["n_sim_events"] == 0
