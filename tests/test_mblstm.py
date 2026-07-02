"""v16 multi-basin LSTM member: gate, checkpoint loading, forward contract.

These tests build a tiny torch checkpoint, so they need torch. The Pages CI
runner is torch-free (the production serving path imports torch lazily and the
member silently no-ops without it), so skip the whole module when torch is
absent rather than fail the deploy gate. Run locally with torch for coverage.
"""
import importlib
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="MB-LSTM tests need torch; CI runner is torch-free")

from app import mblstm
from app.weather import DAILY_VARS


def _fresh(monkeypatch, enabled, ckpt_path=None):
    importlib.reload(mblstm)
    if enabled:
        monkeypatch.setenv("RW2_ENABLE_MBLSTM", "1")
    else:
        monkeypatch.delenv("RW2_ENABLE_MBLSTM", raising=False)
    if ckpt_path is not None:
        monkeypatch.setenv("RW2_MBLSTM_CKPT_PATH", str(ckpt_path))
    return mblstm


def _synthetic_inputs(days=900, horizon=14):
    d0 = date(2023, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(days)]
    t = np.arange(days)
    q = 200 + 150 * np.sin(2 * np.pi * t / 365.25) + np.random.default_rng(0).normal(0, 10, days)
    q_hist = pd.DataFrame({"date": dates, "q_cfs": np.clip(q, 1, None)})
    wx_hist = pd.DataFrame({"date": dates})
    for c in DAILY_VARS:
        wx_hist[c] = np.random.default_rng(1).random(days)
    fdates = [dates[-1] + timedelta(days=i + 1) for i in range(horizon)]
    wx_fcst = pd.DataFrame({"date": fdates})
    for c in DAILY_VARS:
        wx_fcst[c] = np.random.default_rng(2).random(horizon)
    attrs = {"lat": 40.0, "lon": -105.0, "alt_ft": 5000.0, "drain_area_sqmi": 350.0,
             "FORESTNLCD06": 60.0, "BFI_AVE": 50.0}
    return q_hist, wx_hist, wx_fcst, attrs


def _tiny_ckpt(tmp_path):
    import torch
    cfg = {
        "enc_vars": mblstm.ENC_VARS, "dec_vars": mblstm.DEC_VARS,
        "static_feats": mblstm.STATIC_FEATS, "quantiles": list(mblstm.QUANTILES),
        "hidden": 16, "horizon": 14, "context": mblstm.CONTEXT_DAYS,
        "wx_mean": {c: 0.5 for c in mblstm.ENC_VARS},
        "wx_std": {c: 1.0 for c in mblstm.ENC_VARS},
        "static_median": [0.0] * len(mblstm.STATIC_FEATS),
        "static_mean": [0.0] * len(mblstm.STATIC_FEATS),
        "static_std": [1.0] * len(mblstm.STATIC_FEATS),
    }
    model = mblstm.build_model(cfg)
    p = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, p)
    return p


def test_gate_off_returns_none(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, enabled=False, ckpt_path=_tiny_ckpt(tmp_path))
    q_hist, wx_hist, wx_fcst, attrs = _synthetic_inputs()
    assert mod.forecast(q_hist, wx_hist, wx_fcst, attrs, 14) is None


def test_missing_ckpt_returns_none(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, enabled=True, ckpt_path=tmp_path / "nope.pt")
    q_hist, wx_hist, wx_fcst, attrs = _synthetic_inputs()
    assert mod.forecast(q_hist, wx_hist, wx_fcst, attrs, 14) is None


def test_forecast_contract(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, enabled=True, ckpt_path=_tiny_ckpt(tmp_path))
    q_hist, wx_hist, wx_fcst, attrs = _synthetic_inputs()
    rows = mod.forecast(q_hist, wx_hist, wx_fcst, attrs, 14)
    assert rows is not None and len(rows) == 14
    last = pd.Timestamp(q_hist["date"].iloc[-1])
    for i, r in enumerate(rows, 1):
        assert r["date"] == (last + pd.Timedelta(days=i)).date().isoformat()
        assert math.isfinite(r["q_cfs"]) and r["q_cfs"] >= 0.0
        assert r["q_lo"] <= r["q_cfs"] <= r["q_hi"]


def test_short_history_returns_none(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, enabled=True, ckpt_path=_tiny_ckpt(tmp_path))
    q_hist, wx_hist, wx_fcst, attrs = _synthetic_inputs(days=120)
    assert mod.forecast(q_hist, wx_hist, wx_fcst, attrs, 14) is None


def test_static_vector_imputes_missing(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, enabled=True, ckpt_path=_tiny_ckpt(tmp_path))
    assert mod._try_load()
    sv = mod.static_vector({}, mod._cfg)
    assert sv.shape == (len(mod.STATIC_FEATS),)
    assert np.all(np.isfinite(sv))


def test_forecast_emits_true_median(monkeypatch, tmp_path):
    """q_med must be present, finite, and inside the [q_lo, q_hi] band —
    it feeds the 0.5 slot of quantile-based CRPS regardless of point policy."""
    mod = _fresh(monkeypatch, enabled=True, ckpt_path=_tiny_ckpt(tmp_path))
    q_hist, wx_hist, wx_fcst, attrs = _synthetic_inputs()
    rows = mod.forecast(q_hist, wx_hist, wx_fcst, attrs, 14)
    assert rows is not None
    for r in rows:
        assert math.isfinite(r["q_med"]) and r["q_med"] >= 0.0
        assert r["q_lo"] <= r["q_med"] <= r["q_hi"]


def test_point_policy_blend_leans_high(monkeypatch, tmp_path):
    """blend point policy must sit between the median and q_hi."""
    ckpt = _tiny_ckpt(tmp_path)
    mod = _fresh(monkeypatch, enabled=True, ckpt_path=ckpt)
    q_hist, wx_hist, wx_fcst, attrs = _synthetic_inputs()
    med_rows = mod.forecast(q_hist, wx_hist, wx_fcst, attrs, 14)
    monkeypatch.setenv("RW2_MBLSTM_POINT", "blend0.3")
    mod2 = _fresh(monkeypatch, enabled=True, ckpt_path=ckpt)
    blend_rows = mod2.forecast(q_hist, wx_hist, wx_fcst, attrs, 14)
    for m, b in zip(med_rows, blend_rows):
        assert b["q_cfs"] >= m["q_med"] - 1e-9
        assert b["q_cfs"] <= b["q_hi"] + 1e-9
        # the raw quantiles themselves must not move with the policy
        assert abs(m["q_med"] - b["q_med"]) < 1e-9


def _load_backtest_module():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "bt", Path(__file__).resolve().parents[1] / "scripts" / "backtest_mblstm.py")
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    return bt


def test_member_quantile_pooling_values():
    """Pooled band = empirical 10/50/90 of the 3xM member quantile values;
    point = median of member medians; identical members pool to themselves."""
    bt = _load_backtest_module()
    # Horizon 1: three disagreeing members; horizon 2: all identical.
    lo = np.array([[10.0, 100.0], [40.0, 100.0], [70.0, 100.0]])
    med = np.array([[20.0, 200.0], [50.0, 200.0], [80.0, 200.0]])
    hi = np.array([[30.0, 300.0], [60.0, 300.0], [90.0, 300.0]])
    plo, pmed, phi, point = bt.pool_member_quantiles(lo, med, hi)
    pooled_h1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
    assert plo[0] == pytest.approx(np.percentile(pooled_h1, 10))
    assert pmed[0] == pytest.approx(np.percentile(pooled_h1, 50))
    assert phi[0] == pytest.approx(np.percentile(pooled_h1, 90))
    assert point[0] == pytest.approx(50.0)  # median of {20, 50, 80}
    # Disagreeing members widen the band beyond any single member's (20 cfs).
    assert phi[0] - plo[0] > 20.0
    # Degenerate ensemble: pooled triplet == the common member triplet.
    assert plo[1] == pytest.approx(100.0)
    assert pmed[1] == pytest.approx(200.0)
    assert phi[1] == pytest.approx(300.0)
    assert point[1] == pytest.approx(200.0)


def test_member_quantile_pooling_ordered():
    """Pooled quantiles are ordered lo <= med <= hi at every horizon for
    arbitrary (ordered) member triplets."""
    bt = _load_backtest_module()
    rng = np.random.default_rng(7)
    lo = rng.random((8, 14)) * 100.0
    med = lo + rng.random((8, 14)) * 100.0
    hi = med + rng.random((8, 14)) * 100.0
    plo, pmed, phi, point = bt.pool_member_quantiles(lo, med, hi)
    assert plo.shape == pmed.shape == phi.shape == point.shape == (14,)
    assert np.all(np.isfinite(plo)) and np.all(np.isfinite(point))
    assert np.all(plo <= pmed) and np.all(pmed <= phi)
    # Point stays inside the pooled band (median of medians vs mixture band).
    assert np.all(point >= plo - 1e-9) and np.all(point <= phi + 1e-9)


def test_backtest_anchor_formula():
    """Anchor correction: full offset at h=1, linear decay, zero past 1+decay."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "bt", Path(__file__).resolve().parents[1] / "scripts" / "backtest_mblstm.py")
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    y = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    out = bt.anchor(y, y_h1=100.0, q_obs_t0=160.0, decay_h=2.0)
    assert out[0] == pytest.approx(160.0)        # h1 pinned to obs
    assert out[1] == pytest.approx(110.0 + 30.0)  # half offset
    assert out[2] == pytest.approx(120.0)         # decayed to zero
    assert np.allclose(out[3:], y[3:])
    assert np.allclose(bt.anchor(y, 100.0, 160.0, 0.0), y)  # decay 0 = off
