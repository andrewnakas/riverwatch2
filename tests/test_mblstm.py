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
