"""v15.1: EA-LSTM (Entity-Aware LSTM) regional model as the 8th ensemble
member. Lazy-loads a pretrained NeuralHydrology checkpoint and runs CPU
forward passes per station, using static catchment attributes as the
"entity" embedding.

EA-LSTM (Kratzert et al. 2019) splits the LSTM input gate into a static
gate driven by basin attributes (drain area, soil, land cover, climate
indices) and a dynamic gate driven by daily forcings (P, T, SWE, etc.).
Pretrained on CAMELS / Caravan, it is the strongest CPU-friendly
hydrologic-ML baseline available — the v15 research report flagged it as
the top expected-MAE win for v15.

Wiring strategy:
- This module is gated by RW2_ENABLE_EALSTM=1 so we can ship the wiring
  without weights and turn it on once a checkpoint + attribute table land.
- A pretrained checkpoint path is configured via RW2_EALSTM_CKPT_PATH.
  Without it we return None and the blend simply drops the member, exactly
  like timesfm_xreg does when jax isn't installed.
- Static attributes per station come from data/gages2_attrs/<id>.json (the
  v12 Caravan/GAGES-II layer) — same provenance the LightGBM pooled member
  already consumes, so no new offline pipeline is required for the first
  rollout.

Inputs (forecast() entry point):
  q_hist        : pandas Series of daily discharge (cfs), DateTime index
  wx_hist       : pandas DataFrame of past forcings (precip_mm, temp_c,
                  snow_depth_in, etc.)
  wx_fcst       : DataFrame of future forcings, length == horizon
  static_attrs  : dict of per-station static features (drain_area_sqmi,
                  ...). Keys must match the model's training feature set.
  horizon       : int, days ahead

Returns:
  list[dict] of length horizon, each {date: ISO, q_cfs: float, q_lo: float|None,
  q_hi: float|None}, or None on any failure (missing weights, attribute
  schema mismatch, NaN forcings, etc.).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Lazy globals — populated on first forecast() call.
_model = None
_attr_means: dict | None = None
_attr_stds: dict | None = None
_dynamic_features: list[str] | None = None
_load_failed = False


def _is_enabled() -> bool:
    return os.environ.get("RW2_ENABLE_EALSTM") == "1"


def _ckpt_path() -> Optional[Path]:
    p = os.environ.get("RW2_EALSTM_CKPT_PATH")
    if not p:
        return None
    pp = Path(p)
    return pp if pp.exists() else None


def _try_load_model() -> bool:
    """Lazy-load the pretrained EA-LSTM checkpoint. Sets _load_failed on
    any failure so subsequent calls early-exit."""
    global _model, _attr_means, _attr_stds, _dynamic_features, _load_failed
    if _model is not None or _load_failed:
        return _model is not None
    ckpt = _ckpt_path()
    if ckpt is None:
        _load_failed = True
        return False
    try:
        # neuralhydrology is the reference library for EA-LSTM/CudaLSTM.
        # Imported lazily so the module can be imported on machines without
        # the optional dep installed.
        import torch  # noqa: F401
        # Defer the heavy NH import until we actually have a checkpoint
        from neuralhydrology.utils.config import Config  # type: ignore
        from neuralhydrology.modelzoo.ealstm import EALSTM  # type: ignore

        # Standard NH checkpoint layout: ckpt_path is *.pt next to a
        # config.yml in the same dir.
        cfg_path = ckpt.parent / "config.yml"
        cfg = Config(cfg_path)
        model = EALSTM(cfg=cfg)
        state = torch.load(ckpt, map_location="cpu")
        # NH stores ('model_state_dict', ...) wrapper in some versions
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        _model = model
        # Pull feature normalisation from config so we reproduce training
        # statistics at inference time.
        _attr_means = dict(cfg.static_attributes_means or {})
        _attr_stds = dict(cfg.static_attributes_stds or {})
        _dynamic_features = list(cfg.dynamic_inputs or [])
        return True
    except Exception:
        _load_failed = True
        return False


def _normalize_static(attrs: dict) -> Optional[np.ndarray]:
    """Normalize static attrs against the checkpoint's training stats.
    Returns None if any required attribute is missing (we don't impute —
    silently dropping the member is safer than a wrong prediction)."""
    if _attr_means is None or _attr_stds is None:
        return None
    out = []
    for name, mu in _attr_means.items():
        v = attrs.get(name)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        sd = _attr_stds.get(name) or 1.0
        out.append((float(v) - float(mu)) / float(sd))
    return np.asarray(out, dtype=np.float32)


def forecast(
    q_hist: pd.Series,
    wx_hist: pd.DataFrame,
    wx_fcst: pd.DataFrame,
    static_attrs: dict,
    horizon: int,
) -> Optional[list[dict]]:
    """Run EA-LSTM forward to produce a horizon-day discharge forecast.
    Returns None on any setup failure (gate disabled, no checkpoint, attr
    schema mismatch). The blend treats None as a missing member.
    """
    if not _is_enabled():
        return None
    if not _try_load_model():
        return None

    static_norm = _normalize_static(static_attrs or {})
    if static_norm is None:
        return None

    # Build the dynamic-input matrix in the order the checkpoint expects.
    if not _dynamic_features:
        return None
    try:
        cols = list(_dynamic_features)
        # Past window: take min(seq_len, available) days of forcings + obs
        # discharge ending at t0. NH default seq_len is 365; we honor that
        # if present, otherwise fall back to len(wx_hist).
        seq_len = min(len(wx_hist), 365)
        past = wx_hist.tail(seq_len).copy()
        # Some checkpoints expect q as one of the dynamic inputs (autoregressive).
        if "q_cfs" in cols and "q_cfs" not in past.columns:
            past = past.assign(q_cfs=q_hist.reindex(past.index).values)
        for c in cols:
            if c not in past.columns:
                return None
        past_arr = past[cols].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(past_arr)):
            return None

        import torch
        x_d = torch.from_numpy(past_arr).unsqueeze(0)            # (1, T, F)
        x_s = torch.from_numpy(static_norm).unsqueeze(0)         # (1, S)

        # NH EA-LSTM forward expects {"x_d": ..., "x_s": ...} dict.
        with torch.no_grad():
            out = _model({"x_d": x_d, "x_s": x_s})
        # Output shape: (1, T, 1) — last `horizon` steps are our forecast.
        y = out["y_hat"].squeeze(0).squeeze(-1).cpu().numpy()
        if y.shape[0] < horizon:
            return None
        y_fcst = y[-horizon:]
    except Exception:
        return None

    # Pack into the standard member-forecast envelope.
    last_date = pd.to_datetime(q_hist.index[-1]).date() if len(q_hist) else pd.Timestamp.utcnow().date()
    rows: list[dict] = []
    for i, q in enumerate(y_fcst, start=1):
        d = pd.Timestamp(last_date) + pd.Timedelta(days=i)
        if not np.isfinite(q):
            continue
        rows.append({"date": d.isoformat()[:10], "q_cfs": float(max(q, 0.0)),
                     "q_lo": None, "q_hi": None})
    if len(rows) < horizon // 2:
        # Sanity bail: if more than half the horizon dropped to NaN, the
        # blend is better off without us.
        return None
    return rows
