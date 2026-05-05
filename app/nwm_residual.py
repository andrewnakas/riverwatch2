"""v15.1: apply the learned NWM residual model at forecast time.

Loads per-horizon LightGBM models from data/nwm_residual_models/h{N}.pkl
(trained by scripts/train_nwm_residual.py against the v14.2 nwm-archive)
and adds a learned correction on top of v14.1's multiplicative bias
correction. Result is a 9th ensemble member: `nwm_residual`.

Strategy:
  - The blend already has `nwm` (v14.1 bias-corrected raw forecast). Rather
    than replace it, we add a separate member so the LightGBM stacker
    can learn how much to trust the residual correction per station.
  - The residual is predicted in log1p-space: at inference we add the
    predicted residual to log1p(q_nwm_corrected), then expm1 back.
  - Gracefully degrades: if no model is found for a horizon, that horizon
    falls back to the v14.1 bias-corrected value (i.e. zero learned
    residual). If no models exist at all, returns None and the blend
    drops the member.

Gate: RW2_ENABLE_NWM_RESIDUAL=1.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "nwm_residual_models"

# Lazy globals
_models: dict[int, dict] | None = None
_load_failed = False


def _is_enabled() -> bool:
    return os.environ.get("RW2_ENABLE_NWM_RESIDUAL") == "1"


def _try_load() -> bool:
    global _models, _load_failed
    if _models is not None or _load_failed:
        return _models is not None
    if not MODELS_DIR.exists():
        _load_failed = True
        return False
    out: dict[int, dict] = {}
    for p in MODELS_DIR.glob("h*.pkl"):
        try:
            with open(p, "rb") as f:
                bundle = pickle.load(f)
            h = int(p.stem[1:])
            out[h] = bundle
        except Exception:
            continue
    if not out:
        _load_failed = True
        return False
    _models = out
    return True


def apply_residual(
    nwm_raw: list[float],
    nwm_corrected: list[float],
    bias_scale: Optional[float],
    q_obs_today: Optional[float],
    issued_date: pd.Timestamp,
) -> Optional[list[float]]:
    """Return a per-horizon residual-corrected forecast, or None if the
    feature is gated off / no models loaded.

    nwm_raw and nwm_corrected must be the same length (= horizon).
    """
    if not _is_enabled():
        return None
    if not _try_load():
        return None
    if not nwm_raw or len(nwm_raw) != len(nwm_corrected):
        return None
    bs = float(bias_scale) if bias_scale is not None and np.isfinite(bias_scale) else 1.0
    q_t0 = float(q_obs_today) if q_obs_today is not None and np.isfinite(q_obs_today) else 0.0
    issued = pd.Timestamp(issued_date)
    out: list[float] = []
    for h, (qr, qc) in enumerate(zip(nwm_raw, nwm_corrected), start=1):
        bundle = _models.get(h) if _models else None
        if bundle is None:
            # No learned model for this horizon — pass through corrected value.
            out.append(float(max(qc, 0.0)))
            continue
        target = issued + pd.Timedelta(days=h)
        feats = {
            "log1p_q_nwm_raw": np.log1p(max(qr, 0.0)),
            "log1p_q_nwm_corrected": np.log1p(max(qc, 0.0)),
            "log1p_q_obs_t0": np.log1p(q_t0),
            "bias_scale": bs,
            "doy": target.dayofyear,
            "month": target.month,
        }
        cols = bundle["feature_cols"]
        x = np.array([[feats[c] for c in cols]], dtype=np.float32)
        try:
            delta = float(bundle["model"].predict(x)[0])
        except Exception:
            out.append(float(max(qc, 0.0)))
            continue
        q_pred = float(np.expm1(np.log1p(max(qc, 0.0)) + delta))
        out.append(max(q_pred, 0.0))
    return out
