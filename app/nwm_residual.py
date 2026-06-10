"""v15.9: apply the learned NWM residual ensemble at forecast time.

Loads per-horizon LightGBM models trained by scripts/train_nwm_residual.py
against the v14.2 nwm-archive and adds a learned correction on top of
v14.1's multiplicative bias correction. Result is the `nwm_residual`
ensemble member.

v15.9 (post-BACKTEST_REPORT honest rebuild):
  - Three feature variants (v1/v2/v3) trained separately and averaged in
    log space. The 2026-06 temporal-holdout backtest
    (scripts/backtest_nwm_residual.py, benchmarks/nwm_backtest_v4.json)
    showed the average is the robust winner: it beat bias-corrected NWM
    at every horizon (median-station MAE ratio ~0.60 at h1 to ~0.62 at
    h14) where single variants traded wins by horizon.
  - This module is the single source of truth for feature definitions —
    the trainer imports FEAT_COLS_* and the helpers below, so the model
    can no longer be trained on inputs it never sees in serving (the
    v15.1 bug: trained with q_obs_t0=0, served with the real value).
  - v2/v3 features need recent observed flow (lags/trend/trailing mean,
    from q_hist) and two per-station trailing-skill stats shipped by the
    trainer in sidecar.json. Missing pieces degrade per-variant: a
    variant without its inputs is skipped, and if none can run the
    member falls back to the v14.1 bias-corrected value.

Layout under data/nwm_residual_models/:
  v1/h{N}.pkl, v2/h{N}.pkl, v3/h{N}.pkl   per-variant per-horizon models
  sidecar.json                            per-station trailing stats
  manifest.json                           honest holdout metrics
Legacy flat h{N}.pkl files (pre-v15.9) are still loaded as "v1".

Gate: RW2_ENABLE_NWM_RESIDUAL=1.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "nwm_residual_models"

# v1 = the original v15.1 feature set (kept for continuity; strongest at
# short horizons in the backtest).
FEAT_COLS_V1 = [
    "log1p_q_nwm_raw", "log1p_q_nwm_corrected", "log1p_q_obs_t0",
    "bias_scale", "doy", "month",
]
# v2 = + anchor gap, obs recency/trend, trailing local NWM skill.
FEAT_COLS_V2 = FEAT_COLS_V1 + [
    "d_anchor",            # log1p obs_t0 - log1p corrected at this horizon
    "log1p_q_obs_lag3",
    "log1p_q_obs_lag7",
    "obs_trend_3d",        # rising/falling limb
    "log1p_obs_trail30",   # station scale anchor
    "trail_h1_logmae",     # trailing |log error| of NWM h1 here (sidecar)
]
# v3 = + per-(station,horizon) trailing signed log residual (sidecar).
FEAT_COLS_V3 = FEAT_COLS_V2 + ["trail_resid_h"]

VARIANT_COLS = {"v1": FEAT_COLS_V1, "v2": FEAT_COLS_V2, "v3": FEAT_COLS_V3}

# Lazy globals
_models: dict[str, dict[int, dict]] | None = None
_sidecar: dict | None = None
_load_failed = False
_n_invocations = 0  # observability: how many stations actually used residual


def _is_enabled() -> bool:
    return os.environ.get("RW2_ENABLE_NWM_RESIDUAL") == "1"


def _load_dir(d: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in d.glob("h*.pkl"):
        try:
            with open(p, "rb") as f:
                bundle = pickle.load(f)
            out[int(p.stem[1:])] = bundle
        except Exception:
            continue
    return out


def _try_load() -> bool:
    global _models, _sidecar, _load_failed
    if _models is not None or _load_failed:
        return _models is not None
    if not MODELS_DIR.exists():
        _load_failed = True
        return False
    out: dict[str, dict[int, dict]] = {}
    for v in VARIANT_COLS:
        sub = MODELS_DIR / v
        if sub.is_dir():
            loaded = _load_dir(sub)
            if loaded:
                out[v] = loaded
    if not out:
        legacy = _load_dir(MODELS_DIR)  # pre-v15.9 flat layout
        if legacy:
            out["v1"] = legacy
    if not out:
        _load_failed = True
        return False
    _models = out
    try:
        _sidecar = json.loads((MODELS_DIR / "sidecar.json").read_text())
    except Exception:
        _sidecar = {}
    print(f"[nwm_residual] loaded variants: "
          f"{ {v: sorted(m.keys()) for v, m in out.items()} }")
    return True


def _obs_features(q_hist: Optional[pd.DataFrame]) -> Optional[dict]:
    """Issuance-time observed-flow features from the station's daily
    history (date, q_cfs). Returns None if there's not enough history."""
    if q_hist is None or len(q_hist) < 8:
        return None
    qh = q_hist.dropna(subset=["q_cfs"])
    if len(qh) < 8:
        return None
    by_date = {pd.Timestamp(d).date(): float(q)
               for d, q in zip(qh["date"], qh["q_cfs"])}
    last = max(by_date)
    t0 = by_date[last]

    def _at(days_back: int) -> Optional[float]:
        return by_date.get(last - pd.Timedelta(days=days_back))

    lag3, lag7 = _at(3), _at(7)
    trail = [by_date.get(last - pd.Timedelta(days=k)) for k in range(0, 30)]
    trail = [v for v in trail if v is not None]
    log_t0 = float(np.log1p(max(t0, 0.0)))
    log_lag3 = float(np.log1p(max(lag3, 0.0))) if lag3 is not None else log_t0
    return {
        "log1p_q_obs_lag3": log_lag3,
        "log1p_q_obs_lag7": float(np.log1p(max(lag7, 0.0))) if lag7 is not None else log_t0,
        "obs_trend_3d": log_t0 - log_lag3,
        "log1p_obs_trail30": float(np.log1p(max(np.mean(trail), 0.0))) if trail else log_t0,
    }


def apply_residual(
    nwm_raw: list[float],
    nwm_corrected: list[float],
    bias_scale: Optional[float],
    q_obs_today: Optional[float],
    issued_date: pd.Timestamp,
    q_hist: Optional[pd.DataFrame] = None,
    station_id: Optional[str] = None,
) -> Optional[list[float]]:
    """Return a per-horizon residual-corrected forecast, or None if the
    feature is gated off / no models loaded.

    nwm_raw and nwm_corrected must be the same length (= horizon).
    """
    global _n_invocations
    if not _is_enabled():
        return None
    if not _try_load():
        return None
    if not nwm_raw or len(nwm_raw) != len(nwm_corrected):
        return None
    # v15.9: the models are trained on the REAL observation at issuance.
    # Feeding 0 when it's missing (the old behavior) makes them predict a
    # collapse to near-zero flow. No observation → no residual member.
    if q_obs_today is None or not np.isfinite(q_obs_today):
        return None
    _n_invocations += 1
    bs = float(bias_scale) if bias_scale is not None and np.isfinite(bias_scale) else 1.0
    q_t0 = float(q_obs_today)
    issued = pd.Timestamp(issued_date)
    obs_feats = _obs_features(q_hist)
    side = (_sidecar or {}).get(str(station_id) or "", {})
    side_default = (_sidecar or {}).get("_default", {})
    trail_h1 = side.get("trail_h1_logmae", side_default.get("trail_h1_logmae", 0.0))
    trail_resid = side.get("trail_resid_h", {})

    out: list[float] = []
    for h, (qr, qc) in enumerate(zip(nwm_raw, nwm_corrected), start=1):
        target = issued + pd.Timedelta(days=h)
        log_qc = float(np.log1p(max(qc, 0.0)))
        feats = {
            "log1p_q_nwm_raw": float(np.log1p(max(qr, 0.0))),
            "log1p_q_nwm_corrected": log_qc,
            "log1p_q_obs_t0": float(np.log1p(q_t0)),
            "bias_scale": bs,
            "doy": target.dayofyear,
            "month": target.month,
            "d_anchor": float(np.log1p(q_t0)) - log_qc,
            "trail_h1_logmae": float(trail_h1),
            "trail_resid_h": float(trail_resid.get(str(h), 0.0)),
        }
        if obs_feats is not None:
            feats.update(obs_feats)

        deltas: list[float] = []
        for v, per_h in (_models or {}).items():
            bundle = per_h.get(h)
            if bundle is None:
                continue
            cols = bundle["feature_cols"]
            if any(c not in feats for c in cols):
                continue  # e.g. v2/v3 without q_hist
            x = np.array([[feats[c] for c in cols]], dtype=np.float32)
            try:
                deltas.append(float(bundle["model"].predict(x)[0]))
            except Exception:
                continue
        if not deltas:
            # No usable model for this horizon — pass through corrected value.
            out.append(float(max(qc, 0.0)))
            continue
        # Log-space ensemble mean == mean of deltas applied to log1p(qc).
        q_pred = float(np.expm1(log_qc + float(np.mean(deltas))))
        out.append(max(q_pred, 0.0))
    return out


def summary() -> dict:
    """Observability: which variants/horizons loaded + invocation count."""
    return {
        "variants": ({v: sorted(m.keys()) for v, m in _models.items()}
                     if _models else {}),
        "models_loaded": sum(len(m) for m in _models.values()) if _models else 0,
        "n_invocations": _n_invocations,
        "enabled": _is_enabled(),
    }
