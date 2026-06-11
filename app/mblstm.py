"""v16: Multi-basin LSTM (MB-LSTM) — homegrown Google-Flood-Hub-style member.

Architecture (per Nearing et al. 2024, Nature 626:1011, adapted):
  - Encoder LSTM over a 365-day hindcast window of weather forcings PLUS
    observed discharge (autoregressive input — the edge neither Google's
    gauge-free design nor NWM's nudging DA fully exploits).
  - Decoder LSTM over the forecast horizon driven by forecast weather,
    initialized from the encoder's final (h, c) state.
  - Quantile head (0.1 / 0.5 / 0.9) trained with pinball loss in per-basin
    normalized asinh space — the median is the point forecast, the outer
    quantiles populate q_lo / q_hi.

Normalization: discharge is asinh-transformed then standardized with
per-station (mu, sigma) computed from the station's own history at inference
time (training uses train-period stats), so the model is self-normalizing and
zero-shot at new gauges. Weather is standardized with global training stats
stored in the checkpoint.

Gated by RW2_ENABLE_MBLSTM=1; checkpoint at RW2_MBLSTM_CKPT_PATH (default
data/mblstm/model.pt). Returns None on any failure so the blend silently
drops the member, same contract as ealstm.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Encoder sees everything we have historically; decoder only what a weather
# forecast can actually supply.
ENC_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "rain_sum", "snowfall_sum",
    "shortwave_radiation_sum", "windspeed_10m_max",
    "et0_fao_evapotranspiration",
    "soil_moisture_0_to_10cm_mean", "soil_moisture_28_to_100cm_mean",
    "soil_temperature_0_to_7cm_mean", "snow_depth_max",
]
DEC_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "rain_sum", "snowfall_sum",
    "shortwave_radiation_sum", "windspeed_10m_max",
    "et0_fao_evapotranspiration",
]
# Static catchment descriptors: registry fields + GAGES-II basin attributes.
STATIC_FEATS = [
    "lat", "lon", "alt_ft", "log_drain_area",
    "FORESTNLCD06", "DEVNLCD06", "PPTAVG_BASIN", "SNOW_PCT_PRECIP",
    "SLOPE_PCT", "BFI_AVE", "AWCAVE", "PERMAVE",
    "ELEV_MEAN_M_BASIN", "RUNAVE7100",
]
CONTEXT_DAYS = 365
QUANTILES = (0.1, 0.5, 0.9)

_model = None
_cfg: dict | None = None
_load_failed = False


def _is_enabled() -> bool:
    return os.environ.get("RW2_ENABLE_MBLSTM") == "1"


def _ckpt_path() -> Path:
    p = os.environ.get("RW2_MBLSTM_CKPT_PATH")
    if p:
        return Path(p)
    return Path(__file__).resolve().parents[1] / "data" / "mblstm" / "model.pt"


def build_model(cfg: dict):
    """Construct the torch module from a checkpoint cfg dict. Lives here so
    training and serving can never drift apart on architecture."""
    import torch.nn as nn

    enc_in = len(cfg["enc_vars"]) + 2 + 2 + len(cfg["static_feats"])  # +q,+qmask,+doy
    dec_in = len(cfg["dec_vars"]) + 2 + 1 + len(cfg["static_feats"])  # +doy,+lead
    hidden = int(cfg["hidden"])
    nq = len(cfg["quantiles"])

    class MBLSTMNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(enc_in, hidden, batch_first=True)
            self.decoder = nn.LSTM(dec_in, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, nq),
            )

        def forward(self, x_enc, x_dec):
            _, hc = self.encoder(x_enc)
            out, _ = self.decoder(x_dec, hc)
            return self.head(out)  # (B, H, nq)

    return MBLSTMNet()


def static_vector(attrs: dict, cfg: dict) -> np.ndarray:
    """Standardized static-feature vector with median imputation."""
    med = cfg["static_median"]
    mu, sd = cfg["static_mean"], cfg["static_std"]
    out = []
    for i, name in enumerate(cfg["static_feats"]):
        if name == "log_drain_area":
            da = attrs.get("drain_area_sqmi")
            v = math.log1p(float(da)) if da is not None and np.isfinite(da) and da > 0 else None
        else:
            v = attrs.get(name)
            v = float(v) if v is not None and np.isfinite(v) else None
        if v is None:
            v = med[i]
        s = sd[i] if sd[i] > 1e-9 else 1.0
        out.append((v - mu[i]) / s)
    return np.asarray(out, dtype=np.float32)


def _doy_sincos(dates: pd.Series) -> np.ndarray:
    doy = pd.to_datetime(dates).dt.dayofyear.to_numpy(dtype=np.float32)
    ang = 2.0 * np.pi * doy / 366.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


def norm_wx(df: pd.DataFrame, cols: list[str], cfg: dict) -> np.ndarray:
    """Standardize weather columns with global training stats; NaN → 0
    (i.e. the training mean)."""
    mu = np.asarray([cfg["wx_mean"][c] for c in cols], dtype=np.float32)
    sd = np.asarray([max(cfg["wx_std"][c], 1e-6) for c in cols], dtype=np.float32)
    arr = df.reindex(columns=cols).to_numpy(dtype=np.float32)
    arr = (arr - mu) / sd
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def q_norm_stats(q_cfs: np.ndarray) -> Optional[tuple[float, float]]:
    """Per-station asinh-discharge normalization stats."""
    v = np.asinh(np.clip(q_cfs[np.isfinite(q_cfs)], 0.0, None))
    if len(v) < 180:
        return None
    sd = float(np.std(v))
    if sd < 1e-6:
        return None
    return float(np.mean(v)), sd


def _try_load() -> bool:
    global _model, _cfg, _load_failed
    if _model is not None or _load_failed:
        return _model is not None
    ckpt = _ckpt_path()
    if not ckpt.exists():
        _load_failed = True
        return False
    try:
        import torch
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = payload["cfg"]
        model = build_model(cfg)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        _model, _cfg = model, cfg
        return True
    except Exception:
        _load_failed = True
        return False


def forecast(
    q_hist: pd.DataFrame,
    wx_hist: pd.DataFrame,
    wx_fcst: pd.DataFrame,
    static_attrs: dict,
    horizon: int,
) -> Optional[list[dict]]:
    """Standard member entry point — same contract as ealstm.forecast()."""
    if not _is_enabled() or not _try_load():
        return None
    cfg = _cfg or {}
    try:
        if q_hist is None or len(q_hist) < 365 or wx_hist is None or len(wx_hist) < 30:
            return None
        q = q_hist.copy()
        q["date"] = pd.to_datetime(q["date"])
        stats = q_norm_stats(q["q_cfs"].to_numpy(dtype=np.float64))
        if stats is None:
            return None
        mu_q, sd_q = stats

        wx = wx_hist.copy()
        wx["date"] = pd.to_datetime(wx["date"])
        last_date = q["date"].iloc[-1]
        # Daily-continuous 365-day window ending at the last observation.
        idx = pd.date_range(last_date - pd.Timedelta(days=CONTEXT_DAYS - 1), last_date, freq="D")
        wx_win = wx.set_index("date").reindex(idx)
        q_win = q.set_index("date")["q_cfs"].reindex(idx).to_numpy(dtype=np.float64)

        q_asinh = np.asinh(np.clip(q_win, 0.0, None))
        q_mask = np.isfinite(q_asinh).astype(np.float32)
        q_n = np.nan_to_num((q_asinh - mu_q) / sd_q, nan=0.0).astype(np.float32)

        sv = static_vector(static_attrs or {}, cfg)
        enc_wx = norm_wx(wx_win.reset_index(drop=True), cfg["enc_vars"], cfg)
        enc_doy = _doy_sincos(pd.Series(idx))
        T = len(idx)
        x_enc = np.concatenate(
            [enc_wx, q_n[:, None], q_mask[:, None], enc_doy,
             np.repeat(sv[None, :], T, axis=0)], axis=1)

        fut_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        wf = wx_fcst.copy()
        if len(wf):
            wf["date"] = pd.to_datetime(wf["date"])
            wf = wf.set_index("date").reindex(fut_idx)
        else:
            wf = pd.DataFrame(index=fut_idx)
        dec_wx = norm_wx(wf.reset_index(drop=True), cfg["dec_vars"], cfg)
        dec_doy = _doy_sincos(pd.Series(fut_idx))
        lead = (np.arange(1, horizon + 1, dtype=np.float32) / float(cfg["horizon"]))[:, None]
        x_dec = np.concatenate(
            [dec_wx, dec_doy, lead, np.repeat(sv[None, :], horizon, axis=0)], axis=1)

        import torch
        with torch.no_grad():
            yq = _model(
                torch.from_numpy(x_enc[None, :, :]),
                torch.from_numpy(x_dec[None, :, :]),
            ).squeeze(0).numpy()  # (horizon, nq)

        # Denormalize: normalized asinh → cfs. Enforce quantile ordering.
        yq = np.sort(yq, axis=1)
        q_cfs = np.sinh(yq * sd_q + mu_q)
        q_cfs = np.clip(q_cfs, 0.0, None)
        if not np.all(np.isfinite(q_cfs)):
            return None

        lo_i, med_i, hi_i = 0, len(cfg["quantiles"]) // 2, len(cfg["quantiles"]) - 1
        rows = []
        for i in range(horizon):
            d = (last_date + pd.Timedelta(days=i + 1)).date()
            rows.append({
                "date": d.isoformat(),
                "q_cfs": float(q_cfs[i, med_i]),
                "q_lo": float(q_cfs[i, lo_i]),
                "q_hi": float(q_cfs[i, hi_i]),
            })
        return rows
    except Exception:
        return None
