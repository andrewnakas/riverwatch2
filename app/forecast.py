"""Live river-discharge forecasting engine.

Three forecasters are wired together and blended:
  1. persistence_lag1            — yhat[t+1] = q[t]
  2. runoff_ridge                — Ridge on lagged-discharge + DOY + recent precip/temp
  3. chronos_bolt (optional)     — Amazon Chronos-Bolt zero-shot foundation model

The ensemble is a per-station weighted blend selected on rolling validation.
Each forecast call runs the models on demand against fresh USGS + Open-Meteo data.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from . import usgs, weather

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HORIZON_DAYS = 14
TRAIN_LOOKBACK_DAYS = 1095
LAGS = [1, 2, 3, 5, 7, 14, 30, 60]
PRECIP_WINDOWS = [1, 3, 7, 14, 30]
TEMP_WINDOWS = [3, 7, 14]


@dataclass
class StationForecast:
    station_id: str
    issued_at: str
    history: List[dict]
    members: Dict[str, List[dict]]
    blend: List[dict]
    weights: Dict[str, float]
    rolling_mae: Dict[str, float]
    chosen: str
    notes: List[str] = field(default_factory=list)


def _build_features(q_hist: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    df = q_hist.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").asfreq("D")
    df["q_cfs"] = df["q_cfs"].interpolate(limit=3)

    wx2 = wx.copy()
    wx2["date"] = pd.to_datetime(wx2["date"])
    wx2 = wx2.set_index("date").asfreq("D")
    df = df.join(wx2, how="left")

    for lag in LAGS:
        df[f"q_lag_{lag}"] = df["q_cfs"].shift(lag)
    df["q_log"] = np.log1p(df["q_cfs"].clip(lower=0))
    for lag in LAGS:
        df[f"qlog_lag_{lag}"] = df["q_log"].shift(lag)

    if "precipitation_sum" in df.columns:
        for w in PRECIP_WINDOWS:
            df[f"precip_{w}d"] = df["precipitation_sum"].rolling(w, min_periods=1).sum()
    if "temperature_2m_mean" in df.columns:
        for w in TEMP_WINDOWS:
            df[f"tmean_{w}d"] = df["temperature_2m_mean"].rolling(w, min_periods=1).mean()
        df["pos_dd_7d"] = df["temperature_2m_mean"].clip(lower=0).rolling(7, min_periods=1).sum()
    if "snowfall_sum" in df.columns:
        df["snow_7d"] = df["snowfall_sum"].rolling(7, min_periods=1).sum()
        df["snow_30d"] = df["snowfall_sum"].rolling(30, min_periods=1).sum()

    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def _feature_columns(df: pd.DataFrame) -> List[str]:
    candidates = [c for c in df.columns if c not in {"q_cfs", "q_log"}]
    return [c for c in candidates if df[c].notna().sum() > 10]


def persistence_forecast(q_hist: pd.DataFrame, horizon: int) -> List[float]:
    if q_hist.empty:
        return [float("nan")] * horizon
    last = float(q_hist["q_cfs"].iloc[-1])
    return [last] * horizon


def runoff_ridge_forecast(
    q_hist: pd.DataFrame,
    wx_hist: pd.DataFrame,
    wx_fcst: pd.DataFrame,
    horizon: int,
) -> tuple[List[float], Dict[str, float]]:
    """Direct multi-step ridge: one model per horizon day, no recursion.

    Each model predicts log-discharge at t+h using only features known at t,
    so there's no compounding inference error across the horizon.
    Returns (predictions_cfs, rolling_validation_metrics).
    """
    if q_hist.empty:
        return [float("nan")] * horizon, {}

    wx_combined = pd.concat([wx_hist, wx_fcst], ignore_index=True)
    wx_combined = wx_combined.drop_duplicates(subset="date", keep="last").sort_values("date")

    feats = _build_features(q_hist, wx_combined)
    cols = _feature_columns(feats)
    if not cols:
        return persistence_forecast(q_hist, horizon), {}

    last_date = pd.to_datetime(q_hist["date"].iloc[-1])
    last_q = float(q_hist["q_cfs"].iloc[-1])

    # Build training matrix once per horizon h: target = q_log shifted by -h
    # (i.e. value h days into the future). Features come from the same row.
    preds: List[float] = []
    horizon_maes: List[float] = []

    # The "now" row — features known as of last_date.
    feats_now = feats.loc[last_date, cols] if last_date in feats.index else feats[cols].dropna().iloc[-1]
    feats_now = feats_now.fillna(0.0)

    for h in range(1, horizon + 1):
        df_h = feats.copy()
        df_h["target_log"] = df_h["q_log"].shift(-h)
        train = df_h.dropna(subset=cols + ["target_log"])
        if len(train) < 30:
            preds.append(last_q)
            continue
        Xtr = train[cols].values
        ytr = train["target_log"].values
        sc = StandardScaler().fit(Xtr)
        model = Ridge(alpha=1.0, random_state=0).fit(sc.transform(Xtr), ytr)
        x = sc.transform(feats_now.values.reshape(1, -1))
        yhat_log = float(model.predict(x)[0])
        yhat = float(np.expm1(yhat_log))
        if not math.isfinite(yhat) or yhat < 0:
            yhat = last_q
        preds.append(yhat)

        # Rolling MAE at this horizon: hold out last 30 days
        if len(train) > 90:
            cut = len(train) - 30
            sc_v = StandardScaler().fit(train[cols].iloc[:cut].values)
            m_v = Ridge(alpha=1.0, random_state=0).fit(sc_v.transform(train[cols].iloc[:cut].values), train["target_log"].iloc[:cut].values)
            yhat_v = np.expm1(m_v.predict(sc_v.transform(train[cols].iloc[cut:].values)))
            yhat_v = np.clip(yhat_v, 0, None)
            ytrue_v = np.expm1(train["target_log"].iloc[cut:].values)
            horizon_maes.append(float(mean_absolute_error(ytrue_v, yhat_v)))

    rolling_mae = {"mae_mean": float(np.mean(horizon_maes))} if horizon_maes else {}
    return preds, rolling_mae


# ---------------------------------------------------------------------------
# Chronos-Bolt zero-shot
# ---------------------------------------------------------------------------

_chronos_pipeline = None
_chronos_failed = False


def _get_chronos():
    """Lazy-load Chronos-Bolt. Returns None if not installed."""
    global _chronos_pipeline, _chronos_failed
    if _chronos_pipeline is not None or _chronos_failed:
        return _chronos_pipeline
    try:
        from chronos import BaseChronosPipeline  # type: ignore
        import torch
        _chronos_pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    except Exception as exc:
        _chronos_failed = True
        print(f"[chronos] disabled: {exc}")
        _chronos_pipeline = None
    return _chronos_pipeline


def _seasonal_scale(q_hist: pd.DataFrame, horizon: int) -> Optional[np.ndarray]:
    """Per-DOY climatological ratio: q_climatology(t+h) / q_climatology(t).

    Uses a ±7-day window around each DOY to get a smooth climatology. Returns
    None if history is too short to estimate one full year.
    """
    if len(q_hist) < 540:
        return None
    df = q_hist.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["q_log"] = np.log1p(df["q_cfs"].clip(lower=0))
    clim = df.groupby("doy")["q_log"].mean()
    smoothed = pd.Series(
        [clim.reindex(range(d - 7, d + 8)).dropna().mean() for d in range(1, 367)],
        index=range(1, 367),
    )
    if smoothed.isna().all():
        return None
    last_date = df["date"].iloc[-1]
    base_doy = int(last_date.dayofyear)
    base = smoothed.get(base_doy, smoothed.dropna().mean())
    out = []
    for h in range(1, horizon + 1):
        doy_h = ((base_doy - 1 + h) % 366) + 1
        v = smoothed.get(doy_h, base)
        if not np.isfinite(v) or not np.isfinite(base):
            out.append(1.0)
        else:
            out.append(float(np.exp(v - base)))
    return np.array(out)


def chronos_forecast(q_hist: pd.DataFrame, horizon: int) -> Optional[List[float]]:
    pipe = _get_chronos()
    if pipe is None or q_hist.empty:
        return None
    try:
        import torch
        ctx = torch.tensor(q_hist["q_cfs"].astype(float).tolist())
        quantiles, _mean = pipe.predict_quantiles(
            inputs=ctx,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        median = np.array(quantiles[0, :, 1].tolist(), dtype=float)
        scale = _seasonal_scale(q_hist, horizon)
        if scale is not None:
            last_q = float(q_hist["q_cfs"].iloc[-1])
            seasonal = last_q * scale
            median = 0.5 * median + 0.5 * seasonal
        return [max(0.0, float(x)) for x in median]
    except Exception as exc:
        print(f"[chronos] inference failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def forecast_station(
    station_id: str,
    lat: float,
    lon: float,
    *,
    horizon: int = HORIZON_DAYS,
    history_days: int = TRAIN_LOOKBACK_DAYS,
) -> StationForecast:
    today = date.today()
    start = today - timedelta(days=history_days)

    notes: List[str] = []
    q_hist = usgs.fetch_daily_discharge(station_id, start, today)
    if q_hist.empty:
        raise RuntimeError(f"no USGS daily discharge for {station_id}")

    try:
        wx_hist = weather.fetch_history(lat, lon, start, today - timedelta(days=1))
    except Exception as exc:
        notes.append(f"weather history failed: {exc}")
        wx_hist = pd.DataFrame(columns=["date"] + weather.DAILY_VARS)

    try:
        wx_fcst = weather.fetch_forecast(lat, lon, days=horizon + 2)
    except Exception as exc:
        notes.append(f"weather forecast failed: {exc}")
        wx_fcst = pd.DataFrame(columns=["date"] + weather.DAILY_VARS)

    last_date = pd.to_datetime(q_hist["date"].iloc[-1])
    future_dates = [(last_date + timedelta(days=h)).date().isoformat() for h in range(1, horizon + 1)]

    members: Dict[str, List[dict]] = {}

    persist = persistence_forecast(q_hist, horizon)
    members["persistence_lag1"] = [{"date": d, "q_cfs": v} for d, v in zip(future_dates, persist)]

    try:
        ridge_pred, ridge_mae = runoff_ridge_forecast(q_hist, wx_hist, wx_fcst, horizon)
    except Exception as exc:
        notes.append(f"ridge failed: {exc}")
        ridge_pred = persist
        ridge_mae = {}
    members["runoff_ridge"] = [{"date": d, "q_cfs": v} for d, v in zip(future_dates, ridge_pred)]

    chronos_pred = chronos_forecast(q_hist, horizon)
    if chronos_pred is not None:
        members["chronos_bolt"] = [{"date": d, "q_cfs": v} for d, v in zip(future_dates, chronos_pred)]
    else:
        notes.append("chronos_bolt unavailable")

    rolling_mae = {"runoff_ridge": ridge_mae.get("mae_mean", float("inf"))}
    persist_mae = _rolling_persistence_mae(q_hist, horizon)
    if persist_mae is not None:
        rolling_mae["persistence_lag1"] = persist_mae
    if chronos_pred is not None:
        chronos_mae = _rolling_chronos_mae(q_hist, horizon)
        if chronos_mae is not None:
            rolling_mae["chronos_bolt"] = chronos_mae

    weights = _blend_weights(rolling_mae, list(members.keys()))
    blend_vals = []
    for h in range(horizon):
        s = 0.0
        wsum = 0.0
        for name, w in weights.items():
            v = members[name][h]["q_cfs"]
            if v is None or not math.isfinite(v):
                continue
            s += w * v
            wsum += w
        blend_vals.append(s / wsum if wsum > 0 else float("nan"))

    chosen = min(rolling_mae, key=lambda k: rolling_mae[k]) if rolling_mae else "runoff_ridge"

    history_out = [
        {"date": pd.Timestamp(d).date().isoformat(), "q_cfs": float(q)}
        for d, q in zip(q_hist["date"].tail(60), q_hist["q_cfs"].tail(60))
    ]

    return StationForecast(
        station_id=station_id,
        issued_at=pd.Timestamp.utcnow().isoformat(),
        history=history_out,
        members=members,
        blend=[{"date": d, "q_cfs": v} for d, v in zip(future_dates, blend_vals)],
        weights=weights,
        rolling_mae={k: float(v) for k, v in rolling_mae.items()},
        chosen=chosen,
        notes=notes,
    )


def _blend_weights(rolling_mae: Dict[str, float], member_names: List[str]) -> Dict[str, float]:
    """Inverse-MAE weighting, restricted to members we have."""
    weights = {}
    for name in member_names:
        m = rolling_mae.get(name)
        if m is None or not math.isfinite(m) or m <= 0:
            weights[name] = 0.05
        else:
            weights[name] = 1.0 / m
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _rolling_persistence_mae(q_hist: pd.DataFrame, horizon: int) -> Optional[float]:
    """Mean MAE across h=1..horizon for persistence (yhat[t+h] = q[t]).

    Computed on the trailing window so it's directly comparable to ridge/chronos.
    """
    if len(q_hist) < horizon + 30:
        return None
    y = q_hist["q_cfs"].values
    maes = []
    for h in range(1, horizon + 1):
        diffs = np.abs(y[h:] - y[:-h])[-180:]
        if len(diffs) == 0:
            continue
        maes.append(float(np.mean(diffs)))
    return float(np.mean(maes)) if maes else None


def _rolling_chronos_mae(q_hist: pd.DataFrame, horizon: int) -> Optional[float]:
    """Backtest chronos on a single horizon-length holdout. Mean MAE over h=1..horizon."""
    if len(q_hist) < horizon + 90:
        return None
    pipe = _get_chronos()
    if pipe is None:
        return None
    try:
        import torch
        ctx_len = len(q_hist) - horizon
        ctx = torch.tensor(q_hist["q_cfs"].iloc[:ctx_len].astype(float).tolist())
        quantiles, _ = pipe.predict_quantiles(
            inputs=ctx,
            prediction_length=horizon,
            quantile_levels=[0.5],
        )
        yhat = np.clip(np.array(quantiles[0, :, 0]), 0, None)
        ytrue = q_hist["q_cfs"].iloc[ctx_len:ctx_len + horizon].values
        if len(ytrue) < horizon:
            return None
        return float(mean_absolute_error(ytrue, yhat))
    except Exception:
        return None
