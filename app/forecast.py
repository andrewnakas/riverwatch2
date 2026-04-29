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
TRAIN_LOOKBACK_DAYS = 10950  # 30 years (capped); per-sensor full available history
LAGS = [1, 2, 3, 5, 7, 14, 30, 60]
PRECIP_WINDOWS = [1, 3, 7, 14, 30]
TEMP_WINDOWS = [3, 7, 14]
PRECIP_LAGS = [1, 2, 3, 5, 7]  # explicit precip-day lags so ridge can learn basin lag time


@dataclass
class StationForecast:
    station_id: str
    issued_at: str
    history: List[dict]
    members: Dict[str, List[dict]]
    blend: List[dict]
    weights: Dict[str, float]
    rolling_mae: Dict[str, float]
    rolling_mae_h7: Dict[str, float] = field(default_factory=dict)
    rolling_mae_h14: Dict[str, float] = field(default_factory=dict)
    chosen: str = ""
    weights_strategy: str = ""
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
        for lag in PRECIP_LAGS:
            df[f"precip_lag_{lag}"] = df["precipitation_sum"].shift(lag)
    if "rain_sum" in df.columns:
        for w in [1, 3, 7]:
            df[f"rain_{w}d"] = df["rain_sum"].rolling(w, min_periods=1).sum()
    if "temperature_2m_mean" in df.columns:
        for w in TEMP_WINDOWS:
            df[f"tmean_{w}d"] = df["temperature_2m_mean"].rolling(w, min_periods=1).mean()
        df["pos_dd_7d"] = df["temperature_2m_mean"].clip(lower=0).rolling(7, min_periods=1).sum()
        df["pos_dd_30d"] = df["temperature_2m_mean"].clip(lower=0).rolling(30, min_periods=1).sum()
    if "temperature_2m_max" in df.columns:
        df["tmax_3d"] = df["temperature_2m_max"].rolling(3, min_periods=1).mean()
        df["pos_dd_max_7d"] = df["temperature_2m_max"].clip(lower=0).rolling(7, min_periods=1).sum()
    if "snowfall_sum" in df.columns:
        df["snow_7d"] = df["snowfall_sum"].rolling(7, min_periods=1).sum()
        df["snow_30d"] = df["snowfall_sum"].rolling(30, min_periods=1).sum()
        df["snow_90d"] = df["snowfall_sum"].rolling(90, min_periods=1).sum()
    if "et0_fao_evapotranspiration" in df.columns:
        df["et_7d"] = df["et0_fao_evapotranspiration"].rolling(7, min_periods=1).sum()
        df["et_30d"] = df["et0_fao_evapotranspiration"].rolling(30, min_periods=1).sum()
    if "shortwave_radiation_sum" in df.columns:
        df["solar_7d"] = df["shortwave_radiation_sum"].rolling(7, min_periods=1).sum()

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


RIDGE_ALPHAS = (0.3, 1.0, 3.0, 10.0)


def runoff_ridge_forecast(
    q_hist: pd.DataFrame,
    wx_hist: pd.DataFrame,
    wx_fcst: pd.DataFrame,
    horizon: int,
) -> tuple[List[float], Dict[str, float]]:
    """Direct multi-step ridge: one model per horizon day, no recursion.

    Each model predicts log-discharge at t+h using only features known at t,
    so there's no compounding inference error across the horizon. Per-horizon
    alpha is picked from RIDGE_ALPHAS on the trailing 30-day holdout.
    Returns (predictions_cfs, {'mae_mean': ..., 'mae_h<n>': ... per horizon}).
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

    preds: List[float] = []
    per_horizon_mae: Dict[int, float] = {}

    feats_now = feats.loc[last_date, cols] if last_date in feats.index else feats[cols].dropna().iloc[-1]
    feats_now = feats_now.fillna(0.0)

    for h in range(1, horizon + 1):
        df_h = feats.copy()
        df_h["target_log"] = df_h["q_log"].shift(-h)
        train = df_h.dropna(subset=cols + ["target_log"])
        if len(train) < 30:
            preds.append(last_q)
            continue

        # Pick alpha on a trailing 30-day holdout when we have enough data.
        best_alpha = 1.0
        if len(train) > 90:
            cut = len(train) - 30
            Xtr_v = train[cols].iloc[:cut].values
            ytr_v = train["target_log"].iloc[:cut].values
            Xv = train[cols].iloc[cut:].values
            yv_log = train["target_log"].iloc[cut:].values
            yv_true = np.expm1(yv_log)
            sc_v = StandardScaler().fit(Xtr_v)
            best_score = float("inf")
            best_yhat = None
            for alpha in RIDGE_ALPHAS:
                m = Ridge(alpha=alpha, random_state=0).fit(sc_v.transform(Xtr_v), ytr_v)
                yh = np.clip(np.expm1(m.predict(sc_v.transform(Xv))), 0, None)
                mae = float(mean_absolute_error(yv_true, yh))
                if mae < best_score:
                    best_score = mae
                    best_alpha = alpha
                    best_yhat = yh
            per_horizon_mae[h] = best_score

        Xtr = train[cols].values
        ytr = train["target_log"].values
        sc = StandardScaler().fit(Xtr)
        model = Ridge(alpha=best_alpha, random_state=0).fit(sc.transform(Xtr), ytr)
        x = sc.transform(feats_now.values.reshape(1, -1))
        yhat_log = float(model.predict(x)[0])
        yhat = float(np.expm1(yhat_log))
        if not math.isfinite(yhat) or yhat < 0:
            yhat = last_q
        preds.append(yhat)

    out: Dict[str, float] = {}
    if per_horizon_mae:
        out["mae_mean"] = float(np.mean(list(per_horizon_mae.values())))
        for h, v in per_horizon_mae.items():
            out[f"mae_h{h}"] = float(v)
    return preds, out


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

    persist_per_h = _rolling_persistence_mae_per_horizon(q_hist, horizon)
    chronos_per_h: Dict[int, float] = {}
    if chronos_pred is not None:
        chronos_per_h = _rolling_chronos_mae_per_horizon(q_hist, horizon)
        if chronos_per_h:
            rolling_mae["chronos_bolt"] = float(np.mean(list(chronos_per_h.values())))

    def _per_h_lookup(per_h: Dict[int, float], h: int) -> Optional[float]:
        if per_h.get(h) is not None:
            return per_h[h]
        # fall back to nearest available horizon
        if not per_h:
            return None
        keys = sorted(per_h.keys())
        nearest = min(keys, key=lambda k: abs(k - h))
        return per_h[nearest]

    rolling_mae_h7: Dict[str, float] = {}
    rolling_mae_h14: Dict[str, float] = {}
    for h_target, target_dict in [(7, rolling_mae_h7), (14, rolling_mae_h14)]:
        if h_target > horizon:
            continue
        v = _per_h_lookup(persist_per_h, h_target)
        if v is not None:
            target_dict["persistence_lag1"] = v
        v = ridge_mae.get(f"mae_h{h_target}")
        if v is not None:
            target_dict["runoff_ridge"] = float(v)
        v = _per_h_lookup(chronos_per_h, h_target)
        if v is not None:
            target_dict["chronos_bolt"] = v

    soft_weights = _blend_weights(rolling_mae, list(members.keys()))

    # Per-station auto-pick: if the best member is decisively better than the
    # runner-up (>= 30% lower MAE), snap weights to ~all on the winner. Stops
    # the blend from being dragged by weak members on rivers where one model
    # clearly dominates (e.g. snowmelt sites where ridge beats chronos badly).
    weights = soft_weights
    weights_strategy = "soft_blend_inv_mae2"
    valid_mae = {k: v for k, v in rolling_mae.items()
                 if k in members and v is not None and math.isfinite(v) and v > 0}
    if len(valid_mae) >= 2:
        ranked = sorted(valid_mae.items(), key=lambda kv: kv[1])
        best_name, best_mae = ranked[0]
        runner_mae = ranked[1][1]
        if runner_mae > 0 and (runner_mae - best_mae) / runner_mae >= 0.30:
            # Decisive winner: 90% on it, 10% spread evenly to keep some safety.
            others = [n for n in members.keys() if n != best_name]
            weights = {best_name: 0.9}
            for n in others:
                weights[n] = 0.1 / max(1, len(others))
            weights_strategy = f"snap_to:{best_name}"

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

    # Estimate ensemble MAE per horizon = weighted geometric mean of member MAEs.
    # (Approximation: blend's MAE is roughly the weight-weighted member MAE,
    # bounded above by min member MAE in the limit of all weight on one model.)
    for target_dict in (rolling_mae_h7, rolling_mae_h14):
        if not target_dict:
            continue
        s = 0.0; ws = 0.0
        for name, w in weights.items():
            v = target_dict.get(name)
            if v is None or not math.isfinite(v):
                continue
            s += w * v; ws += w
        if ws > 0:
            target_dict["ensemble_blend"] = float(s / ws)

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
        rolling_mae_h7={k: float(v) for k, v in rolling_mae_h7.items()},
        rolling_mae_h14={k: float(v) for k, v in rolling_mae_h14.items()},
        chosen=chosen,
        weights_strategy=weights_strategy,
        notes=notes,
    )


def _blend_weights(rolling_mae: Dict[str, float], member_names: List[str]) -> Dict[str, float]:
    """Inverse-MAE-squared weighting: w ∝ 1/MAE^2.

    Concentrates more aggressively on the best member than 1/MAE while still
    falling back to the runner-up when MAEs are close. A single dominant model
    with half the MAE of the next best gets ~4x its weight (vs 2x with linear).
    """
    weights = {}
    for name in member_names:
        m = rolling_mae.get(name)
        if m is None or not math.isfinite(m) or m <= 0:
            weights[name] = 0.01
        else:
            weights[name] = 1.0 / (m * m)
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


def _rolling_persistence_mae_per_horizon(q_hist: pd.DataFrame, max_horizon: int) -> Dict[int, float]:
    """Per-horizon persistence MAE: mae[h] = mean |q[t+h] - q[t]| on trailing 180d."""
    if len(q_hist) < max_horizon + 30:
        return {}
    y = q_hist["q_cfs"].values
    out: Dict[int, float] = {}
    for h in range(1, max_horizon + 1):
        diffs = np.abs(y[h:] - y[:-h])[-180:]
        if len(diffs) == 0:
            continue
        out[h] = float(np.mean(diffs))
    return out


def _rolling_chronos_mae(q_hist: pd.DataFrame, horizon: int) -> Optional[float]:
    """Backtest chronos on a single horizon-length holdout. Mean MAE over h=1..horizon."""
    res = _rolling_chronos_mae_per_horizon(q_hist, horizon)
    if not res:
        return None
    return float(np.mean(list(res.values())))


def _rolling_chronos_mae_per_horizon(q_hist: pd.DataFrame, horizon: int) -> Dict[int, float]:
    """Chronos backtest with per-horizon errors. One forward pass at length=horizon."""
    if len(q_hist) < horizon + 90:
        return {}
    pipe = _get_chronos()
    if pipe is None:
        return {}
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
            return {}
        return {h: float(abs(ytrue[h - 1] - yhat[h - 1])) for h in range(1, horizon + 1)}
    except Exception:
        return {}
