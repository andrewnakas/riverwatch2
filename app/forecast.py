"""Live river-discharge forecasting engine.

Five forecasters are wired together and blended:
  1. persistence_lag1            — yhat[t+1] = q[t]
  2. runoff_ridge                — Ridge on lagged-discharge + DOY + recent precip/temp
  3. chronos_bolt (optional)     — Amazon Chronos-Bolt zero-shot foundation model
  4. ttm (optional)              — IBM Tiny Time Mixers foundation model
  5. timesfm (optional)          — Google TimesFM 2.0 zero-shot foundation model

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

from . import snotel, usgs, usgs_stats, weather

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HORIZON_DAYS = 14
TRAIN_LOOKBACK_DAYS = 36500  # 100 years (capped); USGS goes back this far for many gauges
LOOKBACK_FLOOR = date(1900, 1, 1)  # never request data before this regardless of lookback
MAPE_FLOOR_CFS = 1.0  # avoid blowup near zero-flow days
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
    rolling_mape: Dict[str, float] = field(default_factory=dict)
    rolling_mape_h7: Dict[str, float] = field(default_factory=dict)
    rolling_mape_h14: Dict[str, float] = field(default_factory=dict)
    chosen: str = ""
    weights_strategy: str = ""
    notes: List[str] = field(default_factory=list)
    daily_stats: Optional[dict] = None
    record_start: Optional[str] = None
    record_end: Optional[str] = None
    snotel_site: Optional[dict] = None
    snotel_summary: Optional[dict] = None


def _build_features(
    q_hist: pd.DataFrame,
    wx: pd.DataFrame,
    snotel_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = q_hist.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").asfreq("D")
    df["q_cfs"] = pd.to_numeric(df["q_cfs"], errors="coerce").interpolate(limit=3)

    wx2 = wx.copy()
    wx2["date"] = pd.to_datetime(wx2["date"])
    wx2 = wx2.set_index("date").asfreq("D")
    # Old-cache rows have None for v11 columns, which makes the resulting Series
    # object-dtype; arithmetic like .diff() then hits `None - float`. Coerce.
    for c in wx2.columns:
        wx2[c] = pd.to_numeric(wx2[c], errors="coerce")
    df = df.join(wx2, how="left")

    if snotel_df is not None and not snotel_df.empty:
        s = snotel_df.copy()
        s["date"] = pd.to_datetime(s["date"])
        s = s.set_index("date").asfreq("D")
        snotel_cols = [c for c in ("swe_in", "snow_depth_in") if c in s.columns]
        for c in snotel_cols:
            s[c] = pd.to_numeric(s[c], errors="coerce")
        df = df.join(s[snotel_cols], how="left")

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
        # 180d / 365d cumulative snowfall captures multi-season buildup that
        # melts unevenly into spring runoff (v11).
        df["snow_180d"] = df["snowfall_sum"].rolling(180, min_periods=1).sum()
        df["snow_365d"] = df["snowfall_sum"].rolling(365, min_periods=1).sum()
    if "et0_fao_evapotranspiration" in df.columns:
        df["et_7d"] = df["et0_fao_evapotranspiration"].rolling(7, min_periods=1).sum()
        df["et_30d"] = df["et0_fao_evapotranspiration"].rolling(30, min_periods=1).sum()
        df["et_180d"] = df["et0_fao_evapotranspiration"].rolling(180, min_periods=1).sum()
        df["et_365d"] = df["et0_fao_evapotranspiration"].rolling(365, min_periods=1).sum()
    if "shortwave_radiation_sum" in df.columns:
        df["solar_7d"] = df["shortwave_radiation_sum"].rolling(7, min_periods=1).sum()
    # v11: explicit cold-tail of temperature distribution helps freeze/thaw timing.
    if "temperature_2m_min" in df.columns:
        df["tmin_3d"] = df["temperature_2m_min"].rolling(3, min_periods=1).mean()
        df["tmin_7d"] = df["temperature_2m_min"].rolling(7, min_periods=1).mean()
        # Cumulative freezing-degree-days drive snowpack buildup.
        df["neg_dd_30d"] = (-df["temperature_2m_min"]).clip(lower=0).rolling(30, min_periods=1).sum()
    # v11: soil moisture acts as the catchment's "filled-bucket" indicator —
    # deeper layers especially carry the slow-recession baseflow signal.
    for col in ("soil_moisture_0_to_10cm_mean", "soil_moisture_28_to_100cm_mean"):
        if col in df.columns:
            short = "sm_top" if "0_to_10" in col else "sm_deep"
            df[short] = df[col]
            df[f"{short}_lag1"] = df[col].shift(1)
            df[f"{short}_lag7"] = df[col].shift(7)
            df[f"{short}_30d"] = df[col].rolling(30, min_periods=1).mean()
    if "soil_temperature_0_to_7cm_mean" in df.columns:
        df["soilt"] = df["soil_temperature_0_to_7cm_mean"]
        df["soilt_7d"] = df["soil_temperature_0_to_7cm_mean"].rolling(7, min_periods=1).mean()
    # v11: snow_depth_max gives a direct (rather than cumulative) pack measure;
    # diff vs lagged values approximates the daily melt rate.
    if "snow_depth_max" in df.columns:
        df["snow_depth"] = df["snow_depth_max"]
        df["snow_depth_lag1"] = df["snow_depth_max"].shift(1)
        df["snow_depth_lag7"] = df["snow_depth_max"].shift(7)
        df["snow_depth_change_7d"] = df["snow_depth_max"].diff(7)
        # SWE proxy: snow_depth (m) * mean density of seasonal snowpack ~0.30.
        df["swe_proxy"] = df["snow_depth_max"] * 0.30
    # v11: real SNOTEL SWE (when within 50 km). The 7d/30d change captures
    # active melt; the absolute value tracks how much spring runoff is loaded.
    if "swe_in" in df.columns:
        df["swe_lag1"] = df["swe_in"].shift(1)
        df["swe_lag7"] = df["swe_in"].shift(7)
        df["swe_change_7d"] = df["swe_in"].diff(7)
        df["swe_change_30d"] = df["swe_in"].diff(30)
    if "snow_depth_in" in df.columns:
        df["sntl_depth_change_7d"] = df["snow_depth_in"].diff(7)

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
    snotel_df: Optional[pd.DataFrame] = None,
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

    feats = _build_features(q_hist, wx_combined, snotel_df=snotel_df)
    cols = _feature_columns(feats)
    if not cols:
        return persistence_forecast(q_hist, horizon), {}

    last_date = pd.to_datetime(q_hist["date"].iloc[-1])
    last_q = float(q_hist["q_cfs"].iloc[-1])

    preds: List[float] = []
    per_horizon_mae: Dict[int, float] = {}
    per_horizon_mape: Dict[int, float] = {}

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
            if best_yhat is not None:
                per_horizon_mape[h] = float(np.mean(
                    np.abs(yv_true - best_yhat) / np.maximum(np.abs(yv_true), MAPE_FLOOR_CFS)
                ))

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
    if per_horizon_mape:
        out["__mape_per_h"] = per_horizon_mape  # type: ignore[assignment]
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
# IBM TTM (Tiny Time Mixers) zero-shot
# ---------------------------------------------------------------------------

_ttm_pipeline = None
_ttm_failed = False
TTM_MODEL = "ibm-granite/granite-timeseries-ttm-r2"
TTM_CONTEXT = 1536  # TTM-r2 supports up to 1536; longer ctx = better seasonality capture


def _get_ttm():
    global _ttm_pipeline, _ttm_failed
    if _ttm_pipeline is not None or _ttm_failed:
        return _ttm_pipeline
    try:
        from tsfm_public import TinyTimeMixerForPrediction  # type: ignore
        _ttm_pipeline = TinyTimeMixerForPrediction.from_pretrained(
            TTM_MODEL, prediction_filter_length=HORIZON_DAYS
        )
        _ttm_pipeline.eval()
    except Exception as exc:
        _ttm_failed = True
        print(f"[ttm] disabled: {exc}")
        _ttm_pipeline = None
    return _ttm_pipeline


def ttm_forecast(q_hist: pd.DataFrame, horizon: int) -> Optional[List[float]]:
    pipe = _get_ttm()
    if pipe is None or q_hist.empty or len(q_hist) < 64:
        return None
    try:
        import torch
        # log1p stabilizes the heavy-tailed discharge distribution; reverse at the end.
        raw = q_hist["q_cfs"].astype(float).clip(lower=0).values
        series = np.log1p(raw[-TTM_CONTEXT:])
        # Left-pad to TTM_CONTEXT with the earliest observed value so the model still runs
        # on shorter records.
        if len(series) < TTM_CONTEXT:
            pad_val = float(series[0]) if len(series) else 0.0
            series = np.concatenate([np.full(TTM_CONTEXT - len(series), pad_val), series])
        x = torch.tensor(series, dtype=torch.float32).reshape(1, TTM_CONTEXT, 1)
        with torch.no_grad():
            out = pipe(past_values=x)
        # tsfm returns (B, prediction_length, C); horizon is the second axis
        pred_log = out.prediction_outputs.squeeze(0).squeeze(-1).cpu().numpy()
        pred = np.expm1(pred_log)[:horizon]
        if len(pred) < horizon:
            pad = [float(pred[-1])] * (horizon - len(pred))
            pred = np.concatenate([pred, np.array(pad)])
        return [max(0.0, float(x)) for x in pred]
    except Exception as exc:
        print(f"[ttm] inference failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Google TimesFM 2.0 zero-shot
# ---------------------------------------------------------------------------

_timesfm_pipeline = None
_timesfm_failed = False
TIMESFM_CONTEXT = 2048  # TimesFM 2.0 supports up to 2048; longer ctx materially helps seasonality


def _get_timesfm():
    global _timesfm_pipeline, _timesfm_failed
    if _timesfm_pipeline is not None or _timesfm_failed:
        return _timesfm_pipeline
    try:
        import timesfm  # type: ignore
        _timesfm_pipeline = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=1,
                horizon_len=HORIZON_DAYS,
                context_len=TIMESFM_CONTEXT,
                num_layers=50,
                use_positional_embedding=False,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )
    except Exception as exc:
        _timesfm_failed = True
        print(f"[timesfm] disabled: {exc}")
        _timesfm_pipeline = None
    return _timesfm_pipeline


def timesfm_forecast(q_hist: pd.DataFrame, horizon: int) -> Optional[List[float]]:
    pipe = _get_timesfm()
    if pipe is None or q_hist.empty:
        return None
    try:
        # log1p stabilizes the heavy-tailed discharge distribution (same trick as TTM).
        # TimesFM is trained on raw univariate series so we feed log-flow and expm1 the output.
        raw_ctx = q_hist["q_cfs"].astype(float).clip(lower=0).values[-TIMESFM_CONTEXT:]
        ctx = np.log1p(raw_ctx)
        point, _ = pipe.forecast(
            inputs=[ctx.tolist()],
            freq=[0],  # 0 = high-frequency / daily
        )
        pred_log = np.array(point[0])[:horizon]
        pred = np.expm1(pred_log)
        if len(pred) < horizon:
            pad = [float(pred[-1])] * (horizon - len(pred))
            pred = np.concatenate([pred, np.array(pad)])
        return [max(0.0, float(x)) for x in pred]
    except Exception as exc:
        print(f"[timesfm] inference failed: {exc}")
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
    if start < LOOKBACK_FLOOR:
        start = LOOKBACK_FLOOR

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

    snotel_df: Optional[pd.DataFrame] = None
    snotel_meta: Optional[dict] = None
    # Gate SNOTEL fetches behind RW2_ENABLE_SNOTEL to avoid a slow first-build
    # cold-fill (per-site nearest-mapper + WTEQ history). Enable once we want
    # to populate the SNOTEL caches in the workflow.
    import os as _os
    if _os.environ.get("RW2_ENABLE_SNOTEL") == "1":
        try:
            site = snotel.nearest_site(station_id, lat, lon)
            if site:
                snotel_meta = site
                snotel_df = snotel.fetch_swe_history(site["stationTriplet"], start, today)
                if snotel_df is None or snotel_df.empty:
                    snotel_df = None
        except Exception as exc:
            notes.append(f"snotel failed: {exc}")
            snotel_df = None

    last_date = pd.to_datetime(q_hist["date"].iloc[-1])
    future_dates = [(last_date + timedelta(days=h)).date().isoformat() for h in range(1, horizon + 1)]

    members: Dict[str, List[dict]] = {}

    persist = persistence_forecast(q_hist, horizon)
    members["persistence_lag1"] = [{"date": d, "q_cfs": v} for d, v in zip(future_dates, persist)]

    try:
        ridge_pred, ridge_mae = runoff_ridge_forecast(
            q_hist, wx_hist, wx_fcst, horizon, snotel_df=snotel_df
        )
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

    ttm_pred = ttm_forecast(q_hist, horizon)
    if ttm_pred is not None:
        members["ttm"] = [{"date": d, "q_cfs": v} for d, v in zip(future_dates, ttm_pred)]
    else:
        notes.append("ttm unavailable")

    timesfm_pred = timesfm_forecast(q_hist, horizon)
    if timesfm_pred is not None:
        members["timesfm"] = [{"date": d, "q_cfs": v} for d, v in zip(future_dates, timesfm_pred)]
    else:
        notes.append("timesfm unavailable")

    # v11.4: every member is scored on the SAME 3 holdout windows so per-h
    # MAE values are directly comparable. Previously persistence used a 180-day
    # trailing window while foundation models used 3×horizon-day windows, which
    # warped the blend weights toward whichever member had the easier test set.
    persist_per_h, persist_mape_per_h = _score_holdouts(
        _persistence_predict_on_holdout, q_hist, horizon
    )
    ridge_per_h, ridge_mape_per_h_unified = _score_holdouts(
        _ridge_predict_on_holdout, q_hist, horizon,
        extra_args=(wx_hist, snotel_df),
    )
    chronos_per_h: Dict[int, float] = {}
    chronos_mape_per_h: Dict[int, float] = {}
    if chronos_pred is not None:
        chronos_per_h, chronos_mape_per_h = _score_holdouts(
            lambda q, h, *, end_offset: _foundation_predict_on_holdout(q, h, "chronos_bolt", end_offset=end_offset),
            q_hist, horizon,
        )
    ttm_per_h: Dict[int, float] = {}
    ttm_mape_per_h: Dict[int, float] = {}
    if ttm_pred is not None:
        ttm_per_h, ttm_mape_per_h = _score_holdouts(
            lambda q, h, *, end_offset: _foundation_predict_on_holdout(q, h, "ttm", end_offset=end_offset),
            q_hist, horizon,
        )
    timesfm_per_h: Dict[int, float] = {}
    timesfm_mape_per_h: Dict[int, float] = {}
    if timesfm_pred is not None:
        timesfm_per_h, timesfm_mape_per_h = _score_holdouts(
            lambda q, h, *, end_offset: _foundation_predict_on_holdout(q, h, "timesfm", end_offset=end_offset),
            q_hist, horizon,
        )

    rolling_mae: Dict[str, float] = {}
    if persist_per_h:
        rolling_mae["persistence_lag1"] = float(np.mean(list(persist_per_h.values())))
    if ridge_per_h:
        rolling_mae["runoff_ridge"] = float(np.mean(list(ridge_per_h.values())))
    if chronos_per_h:
        rolling_mae["chronos_bolt"] = float(np.mean(list(chronos_per_h.values())))
    if ttm_per_h:
        rolling_mae["ttm"] = float(np.mean(list(ttm_per_h.values())))
    if timesfm_per_h:
        rolling_mae["timesfm"] = float(np.mean(list(timesfm_per_h.values())))

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
        for name, per_h in (
            ("persistence_lag1", persist_per_h),
            ("runoff_ridge", ridge_per_h),
            ("chronos_bolt", chronos_per_h),
            ("ttm", ttm_per_h),
            ("timesfm", timesfm_per_h),
        ):
            v = _per_h_lookup(per_h, h_target)
            if v is not None:
                target_dict[name] = float(v)

    # MAPE comes from the same unified holdouts (no separate harness anymore).
    ridge_mape_per_h = ridge_mape_per_h_unified
    rolling_mape: Dict[str, float] = {}
    for name, per_h in (
        ("persistence_lag1", persist_mape_per_h),
        ("runoff_ridge", ridge_mape_per_h),
        ("chronos_bolt", chronos_mape_per_h),
        ("ttm", ttm_mape_per_h),
        ("timesfm", timesfm_mape_per_h),
    ):
        if per_h:
            rolling_mape[name] = float(np.mean(list(per_h.values())))

    rolling_mape_h7: Dict[str, float] = {}
    rolling_mape_h14: Dict[str, float] = {}
    for h_target, target_dict in [(7, rolling_mape_h7), (14, rolling_mape_h14)]:
        if h_target > horizon:
            continue
        for name, per_h in (
            ("persistence_lag1", persist_mape_per_h),
            ("runoff_ridge", ridge_mape_per_h),
            ("chronos_bolt", chronos_mape_per_h),
            ("ttm", ttm_mape_per_h),
            ("timesfm", timesfm_mape_per_h),
        ):
            v = _per_h_lookup(per_h, h_target)
            if v is not None:
                target_dict[name] = float(v)

    # v11.4 station-level cap: drop any member whose station-mean MAE is worse
    # than persistence's. Persistence stays in. Then build inverse-MAE² weights
    # over the survivors.
    persist_station = rolling_mae.get("persistence_lag1")
    capped_mae = dict(rolling_mae)
    if persist_station is not None and math.isfinite(persist_station):
        capped_mae = {
            k: v for k, v in rolling_mae.items()
            if k == "persistence_lag1"
            or (v is not None and math.isfinite(v) and v <= persist_station)
        }
    soft_weights = _blend_weights(capped_mae, list(capped_mae.keys()))

    # Per-station auto-pick: if the best member is decisively better than the
    # runner-up (>= 30% lower MAE), snap weights to ~all on the winner. Stops
    # the blend from being dragged by weak members on rivers where one model
    # clearly dominates (e.g. snowmelt sites where ridge beats chronos badly).
    weights = soft_weights
    weights_strategy = "per_horizon_inv_mae2"
    valid_mae = {k: v for k, v in capped_mae.items()
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

    # Per-horizon weights: for each horizon day, build inverse-MAE^2 weights from
    # that horizon's rolling MAE. v11.4: pull directly from each member's
    # per-h dict (unified harness fills 1..H natively, no interpolation needed).
    member_per_h = {
        "persistence_lag1": persist_per_h,
        "runoff_ridge": ridge_per_h,
        "chronos_bolt": chronos_per_h,
        "ttm": ttm_per_h,
        "timesfm": timesfm_per_h,
    }
    per_horizon_mae: Dict[int, Dict[str, float]] = {}
    for h in range(1, horizon + 1):
        per_h: Dict[str, float] = {}
        for name, ph in member_per_h.items():
            if name not in members:
                continue
            v = ph.get(h)
            if v is None:
                # nearest-h fallback so we still emit weights when one offset
                # window happened to miss this horizon
                if not ph:
                    continue
                near = min(ph.keys(), key=lambda k: abs(k - h))
                v = ph[near]
            per_h[name] = float(v)
        per_horizon_mae[h] = per_h

    def _weights_for_horizon(h: int) -> Dict[str, float]:
        per_h = per_horizon_mae.get(h) or {}
        if not per_h:
            return weights
        # When snap-to-winner triggered, keep snap behaviour at every horizon.
        if weights_strategy.startswith("snap_to:"):
            return weights
        # v11.4 "do no harm" cap: drop any member whose MAE at this horizon is
        # WORSE than persistence at the same horizon. The blend has been losing
        # ~5–10% on average to bad members getting non-trivial weight via
        # inverse-MAE² (which gives weak members a small but harmful share).
        # Persistence itself is always kept as the floor.
        persist_v = per_h.get("persistence_lag1")
        if persist_v is not None and math.isfinite(persist_v):
            filtered = {
                name: v for name, v in per_h.items()
                if name == "persistence_lag1"
                or (math.isfinite(v) and v <= persist_v)
            }
            if len(filtered) >= 1:
                per_h = filtered
        return _blend_weights(per_h, list(per_h.keys()))

    blend_vals = []
    for h_idx in range(horizon):
        h = h_idx + 1
        ws = _weights_for_horizon(h)
        s = 0.0
        wsum = 0.0
        for name, w in ws.items():
            v = members[name][h_idx]["q_cfs"]
            if v is None or not math.isfinite(v):
                continue
            s += w * v
            wsum += w
        blend_vals.append(s / wsum if wsum > 0 else float("nan"))

    chosen = min(rolling_mae, key=lambda k: rolling_mae[k]) if rolling_mae else "runoff_ridge"

    # Estimate ensemble MAE/MAPE per horizon = weighted average of member values.
    # (Approximation: blend's error is roughly the weight-weighted member error,
    # bounded above by min member error in the limit of all weight on one model.)
    for target_dict in (rolling_mae_h7, rolling_mae_h14, rolling_mape_h7, rolling_mape_h14):
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

    # Earliest q_hist date is our authoritative record-start (we fetched all the
    # way back to LOOKBACK_FLOOR/start, USGS returned what it has).
    record_start = pd.Timestamp(q_hist["date"].iloc[0]).date().isoformat() if len(q_hist) else None
    record_end = pd.Timestamp(q_hist["date"].iloc[-1]).date().isoformat() if len(q_hist) else None

    daily_stats = None
    try:
        daily_stats = usgs_stats.fetch_daily_stats(station_id)
    except Exception as exc:
        notes.append(f"daily_stats failed: {exc}")
    if daily_stats and daily_stats.get("begin_date"):
        # Prefer USGS's own record-start since it spans the entire WSC archive,
        # not just our lookback window.
        record_start = daily_stats["begin_date"]

    snotel_summary: Optional[dict] = None
    if snotel_df is not None and not snotel_df.empty and "swe_in" in snotel_df.columns:
        s = snotel_df.dropna(subset=["swe_in"]).reset_index(drop=True)
        if len(s):
            curr = float(s["swe_in"].iloc[-1])
            d_now = s["date"].iloc[-1]
            def _delta_at(target_offset: int) -> Optional[float]:
                target_date = pd.Timestamp(d_now) - pd.Timedelta(days=target_offset)
                # find nearest row at or before that date
                eligible = s[pd.to_datetime(s["date"]) <= target_date]
                if eligible.empty:
                    return None
                return curr - float(eligible["swe_in"].iloc[-1])
            snotel_summary = {
                "swe_in": curr,
                "swe_change_7d": _delta_at(7),
                "swe_change_30d": _delta_at(30),
                "as_of": pd.Timestamp(d_now).date().isoformat(),
            }

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
        rolling_mape={k: float(v) for k, v in rolling_mape.items()},
        rolling_mape_h7={k: float(v) for k, v in rolling_mape_h7.items()},
        rolling_mape_h14={k: float(v) for k, v in rolling_mape_h14.items()},
        chosen=chosen,
        weights_strategy=weights_strategy,
        notes=notes,
        daily_stats=daily_stats,
        record_start=record_start,
        record_end=record_end,
        snotel_site=snotel_meta,
        snotel_summary=snotel_summary,
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


# v11.4 unified the per-member harnesses into `_score_holdouts` over the same
# `_FOUNDATION_HOLDOUT_OFFSETS`. The thin shims below preserve the old function
# names for scripts/benchmark_40.py without re-introducing the disparate
# test-set bias they had before.

def _rolling_persistence_mae(q_hist: pd.DataFrame, horizon: int) -> Optional[float]:
    per_h, _ = _score_holdouts(_persistence_predict_on_holdout, q_hist, horizon)
    return float(np.mean(list(per_h.values()))) if per_h else None


def _rolling_chronos_mae(q_hist: pd.DataFrame, horizon: int) -> Optional[float]:
    per_h, _ = _score_holdouts(
        lambda q, h, *, end_offset: _foundation_predict_on_holdout(q, h, "chronos_bolt", end_offset=end_offset),
        q_hist, horizon,
    )
    return float(np.mean(list(per_h.values()))) if per_h else None


def _foundation_predict_on_holdout(
    q_hist: pd.DataFrame, horizon: int, model: str, *, end_offset: int = 0
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Run the named foundation model on a held-out tail. Returns (yhat, ytrue)
    aligned to h=1..horizon, or None if there's not enough history or the model
    is unavailable. `end_offset` shifts the holdout backwards by N days so we
    can score multiple non-overlapping windows."""
    needed = horizon + 90 + end_offset
    if len(q_hist) < needed:
        return None
    end_idx = len(q_hist) - end_offset
    ctx_len = end_idx - horizon
    if ctx_len < 64:
        return None
    hist_ctx = q_hist.iloc[:ctx_len].reset_index(drop=True)
    ytrue = q_hist["q_cfs"].iloc[ctx_len:ctx_len + horizon].values
    if len(ytrue) < horizon:
        return None

    if model == "chronos_bolt":
        pipe = _get_chronos()
        if pipe is None:
            return None
        try:
            import torch
            ctx = torch.tensor(hist_ctx["q_cfs"].astype(float).tolist())
            quantiles, _ = pipe.predict_quantiles(
                inputs=ctx, prediction_length=horizon, quantile_levels=[0.5],
            )
            yhat = np.clip(np.array(quantiles[0, :, 0]), 0, None)
        except Exception:
            return None
    elif model == "ttm":
        pred = ttm_forecast(hist_ctx, horizon)
        if pred is None:
            return None
        yhat = np.array(pred)
    elif model == "timesfm":
        pred = timesfm_forecast(hist_ctx, horizon)
        if pred is None:
            return None
        yhat = np.array(pred)
    else:
        return None

    if len(yhat) < horizon:
        return None
    return yhat[:horizon], ytrue[:horizon]


# Three rolling, non-overlapping holdouts. Each is `horizon` days long; offsets
# stagger by 30 days so we hit different seasons without overlapping the test
# windows. v11.4: every member (persistence, ridge, chronos, ttm, timesfm) is
# scored on these SAME windows so per-horizon MAE comparisons are apples-to-apples
# and the blend weights stop being warped by harness-mismatch.
_FOUNDATION_HOLDOUT_OFFSETS = (0, 30, 60)


def _persistence_predict_on_holdout(
    q_hist: pd.DataFrame, horizon: int, *, end_offset: int = 0
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Persistence on the SAME held-out tail as the foundation models. Returns
    (yhat[h=1..H], ytrue[h=1..H]) so it can share `_score_holdouts` with the rest."""
    needed = horizon + 90 + end_offset
    if len(q_hist) < needed:
        return None
    end_idx = len(q_hist) - end_offset
    ctx_len = end_idx - horizon
    if ctx_len < 64:
        return None
    last_q = float(q_hist["q_cfs"].iloc[ctx_len - 1])
    yhat = np.full(horizon, last_q, dtype=float)
    ytrue = q_hist["q_cfs"].iloc[ctx_len:ctx_len + horizon].values.astype(float)
    if len(ytrue) < horizon:
        return None
    return yhat, ytrue


def _ridge_predict_on_holdout(
    q_hist: pd.DataFrame,
    wx_hist: pd.DataFrame,
    horizon: int,
    snotel_df: Optional[pd.DataFrame],
    *,
    end_offset: int = 0,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Ridge on the SAME held-out tail. Trains per-horizon models using only
    rows whose target falls before the as-of date (no leakage), predicts the
    next horizon days, and returns (yhat, ytrue).

    No `wx_fcst` is passed: at an as-of date in the past we don't have access
    to a future forecast that would have existed back then; we use only history.
    """
    needed = horizon + 90 + end_offset
    if len(q_hist) < needed:
        return None
    end_idx = len(q_hist) - end_offset
    ctx_len = end_idx - horizon
    if ctx_len < 64:
        return None

    q_ctx = q_hist.iloc[:ctx_len].reset_index(drop=True)
    ytrue = q_hist["q_cfs"].iloc[ctx_len:ctx_len + horizon].values.astype(float)
    if len(ytrue) < horizon:
        return None
    last_q = float(q_ctx["q_cfs"].iloc[-1])

    try:
        feats = _build_features(q_ctx, wx_hist, snotel_df=snotel_df)
    except Exception:
        return None
    cols = _feature_columns(feats)
    if not cols:
        return np.full(horizon, last_q), ytrue

    last_date = pd.to_datetime(q_ctx["date"].iloc[-1])
    feats_now = (
        feats.loc[last_date, cols] if last_date in feats.index
        else feats[cols].dropna().iloc[-1]
    ).fillna(0.0)

    yhat = np.empty(horizon, dtype=float)
    for h in range(1, horizon + 1):
        df_h = feats.copy()
        df_h["target_log"] = df_h["q_log"].shift(-h)
        train = df_h.dropna(subset=cols + ["target_log"])
        if len(train) < 30:
            yhat[h - 1] = last_q
            continue
        Xtr = train[cols].values
        ytr = train["target_log"].values
        try:
            sc = StandardScaler().fit(Xtr)
            model = Ridge(alpha=1.0, random_state=0).fit(sc.transform(Xtr), ytr)
            x = sc.transform(feats_now.values.reshape(1, -1))
            yh = float(np.expm1(model.predict(x)[0]))
        except Exception:
            yh = last_q
        if not math.isfinite(yh) or yh < 0:
            yh = last_q
        yhat[h - 1] = yh
    return yhat, ytrue


def _score_holdouts(
    predictor,
    q_hist: pd.DataFrame,
    horizon: int,
    *,
    extra_args: tuple = (),
) -> tuple[Dict[int, float], Dict[int, float]]:
    """Run `predictor(q_hist, horizon, *extra_args, end_offset=off)` on each
    `_FOUNDATION_HOLDOUT_OFFSETS` and return (mae_per_h, mape_per_h) — averaged
    over offsets that returned data. Used for every member so MAE comparisons
    are apples-to-apples across the same dates."""
    mae_acc: Dict[int, list] = {h: [] for h in range(1, horizon + 1)}
    mape_acc: Dict[int, list] = {h: [] for h in range(1, horizon + 1)}
    for off in _FOUNDATION_HOLDOUT_OFFSETS:
        try:
            res = predictor(q_hist, horizon, *extra_args, end_offset=off)
        except Exception:
            res = None
        if res is None:
            continue
        yhat, ytrue = res
        for h in range(1, horizon + 1):
            err = abs(float(ytrue[h - 1]) - float(yhat[h - 1]))
            mae_acc[h].append(err)
            denom = max(abs(float(ytrue[h - 1])), MAPE_FLOOR_CFS)
            mape_acc[h].append(err / denom)
    mae_out = {h: float(np.mean(v)) for h, v in mae_acc.items() if v}
    mape_out = {h: float(np.mean(v)) for h, v in mape_acc.items() if v}
    return mae_out, mape_out


