"""TimesFM 2.5 — both the univariate and XReg-covariate forecast paths.

This module owns the TimesFM 2.5 model loading. Two functions exposed:

  forecast_univariate(q_hist, horizon)           — drop-in for the legacy
                                                    timesfm 2.0 univariate
                                                    member; uses the 2.5
                                                    backbone with no covariates
  forecast(q_hist, wx_hist, wx_fcst, horizon, *,
           static_attrs=None)                    — XReg path with precip/temp
                                                    as future-known forcing

The 2.5 release dropped the legacy `timesfm.TimesFm` class entirely, so the
earlier 2.0 path in forecast.py would fail under 2.5. Routing both members
through this module means we only carry one set of model weights and one
broken-API workaround (#412 — proxies kwarg).


TimesFM 2.5 (Oct 2025) added in-context covariate support via XReg: the model
takes past target observations PLUS past+future values of exogenous variables
(precip, temp), fits a linear model on residuals, and combines with the
foundation-model forecast.

Why this is hydrologically promising:
- TimesFM 2.0 is purely univariate — it sees only past discharge. It can't know
  there's a 2" rain event arriving on day 5.
- Adding precip + temp as XReg gives the foundation model the same kind of
  forcing signal NWM uses (just learned implicitly rather than physics-based).
- "timesfm + xreg" mode is the right choice for hydrology: the linear xreg
  fits the *response* to forcing, then TimesFM forecasts the residual seasonal
  pattern. (The "xreg + timesfm" alternative would treat covariates as a
  *correction* to baseflow, which makes less physical sense for storm response.)

Failure modes we handle gracefully:
- jax not installed → return None (XReg requires jax via timesfm[xreg])
- timesfm 2.5 weights not yet downloaded → falls through, blend just drops the
  member like it does for any other missing forecast
- XReg requires future covariates to extend exactly `horizon` days past the
  context end; we trim/pad to that length before calling.

Gated by RW2_ENABLE_TIMESFM_XREG to avoid breaking deploys until weights and
jax wheels are warm in CI cache.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 2.5 supports up to 2048 like 2.0, but XReg shape mismatches blow up if the
# covariate arrays don't match context length, so we keep this constant in one
# place and feed both inputs at this length.
TFM25_CONTEXT = 2048
TFM25_REPO = "google/timesfm-2.5-200m-pytorch"

# Lazy globals
_pipeline = None
_load_failed = False


def _enabled() -> bool:
    return os.environ.get("RW2_ENABLE_TIMESFM_XREG") == "1"


def _get_pipeline(*, require_xreg: bool):
    """Load TimesFM 2.5. Cached after first call. Returns None on any failure.

    The XReg-only flag (`require_xreg=True`) gates the additional jax/sklearn
    imports the covariate path needs. The univariate path always tries to load
    — it replaces the legacy 2.0 `timesfm` ensemble member that we removed
    when 2.5's API change made the old loader unusable.
    """
    global _pipeline, _load_failed
    if _pipeline is not None or _load_failed:
        return _pipeline
    if require_xreg and not _enabled():
        # Don't poison the cache — the univariate caller may still want to
        # load even when the XReg gate is off.
        return None
    try:
        import timesfm  # type: ignore
        from timesfm import TimesFM_2p5_200M_torch  # type: ignore
        from timesfm import ForecastConfig  # type: ignore

        if require_xreg:
            # Check XReg deps before loading 400MB of weights — if jax/sklearn are
            # missing, fail fast.
            import jax  # type: ignore  # noqa: F401
            import sklearn  # type: ignore  # noqa: F401

        # Bypass TimesFM 2.5's `from_pretrained` because the upstream HF
        # PyTorchModelHubMixin passes `proxies` (and other network kwargs)
        # through to `__init__`, which only accepts `torch_compile`/`config`
        # — see google-research/timesfm#412. The library's own `_from_pretrained`
        # does the actual checkpoint download + model construction; we call
        # it directly with the kwargs it does accept.
        model = TimesFM_2p5_200M_torch._from_pretrained(
            model_id=TFM25_REPO,
            revision=None,
            cache_dir=None,
            force_download=False,
            local_files_only=False,
            token=None,
        )
        # `return_backcast=True` is *required* for forecast_with_covariates.
        #
        # `normalize_inputs=False`: skip the outer ReVIN at
        # timesfm_2p5_torch.py:413-417 and rely only on the model's internal
        # patch-level mask-aware ReVIN (timesfm_2p5_torch.py:173, via
        # update_running_stats). The outer ReVIN would just average against
        # the patch-level one for non-stationary discharge series.
        #
        # `force_flip_invariance=False`: the default flip-invariance leg
        # decodes BOTH `inputs` and `-inputs`, then averages
        # `(f(x) - f(-x)) / 2` (timesfm_2p5_torch.py:431-443). For nonneg
        # streamflow the flipped leg is run on a series TimesFM has never
        # seen (negative discharge with patch-level ReVIN re-centering it
        # to ~0 mean), and the average just adds noise. v13.5 backtests
        # showed median TimesFM/Chronos MAE ratio of 3.2x with flip on;
        # disabling it should narrow that gap. infer_is_positive=True still
        # clips to >=0, so we keep the nonneg guarantee.
        model.compile(
            ForecastConfig(
                max_context=TFM25_CONTEXT,
                max_horizon=14,
                normalize_inputs=False,
                force_flip_invariance=False,
                use_continuous_quantile_head=False,
                return_backcast=True,
                infer_is_positive=True,  # discharge is nonneg; 2.5 enforces it
                per_core_batch_size=1,
            )
        )
        _pipeline = model
    except Exception as exc:
        _load_failed = True
        print(f"[timesfm_xreg] disabled: {exc}")
        _pipeline = None
    return _pipeline


def _aligned_cov(
    wx_hist: pd.DataFrame,
    wx_fcst: pd.DataFrame,
    cov_col: str,
    *,
    context_len: int,
    horizon: int,
) -> Optional[np.ndarray]:
    """Build a covariate vector of length context_len + horizon, aligned so the
    last `horizon` entries are the future-known forecast.

    Open-Meteo can return NaN occasionally; we forward-fill and fall back to
    zeros if a column is entirely missing.
    """
    if cov_col not in wx_hist.columns and cov_col not in wx_fcst.columns:
        return None
    h = wx_hist[["date", cov_col]].copy() if cov_col in wx_hist.columns else pd.DataFrame(columns=["date", cov_col])
    f = wx_fcst[["date", cov_col]].copy() if cov_col in wx_fcst.columns else pd.DataFrame(columns=["date", cov_col])
    combined = pd.concat([h, f], ignore_index=True)
    if combined.empty:
        return None
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
    s = pd.to_numeric(combined[cov_col], errors="coerce").ffill().bfill()
    if s.isna().all():
        return None
    arr = s.fillna(0.0).to_numpy(dtype=np.float32)
    # Last `horizon` are the future window; the rest is treated as past.
    # We need exactly context_len + horizon entries. Trim from the front.
    target_len = context_len + horizon
    if len(arr) < target_len:
        pad = np.full(target_len - len(arr), float(arr[0]), dtype=np.float32)
        arr = np.concatenate([pad, arr])
    elif len(arr) > target_len:
        arr = arr[-target_len:]
    return arr


def forecast_univariate(q_hist: pd.DataFrame, horizon: int) -> Optional[List[float]]:
    """TimesFM 2.5 univariate forecast — no covariates, just past discharge.

    Drop-in for the legacy 2.0 univariate path that lived in forecast.py. The
    2.5 weights/architecture are different enough that we can't share a model
    object with the 2.0 release; we just use 2.5 for both paths.

    Feeds raw CFS (no asinh) and lets 2.5 normalize internally. The asinh path
    we used with 2.0 produced ~4x-low forecasts in the first 2.5 deploy.

    Then blends 50/50 with a seasonal-climatology anchor (last_q × DOY ratio)
    the same way chronos_forecast does. Without this anchor, the deployed
    rolling MAE for TimesFM was 1904 cfs vs Chronos's 469 — the seasonal
    blend is what closes that gap, not any input-transform change.
    """
    pipe = _get_pipeline(require_xreg=False)
    if pipe is None or q_hist.empty:
        return None
    try:
        raw_ctx = q_hist["q_cfs"].astype(float).clip(lower=0).values[-TFM25_CONTEXT:]
        if len(raw_ctx) < 60:
            return None
        point, _quant = pipe.forecast(horizon=horizon, inputs=[np.asarray(raw_ctx, dtype=np.float32)])
        pred = np.asarray(point[0])[:horizon].astype(float)
        if len(pred) < horizon:
            pad = np.full(horizon - len(pred), float(pred[-1]) if len(pred) else 0.0)
            pred = np.concatenate([pred, pad])
        try:
            from .forecast import _seasonal_scale
            scale = _seasonal_scale(q_hist, horizon)
            if scale is not None:
                last_q = float(q_hist["q_cfs"].iloc[-1])
                seasonal = last_q * scale
                pred = 0.5 * pred + 0.5 * seasonal
        except Exception:
            pass
        return [max(0.0, float(x)) for x in pred]
    except Exception as exc:
        print(f"[timesfm_univariate] inference failed: {exc}")
        return None


def forecast(
    q_hist: pd.DataFrame,
    wx_hist: pd.DataFrame,
    wx_fcst: pd.DataFrame,
    horizon: int,
    *,
    static_attrs: Optional[Dict[str, float]] = None,
) -> Optional[List[float]]:
    """Run TimesFM 2.5 + XReg with precip/temp as future-known forcing.

    Returns horizon CFS values, or None if anything in the chain fails (the
    blend drops missing members gracefully).
    """
    pipe = _get_pipeline(require_xreg=True)
    if pipe is None or q_hist.empty:
        return None
    try:
        raw = q_hist["q_cfs"].astype(float).clip(lower=0).to_numpy()
        ctx = raw[-TFM25_CONTEXT:].astype(np.float32)
        ctx_len = int(len(ctx))
        if ctx_len < 60:  # too short for a useful linear xreg fit
            return None

        precip = _aligned_cov(wx_hist, wx_fcst, "precipitation_sum",
                              context_len=ctx_len, horizon=horizon)
        tmax = _aligned_cov(wx_hist, wx_fcst, "temperature_2m_max",
                            context_len=ctx_len, horizon=horizon)
        tmin = _aligned_cov(wx_hist, wx_fcst, "temperature_2m_min",
                            context_len=ctx_len, horizon=horizon)
        dyn_num = {}
        if precip is not None:
            dyn_num["precip"] = [precip.tolist()]
        if tmax is not None:
            dyn_num["tmax"] = [tmax.tolist()]
        if tmin is not None:
            dyn_num["tmin"] = [tmin.tolist()]
        if not dyn_num:
            return None  # no covariates → no point invoking xreg

        # Static numerical: lat/lon/log_alt/log_area help the linear model
        # condition on basin character (same intuition as pooled_lgbm).
        stat_num = {}
        if static_attrs:
            for k in ("lat", "lon", "alt_ft", "drain_area_sqmi"):
                v = static_attrs.get(k)
                if v is None:
                    continue
                if k in ("alt_ft", "drain_area_sqmi"):
                    try:
                        v = float(np.log1p(max(0.0, float(v))))
                    except Exception:
                        continue
                stat_num[f"s_{k}"] = [float(v)]

        point_fc, _xreg_fc = pipe.forecast_with_covariates(
            inputs=[ctx.tolist()],
            dynamic_numerical_covariates=dyn_num,
            static_numerical_covariates=stat_num if stat_num else None,
            xreg_mode="timesfm + xreg",
            normalize_xreg_target_per_input=True,
            ridge=1.0,  # regularize the linear fit (small samples)
            force_on_cpu=True,  # avoid jax/cuda issues on CI runners
        )
        # forecast_with_covariates returns (point, quantile). Use point.
        pred = np.asarray(point_fc[0])[:horizon]
        if len(pred) < horizon:
            pad = np.full(horizon - len(pred), float(pred[-1]) if len(pred) else 0.0)
            pred = np.concatenate([pred, pad])
        return [max(0.0, float(x)) for x in pred]
    except Exception as exc:
        print(f"[timesfm_xreg] inference failed: {exc}")
        return None
