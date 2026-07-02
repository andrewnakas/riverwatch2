"""v16: Multi-basin LSTM (MB-LSTM) — homegrown Google-Flood-Hub-style member.

Architecture (per Nearing et al. 2024, Nature 626:1011, adapted):
  - Encoder LSTM over a 365-day hindcast window of weather forcings PLUS
    observed discharge (autoregressive input — the edge neither Google's
    gauge-free design nor NWM's nudging DA fully exploits).
  - Decoder LSTM over the forecast horizon driven by forecast weather,
    initialized from the encoder's final (h, c) state.
  - Probabilistic head in per-basin normalized asinh space, cfg-selected:
      * "quantile" (default/legacy): 0.1 / 0.5 / 0.9 trained with pinball loss
        — the median is the point forecast, the outer quantiles populate
        q_lo / q_hi.
      * "cmal": Countable Mixture of Asymmetric Laplacians (Google Flood Hub,
        Nearing et al. 2024) trained by NLL — yields sharper right-skewed
        high-flow distributions, analytic quantiles, and a distribution-mean
        point forecast that sits above the median on a right-skew.

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

_models: list | None = None
_cfg: dict | None = None
_load_failed = False


def _is_enabled() -> bool:
    return os.environ.get("RW2_ENABLE_MBLSTM") == "1"


def _ckpt_paths() -> list[Path]:
    """RW2_MBLSTM_CKPT_PATH may be a colon-separated list — forecasts are
    averaged across the listed checkpoints (seed ensemble)."""
    p = os.environ.get("RW2_MBLSTM_CKPT_PATH")
    if p:
        return [Path(s) for s in p.split(":") if s]
    return [Path(__file__).resolve().parents[1] / "data" / "mblstm" / "model.pt"]


def build_model(cfg: dict):
    """Construct the torch module from a checkpoint cfg dict. Lives here so
    training and serving can never drift apart on architecture.

    head = cfg.get("head", "quantile"):
      "quantile" : the legacy MLP head emitting len(cfg["quantiles"]) values,
                   trained with pinball loss. The default, so every existing
                   checkpoint (which never wrote a "head" key) loads byte-for-
                   byte identically.
      "cmal"     : a Countable Mixture of Asymmetric Laplacians head (Google
                   Flood Hub, Nearing et al. 2024) emitting 4*K raw params per
                   (B,H): K logits (-> softmax weights), K locations, K log-
                   scales, K asymmetry logits. The head Linear is sized 4*K and
                   the *raw* output is returned by forward(); the cmal_* helper
                   functions below turn those raw params into an NLL (train) or
                   analytic quantiles / mean (serve).
    """
    import torch.nn as nn

    enc_in = len(cfg["enc_vars"]) + 2 + 2 + len(cfg["static_feats"])  # +q,+qmask,+doy
    dec_in = len(cfg["dec_vars"]) + 2 + 1 + len(cfg["static_feats"])  # +doy,+lead
    hidden = int(cfg["hidden"])
    head = cfg.get("head", "quantile")
    if head == "cmal":
        out_dim = 4 * int(cfg.get("cmal_k", 3))
    else:
        out_dim = len(cfg["quantiles"])

    class MBLSTMNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(enc_in, hidden, batch_first=True)
            self.decoder = nn.LSTM(dec_in, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, out_dim),
            )

        def forward(self, x_enc, x_dec):
            _, hc = self.encoder(x_enc)
            out, _ = self.decoder(x_dec, hc)
            # quantile head: (B, H, nq); cmal head: (B, H, 4*K) raw params.
            return self.head(out)

    return MBLSTMNet()


# --------------------------------------------------------------- CMAL head ----
# Countable Mixture of Asymmetric Laplacians (Nearing et al. 2024, Google
# Flood Hub). The conditional discharge distribution (in per-station asinh-z
# space) is a K-component mixture of Asymmetric Laplace Distributions (ALD).
#
# Parameterization (standard "quantile / asymmetry" ALD, kappa-free form using
# tau in (0,1) as the asymmetry parameter — the value at which the component's
# CDF equals tau is its location mu):
#   component density   f(y; mu, b, tau) =
#       (tau(1-tau)/b) * exp( -rho_tau((y-mu)/b) )
#   where rho_tau(u) = u*(tau - 1{u<0}) is the pinball/check function:
#       u >= 0:  rho = tau * u
#       u <  0:  rho = (tau - 1) * u = (1-tau)*|u|
#   so for y >= mu the tail decays at rate tau/b, for y < mu at rate (1-tau)/b.
#   tau -> small  => long RIGHT tail (right-skew, what high flow needs).
#
#   component CDF:
#       y <= mu:  F = tau       * exp( (1-tau) * (y-mu)/b )
#       y >  mu:  F = 1 - (1-tau)* exp( -tau    * (y-mu)/b )
#   component quantile (invert CDF), for level p in (0,1):
#       p <= tau:  y = mu + (b/(1-tau)) * log(p/tau)
#       p >  tau:  y = mu - (b/tau)     * log((1-p)/(1-tau))
#   component mean:
#       E[y] = mu + b*(1 - 2*tau)/(tau*(1-tau))
#
# Raw head output (last dim, length 4*K) is split into 4 blocks of K:
#   [logits | mu | log_b | tau_logit].
#   weights = softmax(logits);  b = softplus(log_b) + B_FLOOR;
#   tau = sigmoid(tau_logit) clamped to [TAU_EPS, 1-TAU_EPS].
#
# Numerical choices:
#   B_FLOOR = 1e-3  : scale floor keeps log-density finite if softplus -> 0.
#   TAU_EPS = 1e-3  : keeps tau strictly interior so 1/tau, 1/(1-tau),
#                     log(...) terms stay finite.
#   NLL is the masked mean of -log( sum_k w_k f_k(y) ); the per-component log
#   density is computed in log-space and combined with logsumexp for stability.

CMAL_B_FLOOR = 1e-3
CMAL_TAU_EPS = 1e-3


def _cmal_unpack(params, K, lib):
    """Split raw (..., 4K) params into (w, mu, b, tau), each (..., K).

    lib is `torch` (train) or `numpy` (serve); both expose the ops we use via
    a tiny shim so one code path serves both. Returns library-native arrays.
    """
    if lib.__name__ == "torch":
        import torch as T
        logits = params[..., 0 * K:1 * K]
        mu = params[..., 1 * K:2 * K]
        log_b = params[..., 2 * K:3 * K]
        tau_logit = params[..., 3 * K:4 * K]
        w = T.softmax(logits, dim=-1)
        b = T.nn.functional.softplus(log_b) + CMAL_B_FLOOR
        tau = T.sigmoid(tau_logit).clamp(CMAL_TAU_EPS, 1.0 - CMAL_TAU_EPS)
        return w, mu, b, tau
    # numpy path
    np_ = lib
    logits = params[..., 0 * K:1 * K]
    mu = params[..., 1 * K:2 * K]
    log_b = params[..., 2 * K:3 * K]
    tau_logit = params[..., 3 * K:4 * K]
    z = logits - logits.max(axis=-1, keepdims=True)
    ez = np_.exp(z)
    w = ez / ez.sum(axis=-1, keepdims=True)
    # softplus, overflow-safe
    b = np_.logaddexp(0.0, log_b) + CMAL_B_FLOOR
    tau = 1.0 / (1.0 + np_.exp(-tau_logit))
    tau = np_.clip(tau, CMAL_TAU_EPS, 1.0 - CMAL_TAU_EPS)
    return w, mu, b, tau


def cmal_nll(params, y, mask):
    """Masked-mean negative log-likelihood of the CMAL mixture (torch).

    params: (B, H, 4K) raw head output. y, mask: (B, H). Returns a scalar.
    """
    import torch as T
    K = params.shape[-1] // 4
    w, mu, b, tau = _cmal_unpack(params, K, T)
    yk = y.unsqueeze(-1)                       # (B,H,1)
    u = (yk - mu) / b                          # (B,H,K)
    # rho_tau(u) = u*(tau - 1{u<0}); for u>=0 -> tau*u, for u<0 -> (tau-1)*u
    rho = T.where(u >= 0, tau * u, (tau - 1.0) * u)
    # log f_k = log(tau) + log(1-tau) - log(b) - rho
    log_fk = T.log(tau) + T.log(1.0 - tau) - T.log(b) - rho
    log_mix = T.logsumexp(T.log(w) + log_fk, dim=-1)   # (B,H)
    nll = -log_mix
    return (nll * mask).sum() / mask.sum().clamp(min=1)


def cmal_mean(params, lib=None):
    """Mixture mean E[y] = sum_k w_k * (mu_k + b_k*(1-2*tau_k)/(tau_k*(1-tau_k))).

    On a right-skewed mixture this sits ABOVE the median — the peak-aware point
    estimate the pinball median could not give. lib defaults to numpy (serve);
    pass torch for the torch path.
    """
    if lib is None:
        lib = np
    K = params.shape[-1] // 4
    w, mu, b, tau = _cmal_unpack(params, K, lib)
    comp_mean = mu + b * (1.0 - 2.0 * tau) / (tau * (1.0 - tau))
    return (w * comp_mean).sum(axis=-1)


def _cmal_cdf(z, w, mu, b, tau, lib):
    """Mixture CDF at scalar/broadcastable z given unpacked params.

    z: (...,) broadcastable against the (...,K) params' leading dims (we add a
    trailing axis internally). Returns (...,) mixture CDF in [0,1].
    """
    zk = lib.expand_dims(z, -1) if lib.__name__ != "torch" else z.unsqueeze(-1)
    u = (zk - mu) / b
    if lib.__name__ == "torch":
        import torch as T
        # y<=mu: F = tau*exp((1-tau)*u);  y>mu: F = 1-(1-tau)*exp(-tau*u)
        Fk = T.where(u <= 0,
                     tau * T.exp((1.0 - tau) * u),
                     1.0 - (1.0 - tau) * T.exp(-tau * u))
        return (w * Fk).sum(dim=-1)
    np_ = lib
    Fk = np_.where(u <= 0,
                   tau * np_.exp((1.0 - tau) * u),
                   1.0 - (1.0 - tau) * np_.exp(-tau * u))
    return (w * Fk).sum(axis=-1)


def _cmal_comp_quantile(p, mu, b, tau, lib):
    """Per-component analytic ALD quantile at level p (scalar). Used only to
    bracket the mixture bisection. Returns (...,K)."""
    if lib.__name__ == "torch":
        import torch as T
        return T.where(
            p <= tau,
            mu + (b / (1.0 - tau)) * T.log(p / tau),
            mu - (b / tau) * T.log((1.0 - p) / (1.0 - tau)),
        )
    np_ = lib
    return np_.where(
        p <= tau,
        mu + (b / (1.0 - tau)) * np_.log(p / tau),
        mu - (b / tau) * np_.log((1.0 - p) / (1.0 - tau)),
    )


def cmal_quantiles(params, levels, iters=40, lib=None):
    """Analytic mixture quantiles via vectorized bisection in z-space.

    A mixture-of-ALD CDF is strictly increasing but not analytically
    invertible, so we bisect. Bracket [lo, hi] from the per-component analytic
    quantiles (the mixture quantile at level p lies between the min and max of
    the component quantiles at that level), pad slightly, then ~40 bisection
    steps -> ~1e-12 relative interval on a unit-ish z-range.

    params: (B, H, 4K). levels: iterable of probabilities in (0,1).
    Returns array (B, H, len(levels)) of z-space quantiles. lib defaults to
    numpy (serve); pass torch for the torch path.
    """
    if lib is None:
        lib = np
    K = params.shape[-1] // 4
    w, mu, b, tau = _cmal_unpack(params, K, lib)
    is_torch = lib.__name__ == "torch"
    outs = []
    for p in levels:
        if is_torch:
            import torch as T
            pv = T.as_tensor(float(p), dtype=mu.dtype, device=mu.device)
            comp_q = _cmal_comp_quantile(pv, mu, b, tau, lib)  # (B,H,K)
            lo = comp_q.min(dim=-1).values - 1.0
            hi = comp_q.max(dim=-1).values + 1.0
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                f = _cmal_cdf(mid, w, mu, b, tau, lib)
                go_right = f < pv
                lo = T.where(go_right, mid, lo)
                hi = T.where(go_right, hi, mid)
            outs.append(0.5 * (lo + hi))
        else:
            np_ = lib
            pv = float(p)
            comp_q = _cmal_comp_quantile(pv, mu, b, tau, lib)  # (B,H,K)
            lo = comp_q.min(axis=-1) - 1.0
            hi = comp_q.max(axis=-1) + 1.0
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                f = _cmal_cdf(mid, w, mu, b, tau, lib)
                go_right = f < pv
                lo = np_.where(go_right, mid, lo)
                hi = np_.where(go_right, hi, mid)
            outs.append(0.5 * (lo + hi))
    if is_torch:
        import torch as T
        return T.stack(outs, dim=-1)
    return np.stack(outs, axis=-1)


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


def q_norm_stats(q_cfs: np.ndarray, transform: str = "asinh") -> Optional[tuple[float, float]]:
    """Per-station discharge normalization stats (asinh- or linear-space,
    per the checkpoint's q_transform)."""
    v = np.clip(q_cfs[np.isfinite(q_cfs)], 0.0, None)
    if transform == "asinh":
        v = np.asinh(v)
    if len(v) < 180:
        return None
    sd = float(np.std(v))
    if sd < 1e-6:
        return None
    return float(np.mean(v)), sd


def _try_load() -> bool:
    global _models, _cfg, _load_failed
    if _models is not None or _load_failed:
        return _models is not None
    paths = _ckpt_paths()
    if not all(p.exists() for p in paths):
        _load_failed = True
        return False
    try:
        import torch
        models, cfg = [], None
        for ckpt in paths:
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            # All ensemble members must share the input/normalization recipe;
            # cfg (vars + stats) is deterministic given the corpus, so the
            # first checkpoint's copy speaks for all.
            cfg = cfg or payload["cfg"]
            model = build_model(payload["cfg"])
            model.load_state_dict(payload["state_dict"])
            model.eval()
            models.append(model)
        _models, _cfg = models, cfg
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
        q_tf = cfg.get("q_transform", "asinh")
        stats = q_norm_stats(q["q_cfs"].to_numpy(dtype=np.float64), transform=q_tf)
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

        q_asinh = np.clip(q_win, 0.0, None)
        if q_tf == "asinh":
            q_asinh = np.asinh(q_asinh)
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
        head = cfg.get("head", "quantile")

        def _denorm(z):
            v = z * sd_q + mu_q
            return np.sinh(v) if q_tf == "asinh" else v

        with torch.no_grad():
            xe = torch.from_numpy(x_enc[None, :, :])
            xd = torch.from_numpy(x_dec[None, :, :])
            # Ensemble = mean of member outputs in normalized space.
            raw = np.mean([m(xe, xd).squeeze(0).numpy() for m in _models], axis=0)

        if head == "cmal":
            # Analytic quantiles + distribution mean from the mixture, all in
            # z-space, then the usual asinh denorm. The mean (above the median
            # on a right-skew) is the peak-aware served point estimate.
            levels = [0.1, 0.5, 0.9]
            zq = cmal_quantiles(raw, levels)          # (H, 3)
            zq = np.sort(zq, axis=1)                   # guard tiny bisection slack
            z_mean = cmal_mean(raw)                    # (H,)
            qcfs_bands = np.clip(_denorm(zq), 0.0, None)   # (H,3)
            qcfs_mean = np.clip(_denorm(z_mean), 0.0, None)  # (H,)
            # Lay out as [q_lo, q_pt, q_hi] so the shared clamp/denorm tail
            # below operates uniformly; col indices match the quantile path.
            # The true distribution median rides along as a 4th column so it
            # gets the same caps — served as q_med (proper-scoring slot for
            # CRPS; the mean stays the peak-aware point).
            q_cfs = np.stack([qcfs_bands[:, 0], qcfs_mean, qcfs_bands[:, 2],
                              qcfs_bands[:, 1]], axis=1)
        else:
            # Denormalize: normalized (asinh or linear) → cfs. Enforce
            # quantile ordering.
            yq = np.sort(raw, axis=1)
            q_cfs = np.clip(_denorm(yq), 0.0, None)
        # Physical sanity cap: a forecast can't exceed a wide margin over the
        # gauge's own observed record. On intermittent/ephemeral streams (long
        # runs of zero flow, rare flash spikes) the asinh→sinh denormalization
        # can otherwise blow a small normalized error into millions of cfs.
        # 3x the historical max leaves headroom for a genuine record flood
        # while bounding the artifact (these were the 3 NSE=−millions outliers).
        q_obs_max = float(np.nanmax(q["q_cfs"].to_numpy(dtype=np.float64)))
        if np.isfinite(q_obs_max) and q_obs_max > 0:
            q_cfs = np.clip(q_cfs, 0.0, 3.0 * q_obs_max)
        if not np.all(np.isfinite(q_cfs)):
            return None

        # q_cfs columns are [lo, mid, hi(, median)]: for the quantile head
        # mid=q50, for the cmal head mid=distribution mean (peak-aware) and
        # col 3 carries the true median. med_i indexes the mid column.
        if head == "cmal":
            lo_i, med_i, hi_i, true_med_i = 0, 1, 2, 3
        else:
            lo_i, med_i, hi_i = 0, len(cfg["quantiles"]) // 2, len(cfg["quantiles"]) - 1
            true_med_i = med_i
        # The pinball MEDIAN systematically under-predicts peaks (baseline FHV
        # -51%: 99% of gauges under-call high flow). Flow is right-skewed, so a
        # peak-aware point estimate should lean above the median. RW2_MBLSTM_POINT
        # picks the served point forecast:
        #   default        quantile head -> q50 (legacy); cmal head -> mixture
        #                  mean (already in the mid column, sits above median)
        #   median         force q50 / mid column
        #   mean3          mean(q10,mid,q90) — pulls up on right-skew
        #   blend{w}       (1-w)*mid + w*q90, w in [0,1], peak-weighted
        # Lo/hi bands always remain q10/q90.
        point = os.environ.get("RW2_MBLSTM_POINT", "median" if head != "cmal" else "mean")
        if point == "mean3":
            q_pt = q_cfs[:, [lo_i, med_i, hi_i]].mean(axis=1)
        elif point.startswith("blend"):
            try:
                w = float(point[5:]) if len(point) > 5 else 0.3
            except ValueError:
                w = 0.3
            w = min(max(w, 0.0), 1.0)
            q_pt = (1.0 - w) * q_cfs[:, med_i] + w * q_cfs[:, hi_i]
        else:
            # "median", "mean" (cmal), or any unknown -> the mid column, which
            # is q50 for quantile and the mixture mean for cmal.
            q_pt = q_cfs[:, med_i]
        rows = []
        for i in range(horizon):
            d = (last_date + pd.Timedelta(days=i + 1)).date()
            row = {
                "date": d.isoformat(),
                "q_cfs": float(q_pt[i]),
                "q_lo": float(q_cfs[i, lo_i]),
                "q_hi": float(q_cfs[i, hi_i]),
                # True 0.5-quantile regardless of point policy: the proper
                # value for the 0.5 slot of quantile-based CRPS.
                "q_med": float(q_cfs[i, true_med_i]),
            }
            if head == "cmal":
                row["q_mean"] = float(q_cfs[i, med_i])
            rows.append(row)
        return rows
    except Exception:
        return None
