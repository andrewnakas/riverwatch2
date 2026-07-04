"""Shared streamflow-forecast evaluation metrics (SOTA-standard hydrology suite).

Pure numpy, no torch, no I/O — imported by every backtest so MB-LSTM, NWM, and
the blend are scored with identical math. Each metric takes aligned 1-D arrays
`obs`, `sim` (the caller filters NaNs / aligns lengths) and returns a float,
returning NaN — never ±inf — on degenerate input (too few points, zero
variance) so cross-station aggregates stay clean.

Metric references:
  NSE      Nash & Sutcliffe 1970
  log-NSE  NSE on log(q+eps); low-flow skill (Pushpalatha et al. 2012)
  KGE      Gupta et al. 2009 — the hydrology primary metric
  FHV/FLV  high/low-flow bias (Yilmaz et al. 2008)
  CRPS     here an APPROXIMATION from discrete quantiles (see crps_from_quantiles)
"""
from __future__ import annotations

import numpy as np

MIN_N = 20          # below this a per-station metric is not trustworthy → NaN
VAR_FLOOR = 1e-3    # NSE/KGE denominator floor; flat-flow gauges → NaN


def _clean(obs, sim):
    """Aligned finite pairs as float64, or (None, None) if too few."""
    o = np.asarray(obs, dtype=np.float64)
    s = np.asarray(sim, dtype=np.float64)
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    if len(o) < MIN_N:
        return None, None
    return o, s


def nse(obs, sim) -> float:
    o, s = _clean(obs, sim)
    if o is None:
        return float("nan")
    denom = float(np.var(o))
    if denom < VAR_FLOOR:  # undefined for a near-constant series
        return float("nan")
    return float(1.0 - np.mean((o - s) ** 2) / denom)


def log_nse(obs, sim) -> float:
    """NSE in log space; emphasizes low-flow skill. eps = 1% of mean obs
    (Pushpalatha 2012) keeps zeros finite without dominating the transform."""
    o, s = _clean(obs, sim)
    if o is None:
        return float("nan")
    mo = float(np.mean(o))
    if mo <= 0:
        return float("nan")
    eps = 0.01 * mo
    lo = np.log(np.clip(o, 0, None) + eps)
    ls = np.log(np.clip(s, 0, None) + eps)
    denom = float(np.var(lo))
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.mean((lo - ls) ** 2) / denom)


def pearson_r(obs, sim) -> float:
    o, s = _clean(obs, sim)
    if o is None or np.std(o) < 1e-12 or np.std(s) < 1e-12:
        return float("nan")
    return float(np.corrcoef(o, s)[0, 1])


def kge_components(obs, sim):
    """(KGE, r, alpha, beta). alpha = sd_sim/sd_obs (variability ratio),
    beta = mean_sim/mean_obs (bias ratio). KGE = 1 - sqrt((r-1)^2+(a-1)^2+(b-1)^2)."""
    o, s = _clean(obs, sim)
    if o is None:
        return (float("nan"),) * 4
    sd_o, mu_o = float(np.std(o)), float(np.mean(o))
    if sd_o < VAR_FLOOR ** 0.5 or abs(mu_o) < 1e-9:
        return (float("nan"),) * 4
    r = pearson_r(o, s)
    alpha = float(np.std(s)) / sd_o
    beta = float(np.mean(s)) / mu_o
    if not np.isfinite(r):
        return (float("nan"), r, alpha, beta)
    kge = 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return (kge, r, alpha, beta)


def kge(obs, sim) -> float:
    return kge_components(obs, sim)[0]


def pct_bias(obs, sim) -> float:
    """100 * sum(sim - obs) / sum(obs). Positive = overprediction."""
    o, s = _clean(obs, sim)
    if o is None:
        return float("nan")
    tot = float(np.sum(o))
    if abs(tot) < 1e-9:
        return float("nan")
    return float(100.0 * np.sum(s - o) / tot)


def fhv(obs, sim, h: float = 0.02) -> float:
    """High-flow-volume bias (%) on the top `h` fraction of observed flows
    (Yilmaz 2008). 100 * sum(sim_top - obs_top) / sum(obs_top), ranked by obs."""
    o, s = _clean(obs, sim)
    if o is None:
        return float("nan")
    k = max(1, int(np.ceil(h * len(o))))
    idx = np.argsort(o)[-k:]
    tot = float(np.sum(o[idx]))
    if abs(tot) < 1e-9:
        return float("nan")
    return float(100.0 * np.sum(s[idx] - o[idx]) / tot)


def flv(obs, sim, l: float = 0.30) -> float:
    """Low-flow-volume bias (%) on the bottom `l` fraction of observed flows
    (Yilmaz 2008), log-space slope form. Negative = model too dry at low flow."""
    o, s = _clean(obs, sim)
    if o is None:
        return float("nan")
    k = max(2, int(np.ceil(l * len(o))))
    idx = np.argsort(o)[:k]
    eps = 1e-6
    lo = np.log(np.clip(o[idx], 0, None) + eps)
    ls = np.log(np.clip(s[idx], 0, None) + eps)
    lo_min, ls_min = lo.min(), ls.min()
    obs_vol = float(np.sum(lo - lo_min))
    sim_vol = float(np.sum(ls - ls_min))
    if abs(obs_vol) < 1e-9:
        return float("nan")
    return float(100.0 * (sim_vol - obs_vol) / obs_vol)


def crps_from_quantiles(obs, levels, qvals) -> float:
    """APPROX-CRPS from discrete predictive quantiles, NOT integrated CRPS.

    Returns the mean pinball (quantile) loss over the available levels:
        (1/K) * sum_k  mean_t  max(tau_k*(y-q_k), (tau_k-1)*(y-q_k))
    This is a proper, monotone-in-skill discretization that → CRPS as the number
    of levels → ∞ (Gneiting & Raftery 2007). With only 3 levels (0.1/0.5/0.9) it
    is a lower-resolution stand-in, valid for INTERNAL A/B between our own
    models, and must not be compared directly to a paper's integrated CRPS.

    obs:    (T,) observations
    levels: (K,) quantile probabilities, e.g. [0.1, 0.5, 0.9]
    qvals:  (K, T) predicted quantile values aligned to obs
    """
    o = np.asarray(obs, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    qvals = np.asarray(qvals, dtype=np.float64)
    fin = np.isfinite(o)
    if fin.sum() < MIN_N:
        return float("nan")
    o = o[fin]
    qvals = qvals[:, fin]
    losses = []
    for tau, q in zip(levels, qvals):
        e = o - q
        losses.append(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))
    return float(np.mean(losses))


def tercile_masks(obs):
    """Boolean (low, mid, high) masks splitting obs into flow terciles, so any
    metric can be reported stratified (guards against big-river-only wins)."""
    o = np.asarray(obs, dtype=np.float64)
    fin = np.isfinite(o)
    out = {"low": np.zeros_like(fin), "mid": np.zeros_like(fin), "high": np.zeros_like(fin)}
    if fin.sum() < 3 * MIN_N:
        return out  # too few to stratify meaningfully
    vals = o[fin]
    lo_c, hi_c = np.quantile(vals, [1 / 3, 2 / 3])
    out["low"][fin] = o[fin] <= lo_c
    out["high"][fin] = o[fin] > hi_c
    out["mid"][fin] = (o[fin] > lo_c) & (o[fin] <= hi_c)
    return out


# Metric names that aggregate() summarizes (CRPS handled separately by callers
# that have quantiles).
POINT_METRICS = ("nse", "log_nse", "kge", "pearson_r", "pct_bias", "fhv", "flv")
_FNS = {"nse": nse, "log_nse": log_nse, "kge": kge, "pearson_r": pearson_r,
        "pct_bias": pct_bias, "fhv": fhv, "flv": flv}


def all_point_metrics(obs, sim) -> dict:
    """Every point metric for one station's pooled (obs, sim)."""
    return {name: _FNS[name](obs, sim) for name in POINT_METRICS}


def aggregate(per_station: dict) -> dict:
    """Median/mean across stations for each metric, ignoring NaN, plus a
    scorable count and frac_nse>0.5. `per_station` maps sid -> {metric: value}."""
    out: dict = {}
    names = set()
    for d in per_station.values():
        names.update(d.keys())
    for name in names:
        vals = np.asarray([d.get(name, np.nan) for d in per_station.values()], dtype=float)
        fin = vals[np.isfinite(vals)]
        out[name] = {
            "median": float(np.median(fin)) if len(fin) else float("nan"),
            "mean": float(np.mean(fin)) if len(fin) else float("nan"),
            "scorable": int(len(fin)),
        }
    if "nse" in out:
        nses = np.asarray([d.get("nse", np.nan) for d in per_station.values()], dtype=float)
        fin = nses[np.isfinite(nses)]
        out["nse"]["frac_gt_0.5"] = float(np.mean(fin > 0.5)) if len(fin) else float("nan")
    return out
