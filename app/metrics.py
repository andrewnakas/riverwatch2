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
  flood P/R/F1  event skill at return-period thresholds (Nearing et al. 2024)

The flood-event family (annual_maxima / return_period_thresholds /
flood_event_scores) has a different shape from POINT_METRICS — thresholds and
event matching instead of a scalar over (obs, sim) — so it is exported but not
registered in _FNS/aggregate.
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


# ------------------------------------------------- flood events (Nearing 2024)

def annual_maxima(values, year, min_days: int = 300):
    """Per-year maxima of a daily series, for return-period fitting.

    values, year: aligned 1-D arrays (year = integer calendar/water year per
    day). Years with < min_days finite days are dropped — a gappy year's
    maximum is biased low and poisons the extreme-value fit.
    Returns a float array (possibly empty), sorted by year."""
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(year)
    out = []
    for yr in np.unique(y):
        vals = v[y == yr]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= min_days:
            out.append(float(vals.max()))
    return np.asarray(out, dtype=np.float64)


def return_period_thresholds(ann_max, years=(1.0, 2.0, 5.0, 10.0),
                             min_years: int = 10) -> dict:
    """{T: discharge threshold} from annual maxima, GEV fit with empirical
    fallback.

    The T-year threshold is the F = exp(-1/T) quantile of the annual-maximum
    distribution (Langbein's annual-exceedance convention: defined at T = 1,
    where 1 - 1/T degenerates, and → 1 - 1/T for large T). GEV is fitted with
    scipy.stats.genextreme; if scipy is missing, the fit fails, or it returns
    non-finite thresholds, the empirical exp(-1/T) quantile of the annual
    maxima is used instead. Fewer than min_years maxima → all NaN (return
    periods extrapolated from short records are noise, and the 10-yr level
    needs ≥ 10 years to mean anything)."""
    am = np.asarray(ann_max, dtype=np.float64)
    am = am[np.isfinite(am)]
    probs = {float(t): float(np.exp(-1.0 / float(t))) for t in years}
    if len(am) < min_years:
        return {t: float("nan") for t in probs}
    thr = None
    try:
        from scipy.stats import genextreme
        shape, loc, scale = genextreme.fit(am)
        fitted = {t: float(genextreme.ppf(p, shape, loc=loc, scale=scale))
                  for t, p in probs.items()}
        if all(np.isfinite(list(fitted.values()))):
            thr = fitted
    except Exception:
        pass
    if thr is None:
        thr = {t: float(np.quantile(am, p)) for t, p in probs.items()}
    return thr


def _event_starts(values, threshold) -> np.ndarray:
    """Indices where a contiguous above-threshold run begins. NaNs count as
    below threshold, so a NaN gap inside one hydrological event splits it in
    two — callers should score on serially complete daily series."""
    v = np.asarray(values, dtype=np.float64)
    above = np.zeros(len(v), dtype=bool)
    fin = np.isfinite(v)
    above[fin] = v[fin] > threshold
    if not above.any():
        return np.asarray([], dtype=int)
    d = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    if above[0]:
        starts = np.concatenate([[0], starts])
    return starts


def flood_event_scores(obs, sim, obs_thr: float, sim_thr: float,
                       window_days: int = 2) -> dict:
    """Event precision/recall/F1 at one return-period threshold pair
    (Nearing et al. 2024 Nature protocol).

    An event is the start of a contiguous above-threshold run. A simulated
    event is a true positive when an observed event starts within
    ±window_days (their headline uses 2; window_days=0 is the same-day
    variant their critics report, ~half the F1 — publish both). Thresholds
    come in separately for obs and sim because the protocol computes them
    per series — the model is scored on ITS OWN flood frequency, so a
    biased-but-sharp model isn't spuriously penalized (or credited).

    obs, sim: aligned daily arrays, same calendar. Returns precision/recall/
    f1 (NaN when that side has no events; f1 = 0.0 when both sides have
    events but nothing matches) plus n_obs_events / n_sim_events."""
    if not (np.isfinite(obs_thr) and np.isfinite(sim_thr)):
        return {"precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "n_obs_events": 0, "n_sim_events": 0}
    o_starts = _event_starts(obs, obs_thr)
    s_starts = _event_starts(sim, sim_thr)
    out = {"n_obs_events": int(len(o_starts)), "n_sim_events": int(len(s_starts))}
    if len(o_starts) == 0 or len(s_starts) == 0:
        out["recall"] = 0.0 if len(o_starts) and not len(s_starts) else float("nan")
        out["precision"] = 0.0 if len(s_starts) and not len(o_starts) else float("nan")
        out["f1"] = float("nan")
        return out
    dist = np.abs(o_starts[:, None] - s_starts[None, :])
    recall = float(np.mean(dist.min(axis=1) <= window_days))
    precision = float(np.mean(dist.min(axis=0) <= window_days))
    f1 = (0.0 if precision + recall == 0
          else 2.0 * precision * recall / (precision + recall))
    out.update({"precision": precision, "recall": recall, "f1": f1})
    return out
