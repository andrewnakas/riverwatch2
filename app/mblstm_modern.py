"""MODERN-1 research member, served: the `nldasm9` 5-seed ensemble.

WHAT THIS IS. `nldasm9` is one of the three members behind the MODERN-1 with-q
record (day-1 median NSE 0.905829 under PERFECT forcing, 490 CAMELS basins, test
2018-10-01..2025-09-30, equal-weight mean of 5 seeds). It is the ONLY member of
that roster whose inputs exist operationally: `fused4md49fix3` needs 4-product
forcing on the decoder for all 14 forecast days, and `fused4m9fix3` needs Daymet
columns -- Daymet V4R1 is an ANNUAL release. `nldasm9` is single-product, 6
variables, and those 6 names are bare (no product suffix), identical to what
`app/weather.py` already fetches. That is why the operational decay was measured
on this member and no other.

WHAT ITS NUMBERS MEAN. 0.905829 is the three-member ensemble under perfect
forcing; this member's own day-1 lead-1 figure is 0.869798. Under real
1-day-ahead GFS forcing the measured paired cost is -0.048033
(CI [-0.059808, -0.039718], benchmarks/m1_decay_day1_nldasm9.json). Nothing
served here should be presented as 0.905829.

WHY A SEPARATE MODULE. `app/mblstm.py` holds `_models`/`_cfg`/`_load_failed` as
module-level singletons, so loading a second checkpoint set through it would
clobber the production member. This module keeps its own state and reuses that
module's PURE helpers (`build_model`, `norm_wx`, `static_vector`,
`q_norm_stats`, `_doy_sincos`) so the arithmetic cannot drift. The tensor
assembly below mirrors `app/mblstm.py:forecast()` for the quantile head only --
no cmal, no dhbv, no dec_q -- which is all `nldasm9` needs.

DOMAIN GATE. The member refuses any station outside `data/m1_cohort.json`. It is
a 490-basin model; the site serves 9,851 gauges and the other 9,361 are out of
domain. A silent extrapolation is worse than a missing member.

STATICS. The 27 Addor CAMELS attributes via `app/camels_attrs.py`, NOT the 14
GAGES-II features the production member uses. Read from the checkpoint's
`static_feats`, so this is config-driven, not hardcoded.

READOUT. The record's point prediction is `(ylo + yhi) / 2`, recorded as
`"readout"` in `benchmarks/m1_FINAL_withq_corrected.json` -- not the median.
`RW2_MBLSTM_MODERN_POINT` can override it (median | midhl | mean3).

Gated by RW2_ENABLE_MBLSTM_MODERN=1. Returns None on any failure, same contract
as `app/mblstm.py` and `app/ealstm.py`, so the blend silently drops the member.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import camels_attrs
from . import mblstm as _mb

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_DAYS = 365

# 5 seeds, equal-weight mean -- the record's ship bar. Never quote a figure from
# fewer seeds than this ensemble carries.
DEFAULT_CKPTS = ":".join(
    str(ROOT / "data" / "mblstm" / "m1" / f"m1_nldasm9_s{s}.pt") for s in range(501, 506)
)

_models: list | None = None
_cfg: dict | None = None
_load_failed = False


def _is_enabled() -> bool:
    return os.environ.get("RW2_ENABLE_MBLSTM_MODERN") == "1"


def _ckpt_paths() -> list[Path]:
    p = os.environ.get("RW2_MBLSTM_MODERN_CKPT_PATH") or DEFAULT_CKPTS
    return [Path(x) for x in p.split(":") if x]


def _cfg_is_servable(cfg: dict) -> Optional[str]:
    """Refuse recipes this module does not actually implement.

    Returns a reason string, or None when the cfg is servable. These are
    checked rather than assumed because every one of them changes the input
    tensors without changing their SHAPE, so a wrong assumption would train a
    silently different model and read as a measurement. `dec_q` in particular
    appends two decoder columns that the assembly below does not build --
    `app/mblstm.py` has the same gap, which is why this is explicit here.
    """
    if cfg.get("head", "quantile") != "quantile":
        return f"head={cfg.get('head')!r}, this module serves the quantile head only"
    if cfg.get("dec_q"):
        return "dec_q=True needs two extra decoder inputs that are not assembled here"
    if cfg.get("enc_lead"):
        return f"enc_lead={cfg.get('enc_lead')!r} is not reproduced at serve time"
    if cfg.get("dec_lead"):
        return "dec_lead != 0 reads forcing beyond entitlement and is UNSERVABLE"
    if cfg.get("no_q_input"):
        return "no_q_input=True is the no-q arm; serve it through its own member"
    if int(cfg.get("context", CONTEXT_DAYS)) != CONTEXT_DAYS:
        return f"context={cfg.get('context')} != {CONTEXT_DAYS}"
    return None


# ---------------------------------------------------------------------------
# Forcing channel completeness.
#
# `nldasm9` needs 6 variables. Five of them are exactly what `app/weather.py`
# already fetches; the sixth, `vapor_pressure`, is NOT in `weather.DAILY_VARS`.
# `norm_wx()` reindexes a missing column to NaN and then `nan_to_num`s it to 0,
# which in z-space IS the training mean -- so an absent channel would be served
# silently as "average humidity forever". That is a handled failure turning into
# a silent one, so this module refuses instead, and records why.
#
# Where a dewpoint or relative-humidity channel IS available we derive vapour
# pressure from it rather than refusing (Tetens/Magnus over water, Pa, which is
# the unit the CAMELS/NLDAS corpora store):
#     e_s(T) = 611.2 * exp(17.67 * T / (T + 243.5))
#     e      = e_s(Td)           from dewpoint
#     e      = RH/100 * e_s(T)   from relative humidity
# ---------------------------------------------------------------------------
_DEWPOINT_COLS = ("dew_point_2m_mean", "dewpoint_2m_mean", "dew_point_2m")
_RH_COLS = ("relative_humidity_2m_mean", "relativehumidity_2m_mean")

last_skip_reason: Optional[str] = None


def _sat_vp_pa(t_c):
    return 611.2 * np.exp(17.67 * np.asarray(t_c, dtype=np.float64) / (np.asarray(t_c, dtype=np.float64) + 243.5))


def _ensure_vapor_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Add `vapor_pressure` (Pa) if it is absent but derivable. Never overwrites."""
    if df is None or not len(df) or "vapor_pressure" in df.columns:
        return df
    for c in _DEWPOINT_COLS:
        if c in df.columns:
            out = df.copy()
            out["vapor_pressure"] = _sat_vp_pa(out[c].to_numpy(dtype=np.float64))
            return out
    for c in _RH_COLS:
        if c in df.columns and "temperature_2m_mean" in df.columns:
            out = df.copy()
            rh = np.clip(out[c].to_numpy(dtype=np.float64), 0.0, 100.0)
            out["vapor_pressure"] = (rh / 100.0) * _sat_vp_pa(
                out["temperature_2m_mean"].to_numpy(dtype=np.float64))
            return out
    return df


def _missing_channels(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    """Columns that are absent, or present but entirely non-finite."""
    if frame is None or not len(frame):
        return list(cols)
    miss = []
    for c in cols:
        if c not in frame.columns:
            miss.append(c)
        elif not np.isfinite(pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=np.float64)).any():
            miss.append(c)
    return miss


def _try_load() -> bool:
    global _models, _cfg, _load_failed
    if _models is not None or _load_failed:
        return _models is not None
    paths = _ckpt_paths()
    if not paths or not all(p.exists() for p in paths):
        _load_failed = True
        return False
    try:
        import torch

        models, cfg = [], None
        for ckpt in paths:
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            c = payload["cfg"]
            if cfg is None:
                cfg = c
                reason = _cfg_is_servable(cfg)
                if reason:
                    raise ValueError(f"{ckpt.name}: {reason}")
            else:
                # Seeds must share the input and normalization recipe or the
                # ensemble mean is over models that saw different inputs.
                for k in ("enc_vars", "dec_vars", "static_feats", "hidden",
                          "horizon", "q_transform", "quantiles"):
                    if c.get(k) != cfg.get(k):
                        raise ValueError(f"{ckpt.name}: cfg[{k!r}] differs across seeds")
            m = _mb.build_model(cfg)
            m.load_state_dict(payload["state_dict"])
            m.eval()
            models.append(m)
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
    """Standard member entry point -- same contract as `mblstm.forecast()`.

    `static_attrs` must carry the station id under "id" or "site_no"; the
    cohort gate needs it and the contract has no separate argument for it.
    """
    if not _is_enabled() or not _try_load():
        return None
    cfg = _cfg or {}
    try:
        global last_skip_reason
        last_skip_reason = None
        attrs_in = static_attrs or {}
        sid = str(attrs_in.get("id") or attrs_in.get("site_no") or "")
        if not sid or not camels_attrs.in_cohort(sid):
            # Out of domain -> no number at all. Not recorded as a skip reason:
            # 9,361 of the 9,851 served gauges are outside the cohort by design
            # and noting it on each would be pure noise.
            return None
        if q_hist is None or len(q_hist) < CONTEXT_DAYS:
            last_skip_reason = (f"needs {CONTEXT_DAYS}d discharge history, "
                                f"have {0 if q_hist is None else len(q_hist)}")
            return None
        if wx_hist is None or len(wx_hist) < 30:
            last_skip_reason = "needs 30d weather history"
            return None

        wx_hist = _ensure_vapor_pressure(wx_hist)
        wx_fcst = _ensure_vapor_pressure(wx_fcst)
        miss_e = _missing_channels(wx_hist, cfg["enc_vars"])
        miss_d = _missing_channels(wx_fcst, cfg["dec_vars"])
        if miss_e or miss_d:
            # Refuse rather than serve a channel at the training mean.
            last_skip_reason = (
                "forcing channels missing — encoder " + (",".join(miss_e) or "none")
                + "; decoder " + (",".join(miss_d) or "none")
                + ". This member needs NLDAS-2-shaped 6-variable forcing; "
                  "app/weather.py DAILY_VARS carries no vapor_pressure and no "
                  "dewpoint/RH to derive it from.")
            return None

        q = q_hist.copy()
        q["date"] = pd.to_datetime(q["date"])
        q_tf = cfg.get("q_transform", "linear")
        stats = _mb.q_norm_stats(q["q_cfs"].to_numpy(dtype=np.float64), transform=q_tf)
        if stats is None:
            return None
        mu_q, sd_q = stats

        wx = wx_hist.copy()
        wx["date"] = pd.to_datetime(wx["date"])
        last_date = q["date"].iloc[-1]
        idx = pd.date_range(last_date - pd.Timedelta(days=CONTEXT_DAYS - 1), last_date, freq="D")
        wx_win = wx.set_index("date").reindex(idx)
        q_win = q.set_index("date")["q_cfs"].reindex(idx).to_numpy(dtype=np.float64)

        qv = np.clip(q_win, 0.0, None)
        if q_tf == "asinh":
            qv = np.asinh(qv)
        q_mask = np.isfinite(qv).astype(np.float32)
        q_n = np.nan_to_num((qv - mu_q) / sd_q, nan=0.0).astype(np.float32)

        # The 27 Addor attributes, merged over the registry record so
        # `log_drain_area` and lat/lon still resolve for any cfg that wants them.
        sv = _mb.static_vector(camels_attrs.enrich_station_attrs(sid, attrs_in), cfg)

        enc_wx = _mb.norm_wx(wx_win.reset_index(drop=True), cfg["enc_vars"], cfg)
        enc_doy = _mb._doy_sincos(pd.Series(idx))
        T = len(idx)
        x_enc = np.concatenate(
            [enc_wx, q_n[:, None], q_mask[:, None], enc_doy,
             np.repeat(sv[None, :], T, axis=0)], axis=1)

        fut_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        wf = wx_fcst.copy() if wx_fcst is not None else pd.DataFrame()
        if len(wf):
            wf["date"] = pd.to_datetime(wf["date"])
            wf = wf.set_index("date").reindex(fut_idx)
        else:
            wf = pd.DataFrame(index=fut_idx)
        dec_wx = _mb.norm_wx(wf.reset_index(drop=True), cfg["dec_vars"], cfg)
        dec_doy = _mb._doy_sincos(pd.Series(fut_idx))
        lead = (np.arange(1, horizon + 1, dtype=np.float32) / float(cfg["horizon"]))[:, None]
        x_dec = np.concatenate(
            [dec_wx, dec_doy, lead, np.repeat(sv[None, :], horizon, axis=0)], axis=1)

        import torch

        with torch.no_grad():
            xe = torch.from_numpy(x_enc[None, :, :])
            xd = torch.from_numpy(x_dec[None, :, :])
            raws = [m(xe, xd).squeeze(0).numpy() for m in _models]

        # Average seed OUTPUTS first, then sort, then clip -- the order
        # `app/mblstm.py` uses and the order the record's scorer reproduces.
        z = np.mean(np.stack(raws, axis=0), axis=0)
        q_cfs = z * sd_q + mu_q
        if q_tf == "asinh":
            q_cfs = np.sinh(q_cfs)
        q_cfs = np.sort(q_cfs, axis=1)
        q_cfs = np.clip(q_cfs, 0.0, None)

        q_obs_max = float(np.nanmax(q["q_cfs"].to_numpy(dtype=np.float64))) if len(q) else np.nan
        if np.isfinite(q_obs_max) and q_obs_max > 0:
            q_cfs = np.clip(q_cfs, 0.0, 3.0 * q_obs_max)
        if not np.all(np.isfinite(q_cfs)):
            return None

        nq = len(cfg["quantiles"])
        lo_i, med_i, hi_i = 0, nq // 2, nq - 1
        point = os.environ.get("RW2_MBLSTM_MODERN_POINT", "midhl")
        if point == "median":
            q_pt = q_cfs[:, med_i]
        elif point == "mean3":
            q_pt = q_cfs[:, [lo_i, med_i, hi_i]].mean(axis=1)
        else:
            # "midhl" -- the record's readout, (ylo + yhi) / 2.
            q_pt = 0.5 * (q_cfs[:, lo_i] + q_cfs[:, hi_i])

        rows = []
        for i in range(horizon):
            d = (last_date + pd.Timedelta(days=i + 1)).date()
            rows.append({
                "date": d.isoformat(),
                "q_cfs": float(q_pt[i]),
                "q_lo": float(q_cfs[i, lo_i]),
                "q_hi": float(q_cfs[i, hi_i]),
                "q_med": float(q_cfs[i, med_i]),
            })
        return rows
    except Exception:
        return None
