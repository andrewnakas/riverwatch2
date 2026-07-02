#!/usr/bin/env python3
"""Honest temporal backtest for the v16 MB-LSTM member.

Evaluates the trained checkpoint through the *serving* code path
(app.mblstm.forecast) on issue dates the model never saw in training
(default: calendar 2025 = the validation year; 2026 windows can be used once
corpus rows exist for them). For every (station, issue-date) window:

    q_hist / wx_hist : corpus rows  <= t0   (what serving would fetch)
    wx_fcst          : corpus rows in (t0, t0+H]  — observed weather as the
                       "forecast" (perfect-forcing; generous at h>3, same
                       caveat as the trainer and stated in the output)
    truth            : observed q at t0+1 .. t0+H

Reports per-horizon median-station MAE for persistence vs the MB-LSTM median
quantile, the MAE ratio, and per-station pooled NSE — same shape as
benchmarks/nwm_backtest_v4.json so the members can be compared side by side.

Decoder-forcing modes (mutually exclusive):
  default            observed future weather (perfect forcing)
  --gfs [--hrrr]     archived deterministic forecasts (ForcingPlan sugar)
  --forcing-plan     explicit per-lead composition, e.g. 'ecmwf:1-14,hrrr?:1-2'
  --members-source   ensemble forcing: one forecast per NWP member from
                     <init>.members.csv.gz, quantiles pooled across members
                     (see pool_member_quantiles)

Usage:
  RW2_ENABLE_MBLSTM=1 .venv/bin/python scripts/backtest_mblstm.py \
      --ckpt data/mblstm/model.pt --label pilot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RW2_ENABLE_MBLSTM", "1")

from app import gages2  # noqa: E402
from app import mblstm  # noqa: E402
from app import metrics  # noqa: E402
from app.weather import DAILY_VARS  # noqa: E402

CORPUS_DIR = ROOT / "data" / "mblstm" / "corpus"
STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
CAMELS_PATH = ROOT / "data" / "camels_gauge_ids.json"
SCHEMA_VERSION = 2  # v2: full SOTA metric suite + CAMELS subset + approx-CRPS
GFS_DIR = ROOT / "data" / "mblstm" / "gfs_fcst"
HRRR_DIR = ROOT / "data" / "mblstm" / "hrrr_fcst"
OUT_DIR = ROOT / "benchmarks"
HORIZON = 14
MIN_MEMBERS = 6  # ensemble-forcing mode: skip a window with fewer usable members

# Named forcing sources for --forcing-plan (all share the fetcher csv schema).
SRC_DIRS = {
    "gfs": GFS_DIR,
    "hrrr": HRRR_DIR,
    "gefs": ROOT / "data" / "mblstm" / "gefs_fcst",
    "ecmwf": ROOT / "data" / "mblstm" / "ecmwf_fcst",
    "gfs2026": ROOT / "data" / "mblstm" / "gfs_fcst_2026",
    "hrrr2026": ROOT / "data" / "mblstm" / "hrrr_fcst_2026",
}

GFS_VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "shortwave_radiation_sum"]


def load_gfs(start: str, end: str, src_dir: Path = GFS_DIR) -> dict[pd.Timestamp, pd.DataFrame]:
    """init_date -> per-station forecast frame. A 00z init on day D has
    lead_day 1 = calendar day D, so the matching issue date is t0 = D-1
    (observations through yesterday, today's 00z run — the serving setup)."""
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for p in sorted(src_dir.glob("*.csv.gz")):
        # Skip exFAT AppleDouble junk (._*) and per-member companion files
        # (<init>.members.csv.gz) — both match the glob but aren't ens-mean
        # forcing files; the members file would silently corrupt the lookup.
        if p.name.startswith("._") or ".members." in p.name:
            continue
        init = pd.Timestamp(p.name.split(".")[0])
        # t0 = init-1d must fall inside the eval window.
        if not (pd.Timestamp(start) <= init - pd.Timedelta(days=1) <= pd.Timestamp(end)):
            continue
        df = pd.read_csv(p, dtype={"station_id": str})
        out[init] = df.set_index("station_id")
    return out


def load_members(src_dir: Path, start: str, end: str) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    """init_date -> {station_id: float32 array (n_members, HORIZON, n_vars)}.

    Reads the per-member companion archives (<init>.members.csv.gz — the files
    load_gfs deliberately skips). Members missing any of the 14 lead days or
    containing non-finite values are dropped per station at load time, so the
    caller only ever sees complete member forcings. Values are stored as
    compact float32 arrays (~5 MB per init for the full corpus) so a season of
    inits stays resident on an 8 GB box alongside a live training run.
    """
    out: dict[pd.Timestamp, dict[str, np.ndarray]] = {}
    for p in sorted(src_dir.glob("*.members.csv.gz")):
        if p.name.startswith("._"):  # exFAT AppleDouble junk on the SD volume
            continue
        init = pd.Timestamp(p.name.split(".")[0])
        # Same t0 = init-1d window rule as load_gfs.
        if not (pd.Timestamp(start) <= init - pd.Timedelta(days=1) <= pd.Timestamp(end)):
            continue
        df = pd.read_csv(p, dtype={"station_id": str})
        df = df[df["lead_day"].between(1, HORIZON)]
        df = df.drop_duplicates(["station_id", "member", "lead_day"])
        df = df.sort_values(["station_id", "member", "lead_day"])
        # Keep only (station, member) groups with all 14 lead days present.
        complete = df.groupby(["station_id", "member"])["lead_day"].transform("size") == HORIZON
        df = df[complete]
        stations: dict[str, np.ndarray] = {}
        for sid, g in df.groupby("station_id", sort=False):
            # Rows are sorted (member, lead_day) with exactly HORIZON rows per
            # member, so the reshape yields one (HORIZON, n_vars) slab each.
            a = g[GFS_VARS].to_numpy(dtype=np.float32).reshape(-1, HORIZON, len(GFS_VARS))
            a = a[np.isfinite(a).all(axis=(1, 2))]
            if len(a):
                stations[sid] = a
        if stations:
            out[init] = stations
    return out


def pool_member_quantiles(member_lo, member_med, member_hi):
    """Pool per-member quantile triplets into an ensemble band + point.

    Approximation (pre-registered for the phase-4.1 gate): each member's
    (q_lo, q_med, q_hi) is treated as three samples at the 0.1/0.5/0.9 levels
    of that member's predictive distribution. Pooling the 3*M values per
    horizon and taking the empirical 10th/50th/90th percentiles approximates
    those quantiles of the equal-weight mixture across members. With only 3
    samples per member the mixture tails are under-resolved, so the pooled
    band tends to be conservative (narrow) relative to the true mixture —
    i.e. a PICP gain measured this way is not an artifact of the pooling.
    Degenerate case (M >= 2): if all members agree exactly, the pooled triplet
    equals the common member triplet — no artificial widening.

    Point forecast = median across members of q_med.

    Args: three array-likes shaped (M, H) — members x horizons.
    Returns (pooled_lo, pooled_med, pooled_hi, point), each shape (H,).
    """
    lo = np.asarray(member_lo, dtype=float)
    med = np.asarray(member_med, dtype=float)
    hi = np.asarray(member_hi, dtype=float)
    pool = np.concatenate([lo, med, hi], axis=0)  # (3M, H)
    p_lo, p_med, p_hi = np.nanpercentile(pool, [10.0, 50.0, 90.0], axis=0)
    point = np.nanmedian(med, axis=0)
    return p_lo, p_med, p_hi, point


class MemberForcing:
    """Ensemble-forcing mode: one decoder forcing per NWP ensemble member.

    Loads the <init>.members.csv.gz companion archives for one source (ecmwf /
    gefs) and serves, per (station, issue date), the list of complete member
    wx_fcst frames. Returns None (window skipped) when fewer than MIN_MEMBERS
    members are usable for that station/init."""

    def __init__(self, source: str, start: str, end: str):
        self.source = source
        self.inits = load_members(SRC_DIRS[source], start, end)
        if not self.inits:
            raise SystemExit(f"no {source} member files in window — fetch first")

    def issue_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(i - pd.Timedelta(days=1) for i in self.inits))

    def member_wx_fcsts(self, sid: str, t0: pd.Timestamp) -> list[pd.DataFrame] | None:
        arr = self.inits.get(t0 + pd.Timedelta(days=1), {}).get(sid)
        if arr is None or len(arr) < MIN_MEMBERS:
            return None
        dates = [t0 + pd.Timedelta(days=d) for d in range(1, HORIZON + 1)]
        frames = []
        for a in arr:
            f = pd.DataFrame(a.astype(float), columns=GFS_VARS)
            f.insert(0, "date", dates)
            frames.append(f)
        return frames


class ForcingPlan:
    """Per-lead decoder-forcing composition from archived forecast sources.

    Spec: comma-separated segments "<source>:<lo>-<hi>". Later segments
    overwrite earlier ones inside their lead range. A trailing "?" on the
    source name makes the segment an overlay (used where present, silently
    skipped where absent — e.g. HRRR has no Alaska coverage); non-optional
    segments must fully cover their range or the window is skipped.

      gfs:1-14                the plain GFS baseline
      gfs:1-14,hrrr?:1-2      the frozen-baseline hybrid (--gfs --hrrr)
      ecmwf:1-14,hrrr?:1-2    ECMWF ens-mean base with HRRR sharpening
    """

    def __init__(self, spec: str, start: str, end: str):
        self.spec = spec
        self.segments: list[tuple[str, int, int, bool]] = []
        for part in spec.split(","):
            name, rng = part.split(":")
            optional = name.endswith("?")
            name = name.rstrip("?")
            if name not in SRC_DIRS:
                raise SystemExit(f"unknown forcing source {name!r} (have {sorted(SRC_DIRS)})")
            lo, hi = (rng.split("-") if "-" in rng else (rng, rng))
            self.segments.append((name, int(lo), int(hi), optional))
        self.sources: dict[str, dict] = {}
        for name, *_ in self.segments:
            if name not in self.sources:
                self.sources[name] = load_gfs(start, end, src_dir=SRC_DIRS[name])
                if not self.sources[name] and not name.startswith("hrrr"):
                    raise SystemExit(f"no {name} init files in window — fetch first")

    def issue_dates(self) -> pd.DatetimeIndex:
        """t0 grid from the base (first) source's inits, like GFS mode did."""
        base = self.sources[self.segments[0][0]]
        return pd.DatetimeIndex(sorted(i - pd.Timedelta(days=1) for i in base))

    def wx_fcst(self, sid: str, t0: pd.Timestamp) -> pd.DataFrame | None:
        init = t0 + pd.Timedelta(days=1)
        arr = np.full((HORIZON, len(GFS_VARS)), np.nan)
        for name, lo, hi, optional in self.segments:
            df = self.sources[name].get(init)
            rows = None
            if df is not None:
                try:
                    rows = df.loc[[sid]] if sid in df.index else None
                except (KeyError, TypeError):
                    rows = None
            if rows is None:
                if optional:
                    continue
                return None
            rr = rows[(rows["lead_day"] >= lo) & (rows["lead_day"] <= hi)]
            present = set(rr["lead_day"].astype(int))
            if not optional and any(d not in present for d in range(lo, hi + 1)):
                return None
            for _, r in rr.iterrows():
                arr[int(r["lead_day"]) - 1] = r[GFS_VARS].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            return None
        out = pd.DataFrame(arr, columns=GFS_VARS)
        out.insert(0, "date", [t0 + pd.Timedelta(days=d) for d in range(1, HORIZON + 1)])
        return out


def load_camels_ids(which: str) -> set[str]:
    """CAMELS-US gauge ids for the requested subset (671 = full, 531 = the
    'well-behaved' subset that published median-NSE ~0.76 is usually quoted on).
    Returns an empty set if the data file is absent — the harness still runs;
    the CAMELS metric block is simply omitted."""
    if which == "none" or not CAMELS_PATH.exists():
        return set()
    data = json.loads(CAMELS_PATH.read_text())
    ids = set(data["671"]) if which == "671" else set(data.get("531", data["671"]))
    # Normalize to the corpus's zero-padded 8-digit form.
    return {str(s).strip().zfill(8) for s in ids}


def _metric_block(results: dict, sids: list[str]) -> dict:
    """Median/mean across the given stations for every per-station metric."""
    sub = {sid: results[sid] for sid in sids if sid in results}
    if not sub:
        return {}
    keys = ("nse", "kge", "log_nse", "pearson_r", "pct_bias", "fhv", "flv",
            "approx_crps", "picp90", "mpiw_norm")
    per_station = {sid: {k: r[k] for k in keys if k in r} for sid, r in sub.items()}
    agg = metrics.aggregate(per_station)
    agg["n_stations"] = len(sub)
    return agg


def anchor(y: np.ndarray, y_h1: float, q_obs_t0: float, decay_h: float) -> np.ndarray:
    """As-served anchor-to-observed correction (same formula the production
    blend applies per member in app/forecast.py): shift the trajectory by the
    h1 gap to the last observation, decaying linearly to zero by h=1+decay_h.
    decay_h <= 0 disables. Applied as a translation, so bands shift with the
    point."""
    if decay_h <= 0 or not np.isfinite(y_h1) or not np.isfinite(q_obs_t0):
        return y
    hs = np.arange(1, len(y) + 1, dtype=float)
    w = np.clip(1.0 - (hs - 1.0) / float(decay_h), 0.0, 1.0)
    return y + (q_obs_t0 - y_h1) * w


def eval_station(path: Path, attrs: dict, issue_dates: pd.DatetimeIndex,
                 forcing: "ForcingPlan | None" = None,
                 members: "MemberForcing | None" = None,
                 anchor_decay: float = 0.0,
                 min_windows: int = 5,
                 dump: list | None = None) -> dict | None:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    by_date = df.set_index("date")
    daily = by_date.reindex(pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D"))

    per_h: dict[int, dict[str, list]] = {h: {"persist": [], "mblstm": []} for h in range(1, HORIZON + 1)}
    pooled_y, pooled_yhat = [], []
    pooled_lo, pooled_hi = [], []  # model's own 0.1/0.9 quantiles for approx-CRPS
    pooled_med = []                # true q50 (CRPS 0.5 slot; == point unless CMAL/blend policy)
    n_windows = 0
    member_counts: list[int] = []  # usable members per window (ensemble mode)
    sid = path.name.split(".")[0]
    for t0 in issue_dates:
        if t0 not in daily.index:
            continue
        hist = daily.loc[:t0]
        fut = daily.loc[t0 + pd.Timedelta(days=1): t0 + pd.Timedelta(days=HORIZON)]
        if len(fut) < HORIZON or len(hist) < 400:
            continue
        q0 = hist["q_cfs"].dropna()
        if q0.empty or pd.isna(hist["q_cfs"].iloc[-1]):
            continue
        truth = fut["q_cfs"].to_numpy(dtype=float)
        if np.isfinite(truth).sum() < 7:
            continue

        q_hist = hist["q_cfs"].dropna().rename("q_cfs").reset_index()
        q_hist.columns = ["date", "q_cfs"]
        wx_hist = hist.reset_index().rename(columns={"index": "date"})[["date"] + DAILY_VARS]
        if members is not None:
            # Ensemble-forcing mode: one forecast per NWP member, quantile
            # triplets pooled across members (see pool_member_quantiles).
            fcsts = members.member_wx_fcsts(sid, t0)
            if fcsts is None:
                continue
            m_lo, m_med, m_hi = [], [], []
            for wx_m in fcsts:
                rows = mblstm.forecast(q_hist, wx_hist, wx_m, attrs, HORIZON)
                if not rows or len(rows) < HORIZON:
                    continue
                m_lo.append([r.get("q_lo", np.nan) for r in rows])
                m_med.append([r.get("q_med", r["q_cfs"]) for r in rows])
                m_hi.append([r.get("q_hi", np.nan) for r in rows])
            if len(m_med) < MIN_MEMBERS:
                continue
            ylo, ymed, yhi, yhat = pool_member_quantiles(m_lo, m_med, m_hi)
            ymean = np.full(HORIZON, np.nan)  # no pooled mean; nan like non-CMAL runs
            member_counts.append(len(m_med))
        else:
            if forcing is not None:
                wx_fcst = forcing.wx_fcst(sid, t0)
                if wx_fcst is None:
                    continue
            else:
                wx_fcst = fut.reset_index().rename(columns={"index": "date"})[["date"] + DAILY_VARS]

            rows = mblstm.forecast(q_hist, wx_hist, wx_fcst, attrs, HORIZON)
            if not rows or len(rows) < HORIZON:
                continue
            yhat = np.asarray([r["q_cfs"] for r in rows], dtype=float)
            ylo = np.asarray([r.get("q_lo", np.nan) for r in rows], dtype=float)
            yhi = np.asarray([r.get("q_hi", np.nan) for r in rows], dtype=float)
            ymed = np.asarray([r.get("q_med", r["q_cfs"]) for r in rows], dtype=float)
            ymean = np.asarray([r.get("q_mean", np.nan) for r in rows], dtype=float)
        persist = float(q_hist["q_cfs"].iloc[-1])

        if dump is not None:
            dump.append(pd.DataFrame({
                "station_id": sid, "t0": t0.date().isoformat(),
                "h": np.arange(1, HORIZON + 1, dtype=np.int8),
                "truth": truth.astype(np.float32),
                "ylo": ylo.astype(np.float32), "ymed": ymed.astype(np.float32),
                "yhi": yhi.astype(np.float32), "ymean": ymean.astype(np.float32),
                "persist": np.float32(persist),
            }))
        if anchor_decay > 0:
            off_ref = float(yhat[0])
            yhat = anchor(yhat, off_ref, persist, anchor_decay)
            ylo = anchor(ylo, off_ref, persist, anchor_decay)
            yhi = anchor(yhi, off_ref, persist, anchor_decay)
            ymed = anchor(ymed, off_ref, persist, anchor_decay)
            yhat = np.clip(yhat, 0.0, None); ymed = np.clip(ymed, 0.0, None)
            ylo = np.clip(ylo, 0.0, None); yhi = np.clip(yhi, 0.0, None)

        n_windows += 1
        for h in range(1, HORIZON + 1):
            yt = truth[h - 1]
            if not np.isfinite(yt):
                continue
            per_h[h]["persist"].append(abs(yt - persist))
            per_h[h]["mblstm"].append(abs(yt - yhat[h - 1]))
            pooled_y.append(yt)
            pooled_yhat.append(yhat[h - 1])
            pooled_lo.append(ylo[h - 1])
            pooled_hi.append(yhi[h - 1])
            pooled_med.append(ymed[h - 1])

    if n_windows < max(min_windows, 1):
        return None
    pooled_y = np.asarray(pooled_y); pooled_yhat = np.asarray(pooled_yhat)
    pooled_lo = np.asarray(pooled_lo); pooled_hi = np.asarray(pooled_hi)
    pooled_med = np.asarray(pooled_med)
    # Full SOTA metric suite via the shared app.metrics module. metrics.nse
    # reproduces the historical var<1e-3 → NaN guard, so the headline NSE is
    # byte-stable with pre-metrics-module backtest JSONs.
    m = metrics.all_point_metrics(pooled_y, pooled_yhat)
    # approx-CRPS from the model's own quantiles (mean pinball over 0.1/0.5/0.9).
    # The 0.5 slot takes the TRUE median (q_med), not the served point — with a
    # CMAL mean or blended point the old code scored the wrong functional.
    crps = metrics.crps_from_quantiles(
        pooled_y, [0.1, 0.5, 0.9], np.vstack([pooled_lo, pooled_med, pooled_hi]))
    # 90% interval coverage (PICP) and mean width (MPIW, normalized by mean obs).
    cov = float(np.mean((pooled_y >= pooled_lo) & (pooled_y <= pooled_hi))) \
        if np.isfinite(pooled_lo).any() else float("nan")
    mpiw = float(np.mean(pooled_hi - pooled_lo) / max(np.mean(pooled_y), 1e-9)) \
        if np.isfinite(pooled_lo).any() else float("nan")
    # Tercile-stratified NSE/KGE/log-NSE to rule out big-river-only wins.
    tm = metrics.tercile_masks(pooled_y)
    by_tercile = {
        band: {
            "nse": metrics.nse(pooled_y[mask], pooled_yhat[mask]),
            "kge": metrics.kge(pooled_y[mask], pooled_yhat[mask]),
            "log_nse": metrics.log_nse(pooled_y[mask], pooled_yhat[mask]),
        }
        for band, mask in tm.items() if mask.any()
    }
    out = {
        "windows": n_windows,
        "nse": m["nse"],  # primary headline metric, unchanged definition
        "approx_crps": crps, "picp90": cov, "mpiw_norm": mpiw,
        "by_tercile": by_tercile,
        "mae_by_h": {h: {k: float(np.mean(v)) for k, v in d.items() if v} for h, d in per_h.items()},
    }
    out.update({k: v for k, v in m.items() if k != "nse"})  # kge, log_nse, r, pbias, fhv, flv
    if member_counts:
        out["members_mean"] = float(np.mean(member_counts))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "data" / "mblstm" / "model.pt"))
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--limit-stations", type=int, default=0)
    ap.add_argument("--stride-stations", type=int, default=1,
                    help="evaluate every Nth corpus station (fast A/B subsample)")
    ap.add_argument("--label", default="pilot")
    ap.add_argument("--gfs", action="store_true",
                    help="decoder forcing from archived GFS forecasts instead of "
                         "observed future weather (issue dates snap to GFS inits)")
    ap.add_argument("--hrrr", action="store_true",
                    help="with --gfs: overlay 3km HRRR on decoder lead days 1-2")
    ap.add_argument("--camels-subset", choices=["none", "671", "531"], default="none",
                    help="also report metrics on the CAMELS-US basin subset for "
                         "apples-to-apples comparison to published median-NSE")
    ap.add_argument("--forcing-plan", default="",
                    help="per-lead decoder forcing composition, e.g. "
                         "'ecmwf:1-14,hrrr?:1-2' (see ForcingPlan; overrides "
                         "--gfs/--hrrr sugar)")
    ap.add_argument("--members-source", choices=["ecmwf", "gefs"], default="",
                    help="ensemble-forcing mode: one forecast per NWP ensemble "
                         "member from <init>.members.csv.gz, quantile triplets "
                         "pooled across members (see pool_member_quantiles). "
                         "Mutually exclusive with --forcing-plan/--gfs; issue "
                         "dates snap to available member-file inits")
    ap.add_argument("--min-windows", type=int, default=5,
                    help="minimum evaluated windows for a station to score "
                         "(default 5; lower for smoke tests on the still-sparse "
                         "member archives)")
    ap.add_argument("--anchor-decay", type=float, default=0.0,
                    help="as-served anchoring: shift trajectory by the h1 gap to "
                         "the last observation, decaying to zero by h=1+N "
                         "(production uses decay_h=2; 0 = off/unanchored)")
    ap.add_argument("--point", default="",
                    help="sets RW2_MBLSTM_POINT for the run "
                         "(median | mean3 | blend0.2 | ...)")
    ap.add_argument("--dump-windows", default="",
                    help="write per-window raw predictions (unanchored "
                         "lo/med/hi/mean + truth + persistence) to this csv.gz "
                         "path for offline anchor/point-policy sweeps")
    args = ap.parse_args()

    os.environ["RW2_MBLSTM_CKPT_PATH"] = args.ckpt
    if args.point:
        os.environ["RW2_MBLSTM_POINT"] = args.point
    if args.hrrr and not args.gfs:
        print("--hrrr requires --gfs")
        return 1
    if args.members_source and (args.forcing_plan or args.gfs or args.hrrr):
        print("--members-source is mutually exclusive with --forcing-plan/--gfs/--hrrr")
        return 1
    # --gfs/--hrrr are sugar for the equivalent forcing plan.
    plan_spec = args.forcing_plan
    if not plan_spec and args.gfs:
        plan_spec = "gfs:1-14,hrrr?:1-2" if args.hrrr else "gfs:1-14"
    forcing = None
    members = None
    if args.members_source:
        members = MemberForcing(args.members_source, args.start, args.end)
        issue_dates = members.issue_dates()
        print(f"members mode ({args.members_source}): {len(issue_dates)} issue dates "
              f"{issue_dates[0].date()}..{issue_dates[-1].date()}")
    elif plan_spec:
        forcing = ForcingPlan(plan_spec, args.start, args.end)
        issue_dates = forcing.issue_dates()
        print(f"forcing plan '{plan_spec}': {len(issue_dates)} issue dates "
              f"{issue_dates[0].date()}..{issue_dates[-1].date()}")
    else:
        issue_dates = pd.date_range(args.start, args.end, freq=f"{args.stride}D")

    registry = {s["id"]: s for s in json.loads(STATIONS_PATH.read_text())["stations"]}
    files = sorted(p for p in CORPUS_DIR.glob("*.csv.gz")
                   if not p.name.startswith("._"))
    if args.stride_stations > 1:
        files = files[:: args.stride_stations]  # deterministic subsample for fast A/B
    if args.limit_stations:
        files = files[: args.limit_stations]

    results: dict[str, dict] = {}
    dump: list | None = [] if args.dump_windows else None
    t0 = time.time()
    for i, p in enumerate(files, 1):
        sid = p.name.split(".")[0]
        attrs = gages2.enrich_station_attrs(dict(registry.get(sid, {"id": sid})))
        try:
            r = eval_station(p, attrs, issue_dates, forcing=forcing,
                             members=members, anchor_decay=args.anchor_decay,
                             min_windows=args.min_windows, dump=dump)
        except Exception as exc:
            print(f"[{i}/{len(files)}] {sid} ERR {exc}", flush=True)
            continue
        if r is None:
            continue
        results[sid] = r
        if i % 50 == 0:
            print(f"[{i}/{len(files)}] {len(results)} stations evaluated "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if dump:
        dump_path = Path(args.dump_windows)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(dump, ignore_index=True).to_csv(
            dump_path, index=False, compression="gzip")
        print(f"wrote {len(dump)} windows -> {dump_path}", flush=True)

    if not results:
        print("no stations evaluated — is the checkpoint present and corpus built?")
        return 1

    # Median-across-stations MAE per horizon + ratio, matching the NWM
    # backtest's presentation.
    print(f"\nstations={len(results)}  windows/station~="
          f"{np.median([r['windows'] for r in results.values()]):.0f}  "
          f"window: {args.start}..{args.end} stride {args.stride}d")
    print(f"{'h':>3} {'persist':>9} {'mblstm':>9} {'ratio':>7}")
    summary_h = {}
    for h in range(1, HORIZON + 1):
        pm = [r["mae_by_h"][h]["persist"] for r in results.values() if h in r["mae_by_h"] and "persist" in r["mae_by_h"][h]]
        mm = [r["mae_by_h"][h]["mblstm"] for r in results.values() if h in r["mae_by_h"] and "mblstm" in r["mae_by_h"][h]]
        med_p, med_m = float(np.median(pm)), float(np.median(mm))
        summary_h[h] = {"persistence": med_p, "mblstm": med_m,
                        "ratio": med_m / med_p if med_p > 0 else None}
        print(f"{h:>3} {med_p:>9.1f} {med_m:>9.1f} {med_m / med_p:>7.3f}")
    nses = np.asarray([r["nse"] for r in results.values()], dtype=float)
    scorable = nses[np.isfinite(nses)]  # drop the not-NSE-scorable flat-flow gauges
    print(f"\npooled-horizon NSE (cfs): median={np.nanmedian(scorable):.3f}  "
          f"mean={np.nanmean(scorable):.3f}  frac>0.5={np.mean(scorable > 0.5):.2f}  "
          f"(scorable {len(scorable)}/{len(nses)})")

    # Full SOTA metric suite, full corpus and (optionally) CAMELS subset.
    full_block = _metric_block(results, list(results.keys()))
    metric_blocks = {"full": full_block}
    print("\nSOTA metrics (median across stations):")
    for k in ("nse", "kge", "log_nse", "pearson_r", "pct_bias", "fhv", "flv",
              "approx_crps", "picp90", "mpiw_norm"):
        if k in full_block:
            print(f"  {k:>11}: {full_block[k]['median']:+.3f}  (n={full_block[k]['scorable']})")

    camels_ids = load_camels_ids(args.camels_subset)
    if args.camels_subset != "none":
        if not camels_ids:
            print(f"\nCAMELS-{args.camels_subset}: gauge-id file {CAMELS_PATH.name} "
                  f"absent — skipping subset block (run still valid for full corpus)")
        else:
            inter = sorted(set(results) & camels_ids)
            print(f"\nCAMELS-{args.camels_subset} subset: {len(inter)} of "
                  f"{len(results)} evaluated stations are CAMELS basins")
            blk = _metric_block(results, inter)
            metric_blocks[f"camels_{args.camels_subset}"] = blk
            if blk:
                print(f"  CAMELS median NSE={blk.get('nse',{}).get('median',float('nan')):.3f}  "
                      f"KGE={blk.get('kge',{}).get('median',float('nan')):.3f}  "
                      f"(published ensembled-LSTM ref ~0.76 median NSE)")

    members_mean = [r["members_mean"] for r in results.values() if "members_mean" in r]
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"mblstm_backtest_{args.label}.json"
    out.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "label": args.label, "ckpt": args.ckpt,
        "window": [args.start, args.end], "stride_days": args.stride,
        "stride_stations": args.stride_stations,
        "anchor_decay": args.anchor_decay,
        "min_windows": args.min_windows,
        "point_policy": os.environ.get("RW2_MBLSTM_POINT", "default"),
        "forcing_plan": plan_spec or
                        (f"members:{args.members_source}" if args.members_source else "perfect"),
        "members_source": args.members_source or None,
        "members_per_window_mean": (float(np.mean(members_mean)) if members_mean else None),
        "caveat": (f"decoder forcing = archived per-member {args.members_source} "
                   "ensemble forecasts via dynamical.org; per-member quantile "
                   "triplets pooled to 10/50/90 empirical percentiles (real "
                   "forecast error + forcing spread)" if args.members_source else
                   "decoder forcing = archived HRRR d1-2 + GFS d3-14 hybrid via "
                   "dynamical.org (real forecast error)" if args.hrrr else
                   "decoder forcing = archived GFS forecasts via dynamical.org "
                   "(real forecast error)" if args.gfs else
                   f"decoder forcing = archived plan '{plan_spec}' via dynamical.org "
                   "(real forecast error)" if plan_spec else
                   "decoder forcing = observed archive weather (perfect forcing); generous at h>3"),
        "crps_caveat": "approx_crps = mean pinball over {0.1,0.5,0.9}, NOT integrated CRPS "
                       "(valid for internal A/B only; resolved when a CMAL head ships).",
        "stations": len(results),
        "median_mae_by_h": summary_h,
        "nse_median": float(np.nanmedian(scorable)), "nse_mean": float(np.nanmean(scorable)),
        "nse_scorable_stations": int(len(scorable)),
        "metrics": metric_blocks,
        "by_tercile_median": _tercile_summary(results),
        "per_station": {
            sid: {k: r.get(k) for k in
                  ("nse", "kge", "log_nse", "pct_bias", "fhv", "flv",
                   "approx_crps", "picp90", "windows", "members_mean")}
            for sid, r in results.items()
        },
    }, indent=2))
    print(f"wrote {out}")
    return 0


def _tercile_summary(results: dict) -> dict:
    """Median across stations of the per-station tercile NSE/KGE/log-NSE."""
    bands = ("low", "mid", "high")
    out: dict = {}
    for band in bands:
        for metric in ("nse", "kge", "log_nse"):
            vals = [r["by_tercile"][band][metric]
                    for r in results.values()
                    if "by_tercile" in r and band in r["by_tercile"]
                    and np.isfinite(r["by_tercile"][band].get(metric, np.nan))]
            out.setdefault(band, {})[metric] = float(np.median(vals)) if vals else float("nan")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
