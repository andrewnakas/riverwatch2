#!/usr/bin/env python3
"""Phase 4.3: 2026-window blend backtest — MB-LSTM vs the NWM members.

Extends the honest NWM-archive panel (scripts/backtest_nwm_residual.py) to
the MB-LSTM member and two blends, scored on the common cohort where BOTH
sides have inputs: stations in the NWM archive AND in corpus_openmeteo
(365-day encoder history through 2026 issue dates; the old corpus/ ends
2025-11 and is unusable here), with archived 2026 decoder forcings.

Members, per (station, issue-date, horizon) against observed USGS flow:

  persistence     observation at issuance held flat (sanity floor)
  nwm_corrected   q_cfs_raw x bias_scale reconstructed from trailing h=1
                  forecast-vs-obs error (same rules as the NWM backtest)
  nwm_residual    the production v1/v2/v3 pickles combined exactly like
                  app.nwm_residual.apply_residual (log-space delta mean).
                  Manifest trained_at 2026-06-10: issuances before that
                  are inside its training window — leak-tainted in its
                  favor; the MB-LSTM checkpoints never saw 2026 targets.
  mblstm          frozen 4-seed GFS-fine-tuned ensemble through the
                  serving entry point app.mblstm.forecast — median point
                  policy, NO anchoring (measured: anchoring hurts this
                  member) — decoder forcing = archived GFS-2026 with the
                  HRRR-2026 overlay on leads 1-2 where present
  blend_invmae2   inverse-MAE^2 weighted blend of the four members above.
                  Weights: leave-one-issue-date-out per-(station, horizon)
                  MAE within the scored panel, falling back to LOO global
                  per-horizon MAE when a cell has < MIN_CELL_WINDOWS
                  scored issue dates. Diagnostic weighting — LOO prevents
                  self-selection but is not a strictly causal trailing
                  scheme (adjacent issue dates share the flow regime).
  blend_mean2     plain arithmetic mean of mblstm + nwm_residual

Observation/NWM loading is reused from backtest_nwm_residual.py via
importlib (scripts/ is not a package; same pattern as tests/test_mblstm).
The decoder-forcing helpers are re-implemented here for the fixed
gfs2026 + hrrr2026 composition rather than importing backtest_mblstm's
general ForcingPlan. Stations are processed one at a time (corpus read,
forecast, discard) so peak memory stays flat on an 8GB machine.

Output: benchmarks/blend_2026_panel.json + printed table. Median
per-station MAE is the headline, matching the NWM backtest presentation.
Missing truth (USGS provisional-data lag) is handled by scoring only
finite observations.

Usage:
  RW2_ENABLE_MBLSTM=1 .venv/bin/python scripts/backtest_blend_2026.py \
      --archive-dir /path/to/nwm-archive/archive --stride-days 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# torch and lightgbm each bundle their own libomp; loading both into one
# process deadlocks on macOS (observed: torch's copy_ hung forever in
# __kmp_join_barrier waiting on the OTHER runtime's workers). With a single
# OpenMP thread neither runtime forks a worker team, so no cross-runtime
# barrier exists — and it keeps this run polite on the shared 8GB box.
# Must be set before numpy/torch/lightgbm initialize.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RW2_ENABLE_MBLSTM", "1")

# Reuse the NWM backtest's archive/obs/panel machinery verbatim.
_spec = importlib.util.spec_from_file_location(
    "backtest_nwm_residual", Path(__file__).resolve().parent / "backtest_nwm_residual.py")
nwm_bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nwm_bt)

from app import gages2  # noqa: E402
from app import mblstm  # noqa: E402
from app.weather import DAILY_VARS  # noqa: E402

CORPUS_DIR = ROOT / "data" / "mblstm" / "corpus_openmeteo"
GFS_2026_DIR = ROOT / "data" / "mblstm" / "gfs_fcst_2026"
HRRR_2026_DIR = ROOT / "data" / "mblstm" / "hrrr_fcst_2026"
MODELS_DIR = ROOT / "data" / "nwm_residual_models"
STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
OUT_DIR = ROOT / "benchmarks"
HORIZON = 14

# The frozen 4-seed GFS-fine-tuned ensemble (production checkpoints).
DEFAULT_CKPT = ":".join(
    str(ROOT / "data" / "mblstm" / f"model_h256_s{s}_gfsft.pt")
    for s in (101, 102, 103, 104))

GFS_VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "shortwave_radiation_sum"]

# Blend members feeding the inverse-MAE^2 weights (order = column order).
BLEND_MEMBERS = ["mblstm", "nwm_corrected", "nwm_residual", "persistence"]
MIN_CELL_WINDOWS = 6   # LOO per-(station,h) needs >= 5 other issue dates
MAE_FLOOR_CFS = 1e-3   # weight floor: a 0-MAE cell must not absorb the blend


# ------------------------------------------------------------ observations

def load_observations_cohort(stations: list[str], start: date, end: date,
                             *, refresh: bool) -> pd.DataFrame:
    """Cohort-only USGS daily values, cached separately from the full-panel
    cache (that one is keyed by dates alone; reusing it for a subset fetch
    would silently poison full backtest runs)."""
    nwm_bt.OBS_CACHE.mkdir(parents=True, exist_ok=True)
    cache = nwm_bt.OBS_CACHE / (
        f"dv_blend_{start.isoformat()}_{end.isoformat()}_{len(stations)}st.csv.gz")
    if cache.exists() and not refresh:
        return pd.read_csv(cache, dtype={"station_id": str})
    rows: list[tuple[str, str, float]] = []
    chunks = [stations[i:i + 100] for i in range(0, len(stations), 100)]
    for i, chunk in enumerate(chunks, 1):
        got = nwm_bt._fetch_dv_chunk(chunk, start, end)
        for sid, series in got.items():
            rows.extend((sid, d, q) for d, q in series.items())
        print(f"  obs fetch [{i}/{len(chunks)}] +{len(got)} stations ({len(rows):,} rows)")
        time.sleep(0.3)
    obs = pd.DataFrame(rows, columns=["station_id", "date", "q_cfs"])
    obs.to_csv(cache, index=False, compression="gzip")
    print(f"  cached → {cache}")
    return obs


# --------------------------------------------------------- decoder forcing

def load_fcst_inits(src_dir: Path, cohort: set[str]) -> dict[date, pd.DataFrame]:
    """init_date -> station-indexed forecast frame, cohort rows only.

    src_dir is an SD-card symlink: Path.glob follows it, and "._*" exFAT
    AppleDouble junk is skipped (same guards as backtest_mblstm.load_gfs).
    A 00z init on day D has lead_day 1 = calendar day D, so the matching
    issue date is t0 = D - 1."""
    out: dict[date, pd.DataFrame] = {}
    for p in sorted(src_dir.glob("*.csv.gz")):
        if p.name.startswith("._"):
            continue
        df = pd.read_csv(p, dtype={"station_id": str})
        df = df[df["station_id"].isin(cohort)]
        if len(df):
            out[date.fromisoformat(p.name.split(".")[0])] = df.set_index("station_id")
    return out


def build_wx_fcst(sid: str, t0: date,
                  gfs: dict[date, pd.DataFrame],
                  hrrr: dict[date, pd.DataFrame]) -> pd.DataFrame | None:
    """14-day decoder forcing for one window: GFS leads 1-14 (must be
    complete, else None) with HRRR overwriting leads 1-2 where present
    (overlay only — HRRR has no coverage everywhere, e.g. Alaska)."""
    init = t0 + timedelta(days=1)
    gdf = gfs.get(init)
    if gdf is None or sid not in gdf.index:
        return None
    arr = np.full((HORIZON, len(GFS_VARS)), np.nan)
    grows = gdf.loc[[sid]]
    grows = grows[(grows["lead_day"] >= 1) & (grows["lead_day"] <= HORIZON)]
    # Parity with the frozen plan "gfs:1-14,hrrr?:1-2": the GFS base must
    # fully cover the horizon on its own (the overlay never patches gaps).
    if len(set(grows["lead_day"].astype(int))) < HORIZON:
        return None
    for _, r in grows.iterrows():
        arr[int(r["lead_day"]) - 1] = r[GFS_VARS].to_numpy(dtype=float)
    hdf = hrrr.get(init)
    if hdf is not None and sid in hdf.index:
        hrows = hdf.loc[[sid]]
        hrows = hrows[(hrows["lead_day"] >= 1) & (hrows["lead_day"] <= 2)]
        for _, r in hrows.iterrows():
            arr[int(r["lead_day"]) - 1] = r[GFS_VARS].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        return None
    out = pd.DataFrame(arr, columns=GFS_VARS)
    out.insert(0, "date", [pd.Timestamp(t0) + pd.Timedelta(days=d)
                           for d in range(1, HORIZON + 1)])
    return out


# ----------------------------------------------------------------- mblstm

def mblstm_station_preds(corpus_path: Path, attrs: dict, t0_list: list[date],
                         gfs: dict[date, pd.DataFrame],
                         hrrr: dict[date, pd.DataFrame],
                         skips: dict[str, int]) -> list[tuple[str, date, int, float]]:
    """Run the serving entry point over every issue date for one station.

    Mirrors backtest_mblstm.eval_station's input construction: corpus rows
    daily-reindexed, history strictly <= t0, and the window is skipped when
    the corpus lacks a finite q_cfs AT t0 (USGS provisional lag / trickle
    fetch not caught up) — mblstm.forecast anchors its calendar on the last
    observation, so a stale tail would silently misalign the horizons."""
    sid = corpus_path.name.split(".")[0]
    df = pd.read_csv(corpus_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    daily = df.set_index("date").reindex(
        pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D"))

    preds: list[tuple[str, date, int, float]] = []
    for t0 in t0_list:
        ts0 = pd.Timestamp(t0)
        if ts0 not in daily.index:
            skips["corpus_no_t0"] += 1
            continue
        hist = daily.loc[:ts0]
        if len(hist) < 400:
            skips["short_hist"] += 1
            continue
        if pd.isna(hist["q_cfs"].iloc[-1]):
            skips["corpus_no_t0"] += 1
            continue
        wx_fcst = build_wx_fcst(sid, t0, gfs, hrrr)
        if wx_fcst is None:
            skips["no_forcing"] += 1
            continue
        q_hist = hist["q_cfs"].dropna().rename("q_cfs").reset_index()
        q_hist.columns = ["date", "q_cfs"]
        wx_hist = hist.reset_index().rename(columns={"index": "date"})[["date"] + DAILY_VARS]
        rows = mblstm.forecast(q_hist, wx_hist, wx_fcst, attrs, HORIZON)
        if not rows or len(rows) < HORIZON:
            skips["forecast_none"] += 1
            continue
        # r["q_cfs"] is the served point = q50 under the forced median policy.
        preds.extend((sid, t0, h, float(rows[h - 1]["q_cfs"]))
                     for h in range(1, HORIZON + 1))
    return preds


# ------------------------------------------------------------ NWM residual

def predict_residual_prod(panel: pd.DataFrame) -> np.ndarray:
    """Production nwm_residual on the panel: v1/v2/v3 pickles, deltas
    averaged in log space — algebraically identical to
    app.nwm_residual.apply_residual's mean-of-deltas combination.

    MUST run before the first mblstm.forecast call: unpickling loads
    lightgbm, and on macOS its libomp segfaults if torch's copy is already
    resident. Production is safe by the same ordering (app.forecast imports
    lightgbm at module load, torch lazily later)."""
    import warnings
    logs = []
    for v in ("v1", "v2", "v3"):
        models = nwm_bt.load_prod_models(MODELS_DIR / v)
        if models:
            with warnings.catch_warnings():
                # predict_residual feeds bare ndarrays, like production does.
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                logs.append(np.log1p(nwm_bt.predict_residual(panel, models)))
    if not logs:
        raise SystemExit(f"no production residual pickles under {MODELS_DIR}/v*")
    return np.expm1(np.mean(logs, axis=0))


# ------------------------------------------------------------------ blends

def blend_inv_mae2(df: pd.DataFrame) -> np.ndarray:
    """Inverse-MAE^2 blend of BLEND_MEMBERS. Per-row member weight
    1/max(MAE, floor)^2 where MAE is the leave-one-issue-date-out mean
    absolute error of that member in the row's (station, horizon) cell;
    cells with < MIN_CELL_WINDOWS issue dates fall back to the LOO global
    per-horizon MAE (all stations, same-issue-date rows excluded)."""
    obs = df["q_obs"].to_numpy(dtype=np.float64)
    keys_cell = [df["station_id"], df["horizon_day"]]
    keys_hd = [df["horizon_day"], df["issued_date"]]
    n_cell = df.groupby(keys_cell)["q_obs"].transform("size").to_numpy()
    n_h = df.groupby(df["horizon_day"])["q_obs"].transform("size").to_numpy()
    n_hd = df.groupby(keys_hd)["q_obs"].transform("size").to_numpy()
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for m in BLEND_MEMBERS:
        pred = df[m].to_numpy(dtype=np.float64)
        err = pd.Series(np.abs(pred - obs), index=df.index)
        # LOO per-(station, horizon): each issue date is exactly one row.
        loo_cell = (err.groupby(keys_cell).transform("sum").to_numpy()
                    - err.to_numpy()) / np.maximum(n_cell - 1, 1)
        # LOO global per-horizon: drop every row sharing the issue date.
        loo_glob = (err.groupby(df["horizon_day"]).transform("sum").to_numpy()
                    - err.groupby(keys_hd).transform("sum").to_numpy()
                    ) / np.maximum(n_h - n_hd, 1)
        mae = np.where(n_cell >= MIN_CELL_WINDOWS, loo_cell, loo_glob)
        w = 1.0 / np.maximum(mae, MAE_FLOOR_CFS) ** 2
        num += w * pred
        den += w
    return num / np.maximum(den, 1e-12)


# ---------------------------------------------------------------- scoring

def score_members(df: pd.DataFrame, members: list[str]) -> dict:
    """Per horizon, per member: pooled MAE, per-station median/mean MAE,
    win-rate vs nwm_corrected — the NWM backtest's shapes, minus terciles
    (the cohort is too small to stratify meaningfully yet)."""
    obs = df["q_obs"].to_numpy(dtype=np.float64)
    for m in members:
        df[f"err_{m}"] = np.abs(df[m].to_numpy(dtype=np.float64) - obs)
    out: dict = {}
    for h in sorted(df["horizon_day"].unique()):
        sub = df[df["horizon_day"] == h]
        per_station = sub.groupby("station_id")[[f"err_{m}" for m in members]].mean()
        hrow: dict = {"n_pairs": int(len(sub)),
                      "n_stations": int(len(per_station)), "members": {}}
        for m in members:
            col = f"err_{m}"
            entry = {
                "mae_pooled": float(sub[col].mean()),
                "mae_station_median": float(per_station[col].median()),
                "mae_station_mean": float(per_station[col].mean()),
            }
            if m != "nwm_corrected":
                wins = per_station[col] < per_station["err_nwm_corrected"]
                entry["win_rate_vs_corrected"] = float(wins.mean())
            hrow["members"][m] = entry
        out[str(int(h))] = hrow
    return out


def print_table(scores: dict, members: list[str]) -> None:
    print(f"\n{'h':>3} {'n':>6} {'st':>4}  " + "".join(f"{m:>16}" for m in members))
    print(" " * 16 + "".join(f"{'med-MAE (win%)':>16}" for _ in members))
    for h, row in sorted(scores.items(), key=lambda kv: int(kv[0])):
        line = f"{h:>3} {row['n_pairs']:>6} {row['n_stations']:>4}  "
        for m in members:
            e = row["members"][m]
            win = e.get("win_rate_vs_corrected")
            tag = f"{e['mae_station_median']:>9.1f}"
            tag += f" ({100 * win:3.0f}%)" if win is not None else "       "
            line += f"{tag:>16}"
        print(line)


# ------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--stride-days", type=int, default=3,
                    help="evaluate every Nth panel issue date (torch on CPU: "
                         "keeps ~120 stations x ~90 issue dates tractable)")
    ap.add_argument("--limit-stations", type=int, default=0)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="colon-separated MB-LSTM checkpoint list "
                         "(default: the frozen 4-seed gfsft ensemble)")
    ap.add_argument("--label", default="panel")
    ap.add_argument("--refresh-obs", action="store_true")
    args = ap.parse_args()

    os.environ["RW2_MBLSTM_CKPT_PATH"] = args.ckpt
    os.environ["RW2_MBLSTM_POINT"] = "median"  # measured policy for this member

    t_start = time.time()
    arch = nwm_bt.load_archive(Path(args.archive_dir))

    # Cohort: NWM panel stations that also have 2026-capable encoder history.
    corpus_files = sorted(p for p in CORPUS_DIR.glob("*.csv.gz")
                          if not p.name.startswith("._"))
    corpus_sids = {p.name.split(".")[0] for p in corpus_files}
    arch_sids = set(arch["station_id"].unique())
    cohort = sorted(corpus_sids & arch_sids)
    if args.limit_stations:
        cohort = cohort[: args.limit_stations]
    print(f"cohort: {len(cohort)} stations "
          f"(corpus_openmeteo {len(corpus_sids)} ∩ archive {len(arch_sids)})")
    if not cohort:
        raise SystemExit("empty cohort — is corpus_openmeteo mounted?")
    arch = arch[arch["station_id"].isin(cohort)].reset_index(drop=True)

    issue_dates = sorted(arch["issued_date"].unique())
    obs_start = issue_dates[0] - timedelta(days=nwm_bt.BIAS_LOOKBACK_DAYS + 2)
    obs = load_observations_cohort(cohort, obs_start, date.today(),
                                   refresh=args.refresh_obs)
    print(f"observations: {len(obs):,} rows, {obs['station_id'].nunique()} stations")

    bias = nwm_bt.reconstruct_bias_scale(arch, obs)
    panel = nwm_bt.add_trailing_residual(
        nwm_bt.add_features(nwm_bt.build_panel(arch, obs, bias)))
    panel = panel.dropna(subset=["q_obs_t0"]).reset_index(drop=True)
    print(f"labeled panel: {len(panel):,}/{len(arch):,} rows, "
          f"issued {issue_dates[0]} → {issue_dates[-1]} ({len(issue_dates)} days)")

    # nwm_residual first — lightgbm must load before torch (see docstring).
    panel["nwm_residual"] = predict_residual_prod(panel)

    # Issue-date grid: panel days that have a next-day GFS-2026 init, strided.
    gfs = load_fcst_inits(GFS_2026_DIR, set(cohort))
    hrrr = load_fcst_inits(HRRR_2026_DIR, set(cohort))
    t0_grid = [D for D in issue_dates if (D + timedelta(days=1)) in gfs]
    t0_grid = t0_grid[:: max(1, args.stride_days)]
    print(f"forcing: {len(gfs)} GFS / {len(hrrr)} HRRR 2026 inits → "
          f"{len(t0_grid)} strided issue dates "
          f"({t0_grid[0]} → {t0_grid[-1]}, stride {args.stride_days}d)")

    # MB-LSTM, one station at a time (bounded memory; torch loads lazily on
    # the first forecast call).
    registry = {s["id"]: s for s in json.loads(STATIONS_PATH.read_text())["stations"]}
    skips = {"corpus_no_t0": 0, "short_hist": 0, "no_forcing": 0, "forecast_none": 0}
    mb_rows: list[tuple[str, date, int, float]] = []
    cohort_set = set(cohort)
    files = [p for p in corpus_files if p.name.split(".")[0] in cohort_set]
    for i, p in enumerate(files, 1):
        sid = p.name.split(".")[0]
        attrs = gages2.enrich_station_attrs(dict(registry.get(sid, {"id": sid})))
        try:
            mb_rows.extend(mblstm_station_preds(p, attrs, t0_grid, gfs, hrrr, skips))
        except Exception as exc:
            print(f"[{i}/{len(files)}] {sid} ERR {exc}", flush=True)
        if i % 10 == 0:
            print(f"[{i}/{len(files)}] {len(mb_rows):,} mblstm rows "
                  f"({time.time() - t_start:.0f}s)", flush=True)
    if not mb_rows:
        raise SystemExit(f"no mblstm forecasts produced (skips={skips})")
    mb = pd.DataFrame(mb_rows, columns=["station_id", "issued_date",
                                        "horizon_day", "mblstm"])
    print(f"mblstm: {len(mb):,} rows, {mb['station_id'].nunique()} stations, "
          f"skipped windows: {skips}")

    # Common scored cohort: rows with a finite obs (build_panel already
    # dropped missing truth), obs-at-issuance, and an mblstm forecast.
    df = panel.merge(mb, on=["station_id", "issued_date", "horizon_day"], how="inner")
    df = df.rename(columns={"q_obs_t0": "persistence", "q_corrected": "nwm_corrected"})
    df["blend_invmae2"] = blend_inv_mae2(df)
    df["blend_mean2"] = 0.5 * (df["mblstm"] + df["nwm_residual"])

    members = ["persistence", "nwm_corrected", "nwm_residual", "mblstm",
               "blend_invmae2", "blend_mean2"]
    scores = score_members(df, members)
    n_st = int(df["station_id"].nunique())
    print(f"\nscored cohort: {n_st} stations, "
          f"{df['issued_date'].nunique()} issue dates, {len(df):,} pairs")
    if n_st < 30:
        print(f"NOTE: cohort under 30 stations — low statistical power; "
              f"medians will move as the corpus_openmeteo trickle fills in.")
    print_table(scores, members)

    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "label": args.label,
        "ran_at": pd.Timestamp.utcnow().isoformat(),
        "stride_days": args.stride_days,
        "ckpt": args.ckpt,
        "point_policy": "median",
        "anchoring": "none (measured: anchoring hurts the mblstm member)",
        "decoder_forcing": "gfs_fcst_2026 leads 1-14 + hrrr_fcst_2026 overlay leads 1-2",
        "blend_weighting": (
            "inverse-MAE^2 over {mblstm, nwm_corrected, nwm_residual, persistence}; "
            "MAE = leave-one-issue-date-out per-(station,horizon) within the scored "
            f"panel, fallback to LOO global per-horizon when cell has < {MIN_CELL_WINDOWS} "
            "issue dates. Diagnostic, not strictly causal-trailing."),
        "cohort": {
            "n_stations": n_st,
            "n_issue_dates": int(df["issued_date"].nunique()),
            "n_pairs": int(len(df)),
            "candidate_stations": len(cohort),
        },
        "skipped_windows": skips,
        "members": members,
        "horizons": scores,
        "caveats": [
            "nwm_residual production pickles trained through 2026-06-10 — panel "
            "issuances before that date are inside their training window "
            "(leak-tainted in their favor); mblstm checkpoints never saw 2026.",
            "cohort limited to corpus_openmeteo stations (~120 of 1893 panel "
            "stations so far; a background trickle adds more daily).",
            "truth = USGS daily values; windows/targets without finite obs are "
            "skipped (provisional-data lag).",
        ],
    }
    out = OUT_DIR / "blend_2026_panel.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  ({time.time() - t_start:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
