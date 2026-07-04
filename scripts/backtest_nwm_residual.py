#!/usr/bin/env python3
"""Honest temporal-holdout backtest for the NWM members (v15.2).

Closes the gap documented in BACKTEST_REPORT.md: `nwm` and `nwm_residual`
were never scored against held-out observations — `nwm_residual`'s UI MAE
was the training manifest's own validation ratio applied to `nwm`'s MAE.

This script measures, per horizon, against observed USGS flow:

  nwm_raw         q_cfs_raw straight from the archived issued forecast
  nwm_corrected   q_cfs_raw x bias_scale, where bias_scale is reconstructed
                  from data available strictly before issuance (trailing
                  h=1 forecast-vs-obs window; same clip/min-overlap rules
                  as app.nwm.hindcast_skill, which uses the
                  analysis_assimilation overlap that is not refetchable
                  historically)
  nwm_residual    the production pickles in data/nwm_residual_models/,
                  fed the REAL observation-at-issuance (production
                  behavior), on issuance dates after --train-end.
                  NOTE: those pickles saw archive data through their
                  trained_at date, so if --train-end is earlier the score
                  is leak-tainted in their favor — the clean member below
                  is the honest comparison.
  nwm_residual_clean
                  same model family retrained here on issuances
                  <= --train-end only, with honest features (real obs_t0,
                  reconstructed bias_scale), scored on the held-out slice
  persistence     observation at issuance held flat (sanity floor)

Split: by issued_date. Train <= --train-end < test. No test-period
target_date can reach training because every training row's target_date
<= train_end + 14d < first scored target only when horizons respect the
gap; we additionally drop training rows whose target_date lands inside
the test window (strict no-leak guarantee).

Observed truth: USGS daily values (00060, mean) batch-fetched for every
archive station and cached under data/cache/backtest_obs/.

Output: benchmarks/nwm_backtest_<label>.json + printed table. Median
per-station MAE is the headline (mean is dominated by a few snowmelt
giants); flow-tercile stratification guards against big-river-only wins.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import pickle
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROD_MODELS_DIR = ROOT / "data" / "nwm_residual_models"
OBS_CACHE = ROOT / "data" / "cache" / "backtest_obs"
OUT_DIR = ROOT / "benchmarks"

DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
USER_AGENT = "riverwatch2-backtest/0.1 (treesixtyweather@gmail.com)"

# Mirror app.nwm.hindcast_skill's guardrails so nwm_corrected here behaves
# like the live member: >=7 overlapping days, scale clipped to [0.5, 2.0].
BIAS_MIN_OVERLAP = 7
BIAS_LOOKBACK_DAYS = 30
BIAS_CLIP = (0.5, 2.0)

# v1 = production parity (what scripts/train_nwm_residual.py ships).
FEAT_COLS_V1 = [
    "log1p_q_nwm_raw", "log1p_q_nwm_corrected", "log1p_q_obs_t0",
    "bias_scale", "doy", "month",
]
# v2 = + anchor gap, obs recency/trend, trailing local NWM skill. All
# computable at issuance from data the build already holds.
FEAT_COLS_V2 = FEAT_COLS_V1 + [
    "d_anchor",            # log1p obs_t0 - log1p corrected: how far NWM sits from the gauge right now
    "log1p_q_obs_lag3",
    "log1p_q_obs_lag7",
    "obs_trend_3d",        # log1p obs_t0 - log1p obs(D-3): rising/falling limb
    "log1p_obs_trail30",   # trailing 30d mean flow: station scale anchor
    "trail_h1_logmae",     # trailing 30d mean |log error| of NWM h1 here: local NWM skill
]
# v3 = v2 + per-(station,horizon) trailing signed residual: the additive
# log-space analog of bias_scale, sign included, specific to this horizon.
FEAT_COLS_V3 = FEAT_COLS_V2 + ["trail_resid_h"]


# ---------------------------------------------------------------- archive

def load_archive(archive_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(archive_dir).rglob("*.csv.gz")):
        with gzip.open(p, "rt") as f:
            # dtype=str on station_id: pandas would otherwise parse
            # "01030500" as the int 1030500 and the USGS join would
            # silently drop every region-0 gauge (the trainer bug).
            frames.append(pd.read_csv(f, dtype={"station_id": str}))
    if not frames:
        raise SystemExit(f"no archive csvs under {archive_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["issued_date"] = pd.to_datetime(df["issued_date"]).dt.date
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
    df["horizon_day"] = df["horizon_day"].astype(int)
    df["q_cfs_raw"] = pd.to_numeric(df["q_cfs_raw"], errors="coerce")
    df["q_cfs_obs_today"] = pd.to_numeric(df["q_cfs_obs_today"], errors="coerce")
    df = df.dropna(subset=["q_cfs_raw"])
    df = df[df["q_cfs_raw"] >= 0]
    return df.reset_index(drop=True)


# ------------------------------------------------------------ observations

def _fetch_dv_chunk(sites: list[str], start: date, end: date) -> dict[str, dict[str, float]]:
    params = {
        "format": "json", "sites": ",".join(sites),
        "startDT": start.isoformat(), "endDT": end.isoformat(),
        "parameterCd": "00060", "statCd": "00003",
    }
    url = DV_URL + "?" + urlencode(params)
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except Exception:
            if attempt == 2:
                return {}
            time.sleep(2.0 * (attempt + 1))
    out: dict[str, dict[str, float]] = {}
    for ts in (payload.get("value", {}) or {}).get("timeSeries", []) or []:
        try:
            sid = ts["sourceInfo"]["siteCode"][0]["value"]
            vals = ts["values"][0]["value"]
        except Exception:
            continue
        rows = out.setdefault(sid, {})
        for v in vals:
            try:
                q = float(v["value"])
            except Exception:
                continue
            if q <= -999990:  # USGS missing sentinel
                continue
            rows[v["dateTime"][:10]] = q
    return out


def load_observations(stations: list[str], start: date, end: date, *, refresh: bool) -> pd.DataFrame:
    """Daily mean cfs for every station, cached as one csv.gz."""
    OBS_CACHE.mkdir(parents=True, exist_ok=True)
    cache = OBS_CACHE / f"dv_{start.isoformat()}_{end.isoformat()}.csv.gz"
    if cache.exists() and not refresh:
        obs = pd.read_csv(cache, dtype={"station_id": str})
        return obs
    rows: list[tuple[str, str, float]] = []
    chunks = [stations[i:i + 100] for i in range(0, len(stations), 100)]
    for i, chunk in enumerate(chunks, 1):
        got = _fetch_dv_chunk(chunk, start, end)
        for sid, series in got.items():
            rows.extend((sid, d, q) for d, q in series.items())
        print(f"  obs fetch [{i}/{len(chunks)}] +{len(got)} stations ({len(rows):,} rows)")
        time.sleep(0.3)
    obs = pd.DataFrame(rows, columns=["station_id", "date", "q_cfs"])
    obs.to_csv(cache, index=False, compression="gzip")
    print(f"  cached → {cache}")
    return obs


# ------------------------------------------------------------- bias scale

def reconstruct_bias_scale(arch: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Per (station, issued_date) multiplicative bias scale using only
    information available at issuance: h=1 forecasts whose target_date is
    strictly before issued_date, within a 30-day window, joined to obs.

    Production derives this from the analysis_assimilation overlap
    (app.nwm.hindcast_skill); that series can't be refetched for past
    dates, so trailing short-lead forecast error is the honest stand-in —
    same mean-ratio formula, same clip, same min-overlap.
    """
    h1 = arch[arch["horizon_day"] == 1][["station_id", "target_date", "q_cfs_raw"]]
    omap = {(s, d): q for s, d, q in zip(obs["station_id"], obs["date"], obs["q_cfs"])}
    pairs: dict[str, list[tuple[date, float, float]]] = {}
    for s, t, qr in zip(h1["station_id"], h1["target_date"], h1["q_cfs_raw"]):
        qo = omap.get((s, t.isoformat()))
        if qo is None or not np.isfinite(qr):
            continue
        pairs.setdefault(s, []).append((t, qr, qo))

    issue_dates = sorted(arch["issued_date"].unique())
    out_rows = []
    for s, plist in pairs.items():
        plist.sort()
        tds = [p[0] for p in plist]
        for D in issue_dates:
            lo, hi = D - timedelta(days=BIAS_LOOKBACK_DAYS), D - timedelta(days=1)
            i0 = np.searchsorted(tds, lo, side="left")
            i1 = np.searchsorted(tds, hi, side="right")
            window = plist[i0:i1]
            if len(window) < BIAS_MIN_OVERLAP:
                continue
            fc = np.array([w[1] for w in window], dtype=np.float64)
            ob = np.array([w[2] for w in window], dtype=np.float64)
            fc_mean, ob_mean = float(fc.mean()), float(ob.mean())
            if fc_mean > 1e-3 and ob_mean > 1e-3:
                scale = float(np.clip(ob_mean / fc_mean, *BIAS_CLIP))
            else:
                scale = 1.0
            logmae = float(np.mean(np.abs(np.log1p(np.clip(ob, 0, None))
                                          - np.log1p(np.clip(fc, 0, None)))))
            out_rows.append((s, D, scale, logmae))
    return pd.DataFrame(
        out_rows,
        columns=["station_id", "issued_date", "bias_scale", "trail_h1_logmae"],
    )


# ------------------------------------------------------------------ panel

def build_panel(arch: pd.DataFrame, obs: pd.DataFrame, bias: pd.DataFrame) -> pd.DataFrame:
    omap = {(s, d): q for s, d, q in zip(obs["station_id"], obs["date"], obs["q_cfs"])}
    panel = arch.copy()
    panel["q_obs"] = [
        omap.get((s, t.isoformat()))
        for s, t in zip(panel["station_id"], panel["target_date"])
    ]
    # obs at issuance: live snapshot rows carry the value production used;
    # backfilled rows reconstruct it from the records (issued-day obs,
    # falling back one day, matching what the build would have had).
    t0 = []
    for s, D, snap in zip(panel["station_id"], panel["issued_date"], panel["q_cfs_obs_today"]):
        if np.isfinite(snap):
            t0.append(float(snap))
            continue
        v = omap.get((s, D.isoformat()))
        if v is None:
            v = omap.get((s, (D - timedelta(days=1)).isoformat()))
        t0.append(v if v is not None else np.nan)
    panel["q_obs_t0"] = t0

    # Issuance-time obs lags / trailing mean, computed once per
    # (station, issued_date) then merged. All strictly <= issuance day.
    obs_by_st: dict[str, dict[str, float]] = {}
    for s, d, q in zip(obs["station_id"], obs["date"], obs["q_cfs"]):
        obs_by_st.setdefault(s, {})[d] = q
    iss_keys = panel[["station_id", "issued_date"]].drop_duplicates()
    lag_rows = []
    for s, D in zip(iss_keys["station_id"], iss_keys["issued_date"]):
        rows = obs_by_st.get(s, {})
        lag3 = rows.get((D - timedelta(days=3)).isoformat())
        lag7 = rows.get((D - timedelta(days=7)).isoformat())
        trail = [rows.get((D - timedelta(days=k)).isoformat()) for k in range(0, 30)]
        trail = [v for v in trail if v is not None]
        lag_rows.append((s, D, lag3, lag7,
                         float(np.mean(trail)) if trail else np.nan))
    lags = pd.DataFrame(
        lag_rows,
        columns=["station_id", "issued_date", "q_obs_lag3", "q_obs_lag7", "q_obs_trail30"],
    )
    panel = panel.merge(lags, on=["station_id", "issued_date"], how="left")

    panel = panel.merge(bias, on=["station_id", "issued_date"], how="left")
    panel["bias_scale"] = panel["bias_scale"].fillna(1.0)
    panel["trail_h1_logmae"] = panel["trail_h1_logmae"].fillna(panel["trail_h1_logmae"].median())
    panel["q_corrected"] = (panel["q_cfs_raw"] * panel["bias_scale"]).clip(lower=0.0)
    return panel.dropna(subset=["q_obs"]).reset_index(drop=True)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    td = pd.to_datetime(panel["target_date"])
    log_t0 = np.log1p(panel["q_obs_t0"].clip(lower=0).fillna(0))
    log_lag3 = np.log1p(panel["q_obs_lag3"].clip(lower=0))
    panel = panel.assign(
        log1p_q_nwm_raw=np.log1p(panel["q_cfs_raw"].clip(lower=0)),
        log1p_q_nwm_corrected=np.log1p(panel["q_corrected"]),
        log1p_q_obs_t0=log_t0,
        d_anchor=log_t0 - np.log1p(panel["q_corrected"]),
        log1p_q_obs_lag3=log_lag3.fillna(log_t0),
        log1p_q_obs_lag7=np.log1p(panel["q_obs_lag7"].clip(lower=0)).fillna(log_t0),
        obs_trend_3d=(log_t0 - log_lag3).fillna(0.0),
        log1p_obs_trail30=np.log1p(panel["q_obs_trail30"].clip(lower=0)).fillna(log_t0),
        doy=td.dt.dayofyear,
        month=td.dt.month,
    )
    panel["target_log1p_residual"] = (
        np.log1p(panel["q_obs"].clip(lower=0)) - panel["log1p_q_nwm_corrected"]
    )
    return panel


def add_trailing_residual(panel: pd.DataFrame) -> pd.DataFrame:
    """trail_resid_h: trailing 30d mean signed log-residual of THIS
    station+horizon, over rows whose target_date precedes issuance — i.e.
    only outcomes already observable at issue time. Honest by construction."""
    panel = panel.sort_values(["station_id", "horizon_day", "target_date"],
                              kind="stable").reset_index(drop=True)
    resid = panel["target_log1p_residual"].to_numpy(dtype=np.float64)
    out = np.zeros(len(panel))
    for _, idx in panel.groupby(["station_id", "horizon_day"]).indices.items():
        tds = panel["target_date"].to_numpy()[idx]
        iss = panel["issued_date"].to_numpy()[idx]
        r = resid[idx]
        csum = np.concatenate([[0.0], np.cumsum(r)])
        for j, D in enumerate(iss):
            i1 = np.searchsorted(tds, D, side="left")          # target < D
            i0 = np.searchsorted(tds, D - timedelta(days=30), side="left")
            if i1 > i0:
                out[idx[j]] = (csum[i1] - csum[i0]) / (i1 - i0)
    panel["trail_resid_h"] = out
    return panel


# ------------------------------------------------------------- residual ML

def load_prod_models(models_dir: Path) -> dict[int, dict]:
    out = {}
    for p in sorted(Path(models_dir).glob("h*.pkl")):
        try:
            with open(p, "rb") as f:
                out[int(p.stem[1:])] = pickle.load(f)
        except Exception:
            continue
    return out


def predict_residual(panel: pd.DataFrame, models: dict[int, dict]) -> np.ndarray:
    """Vectorized equivalent of app.nwm_residual.apply_residual: predict the
    log1p residual per horizon, add to log1p(corrected), expm1 back.
    Horizons without a model pass the corrected value through."""
    pred = panel["q_corrected"].to_numpy(dtype=np.float64).copy()
    for h, bundle in models.items():
        mask = (panel["horizon_day"] == h).to_numpy()
        if not mask.any():
            continue
        X = panel.loc[mask, bundle["feature_cols"]].to_numpy(dtype=np.float32)
        delta = bundle["model"].predict(X)
        base = np.log1p(panel.loc[mask, "q_corrected"].to_numpy(dtype=np.float64))
        pred[mask] = np.clip(np.expm1(base + delta), 0.0, None)
    return pred


def train_clean(train: pd.DataFrame, horizons: list[int], feat_cols: list[str],
                *, seed: int = 7) -> dict[int, dict]:
    import lightgbm as lgb
    models: dict[int, dict] = {}
    for h in horizons:
        sub = train[train["horizon_day"] == h].sort_values("issued_date")
        if len(sub) < 2000:
            continue
        X = sub[feat_cols].to_numpy(dtype=np.float32)
        y = sub["target_log1p_residual"].to_numpy(dtype=np.float32)
        n_val = max(1, len(X) // 10)  # chronological tail for early stopping
        model = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.04, num_leaves=31,
            min_data_in_leaf=200, feature_fraction=0.9,
            bagging_fraction=0.9, bagging_freq=5,
            objective="regression_l1", verbose=-1, random_state=seed,
        )
        model.fit(X[:-n_val], y[:-n_val], eval_set=[(X[-n_val:], y[-n_val:])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        # cfs-space MAE on the internal chronological tail — used for honest
        # per-horizon model selection (never sees the test slice).
        vtail = sub.iloc[-n_val:]
        base = np.log1p(vtail["q_corrected"].to_numpy(dtype=np.float64))
        qhat = np.clip(np.expm1(base + model.predict(X[-n_val:])), 0, None)
        val_mae_cfs = float(np.mean(np.abs(qhat - vtail["q_obs"].to_numpy())))
        models[h] = {"model": model, "feature_cols": feat_cols,
                     "val_mae_cfs": val_mae_cfs}
    return models


def anchor_member(test: pd.DataFrame, pred: np.ndarray, *, decay_h: int = 7) -> np.ndarray:
    """Replicate app.forecast._anchor_to_observed on a flat panel:
    production serves every member shifted by (h1 prediction − last obs),
    linearly decayed over decay_h horizons. Measuring members as-served
    tells us whether that anchoring helps or hurts each one."""
    df = pd.DataFrame({
        "s": test["station_id"].to_numpy(), "D": test["issued_date"].to_numpy(),
        "h": test["horizon_day"].to_numpy(), "p": pred,
        "t0": test["q_obs_t0"].to_numpy(),
    })
    h1 = df[df["h"] == 1].set_index(["s", "D"])["p"]
    p1 = h1.reindex(pd.MultiIndex.from_arrays([df["s"], df["D"]])).to_numpy()
    delta = p1 - df["t0"].to_numpy()
    delta = np.where(np.isfinite(delta), delta, 0.0)
    w = np.clip(1.0 - (df["h"].to_numpy() - 1) / max(1, decay_h), 0.0, None)
    return np.clip(df["p"].to_numpy() - delta * w, 0.0, None)


def _per_station_median_mae(sub: pd.DataFrame, pred: np.ndarray) -> float:
    err = pd.Series(np.abs(pred - sub["q_obs"].to_numpy()), index=sub.index)
    return float(err.groupby(sub["station_id"]).mean().median())


def cv_select(train: pd.DataFrame, horizons: list[int],
              variant_cols: dict[str, list[str]]) -> dict[int, str]:
    """Forward-chaining CV inside the train window: 3 chronological
    validation blocks spread over its back half, each scored with models
    trained only on earlier issuances. Returns per-horizon winner among
    the variants plus their log-space average. The test slice is never
    consulted."""
    days = sorted(train["issued_date"].unique())
    n = len(days)
    cuts = [int(n * f) for f in (0.55, 0.70, 0.85)] + [n]
    names = list(variant_cols) + ["avg"]
    scores: dict[str, dict[int, list[float]]] = {v: {h: [] for h in horizons} for v in names}
    for b in range(3):
        lo, hi = days[cuts[b]], days[cuts[b + 1] - 1]
        tr = train[train["issued_date"] < lo]
        va = train[(train["issued_date"] >= lo) & (train["issued_date"] <= hi)]
        if not len(tr) or not len(va):
            continue
        preds: dict[str, np.ndarray] = {}
        for v, cols in variant_cols.items():
            mdl = train_clean(tr, horizons, cols)
            if mdl:
                preds[v] = predict_residual(va, mdl)
        if not preds:
            continue
        preds["avg"] = np.expm1(np.mean([np.log1p(p) for p in preds.values()], axis=0))
        for h in horizons:
            hm = (va["horizon_day"] == h).to_numpy()
            if hm.sum() < 200:
                continue
            sub = va[hm]
            for v, p in preds.items():
                scores[v][h].append(_per_station_median_mae(sub, p[hm]))
    picks: dict[int, str] = {}
    for h in horizons:
        cands = [(float(np.mean(s[h])), v) for v, s in scores.items() if s[h]]
        if cands:
            picks[h] = min(cands)[1]
    return picks


# ---------------------------------------------------------------- scoring

def score(test: pd.DataFrame, members: dict[str, np.ndarray]) -> dict:
    """Per horizon, per member: pooled MAE, per-station median/mean MAE,
    win-rate vs nwm_corrected, and per-flow-tercile median MAE."""
    test = test.copy()
    for name, pred in members.items():
        test[f"err_{name}"] = np.abs(pred - test["q_obs"].to_numpy())

    # Flow terciles by station median observed flow over the test window.
    st_flow = test.groupby("station_id")["q_obs"].median()
    q1, q2 = st_flow.quantile([1 / 3, 2 / 3])
    tercile = {s: ("low" if v <= q1 else "mid" if v <= q2 else "high")
               for s, v in st_flow.items()}
    test["tercile"] = test["station_id"].map(tercile)

    out: dict = {"horizons": {}, "tercile_cutoffs_cfs": [float(q1), float(q2)]}
    for h in sorted(test["horizon_day"].unique()):
        sub = test[test["horizon_day"] == h]
        hrow: dict = {"n_pairs": int(len(sub)), "n_stations": int(sub["station_id"].nunique()), "members": {}}
        per_station = sub.groupby("station_id")[[f"err_{m}" for m in members]].mean()
        for name in members:
            col = f"err_{name}"
            entry = {
                "mae_pooled": float(sub[col].mean()),
                "mae_station_median": float(per_station[col].median()),
                "mae_station_mean": float(per_station[col].mean()),
            }
            if name != "nwm_corrected" and "nwm_corrected" in members:
                wins = (per_station[col] < per_station["err_nwm_corrected"])
                entry["win_rate_vs_corrected"] = float(wins.mean())
            entry["tercile_mae_median"] = {}
            for t in ("low", "mid", "high"):
                per_st_t = sub[sub["tercile"] == t].groupby("station_id")[col].mean()
                entry["tercile_mae_median"][t] = (
                    float(per_st_t.median()) if len(per_st_t) else None
                )
            hrow["members"][name] = entry
        out["horizons"][str(h)] = hrow
    return out


def score_sota(test: pd.DataFrame, members: dict[str, np.ndarray]) -> dict:
    """Per-member SOTA hydrology metrics (NSE/KGE/log-NSE/PBIAS), pooled across
    all horizons per station then aggregated across stations — the same shapes
    the MB-LSTM harness reports, so NWM members and MB-LSTM are directly
    comparable. CRPS is omitted: NWM members are deterministic (no quantiles)."""
    try:
        from app import metrics
    except Exception:
        return {}
    obs = test["q_obs"].to_numpy(dtype=float)
    sid_arr = test["station_id"].to_numpy()
    out: dict = {}
    for name, pred in members.items():
        pred = np.asarray(pred, dtype=float)
        per_station: dict = {}
        for sid in np.unique(sid_arr):
            m = sid_arr == sid
            if m.sum() < 20:
                continue
            per_station[str(sid)] = metrics.all_point_metrics(obs[m], pred[m])
        agg = metrics.aggregate(per_station) if per_station else {}
        out[name] = {k: agg[k]["median"] for k in agg if isinstance(agg[k], dict) and "median" in agg[k]}
        out[name]["n_stations"] = len(per_station)
    return out


def print_table(scores: dict, members: list[str]) -> None:
    print(f"\n{'h':>3} {'n':>7}  " + "".join(f"{m:>22}" for m in members))
    print(" " * 12 + "".join(f"{'med-MAE (win%)':>22}" for _ in members))
    for h, row in sorted(scores["horizons"].items(), key=lambda kv: int(kv[0])):
        line = f"{h:>3} {row['n_pairs']:>7}  "
        for m in members:
            e = row["members"][m]
            med = e["mae_station_median"]
            win = e.get("win_rate_vs_corrected")
            tag = f"{med:>10.1f}"
            tag += f" ({100*win:3.0f}%)" if win is not None else "       "
            line += f"{tag:>22}"
        print(line)


# ------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--train-end", default="2026-05-16",
                    help="last issued_date (inclusive) usable for training")
    ap.add_argument("--label", default="v1")
    ap.add_argument("--refresh-obs", action="store_true")
    ap.add_argument("--skip-clean-retrain", action="store_true")
    ap.add_argument("--no-cv", action="store_true",
                    help="skip the forward-chaining CV selection (faster)")
    ap.add_argument("--save-clean-models", default="",
                    help="optional dir to dump the clean retrained pickles")
    args = ap.parse_args()
    train_end = date.fromisoformat(args.train_end)

    t0 = time.time()
    arch = load_archive(Path(args.archive_dir))
    issue_dates = sorted(arch["issued_date"].unique())
    print(f"archive: {len(arch):,} rows, {arch['station_id'].nunique()} stations, "
          f"issued {issue_dates[0]} → {issue_dates[-1]} ({len(issue_dates)} days)")

    stations = sorted(arch["station_id"].unique())
    obs_start = issue_dates[0] - timedelta(days=BIAS_LOOKBACK_DAYS + 2)
    obs = load_observations(stations, obs_start, date.today(), refresh=args.refresh_obs)
    print(f"observations: {len(obs):,} rows, {obs['station_id'].nunique()} stations")

    bias = reconstruct_bias_scale(arch, obs)
    print(f"bias scales: {len(bias):,} (station, issue-day) pairs reconstructed")

    panel = add_trailing_residual(add_features(build_panel(arch, obs, bias)))
    print(f"labeled panel: {len(panel):,}/{len(arch):,} rows "
          f"({100 * len(panel) / len(arch):.1f}% labeled)")

    is_train = panel["issued_date"] <= train_end
    # Strict no-leak: drop training rows whose target lands in the test window.
    train = panel[is_train & (panel["target_date"] <= train_end)]
    test = panel[~is_train].copy()
    test = test.dropna(subset=["q_obs_t0"])
    print(f"split: train={len(train):,} rows (issued ≤ {train_end}), "
          f"test={len(test):,} rows (issued {test['issued_date'].min()} → {test['issued_date'].max()})")

    members: dict[str, np.ndarray] = {
        "persistence": test["q_obs_t0"].to_numpy(dtype=np.float64),
        "nwm_raw": test["q_cfs_raw"].to_numpy(dtype=np.float64),
        "nwm_corrected": test["q_corrected"].to_numpy(dtype=np.float64),
    }

    payload_picks = None
    prod = load_prod_models(PROD_MODELS_DIR)
    if prod:
        members["nwm_residual_prod"] = predict_residual(test, prod)
        print(f"production residual models: h={sorted(prod.keys())} "
              f"(trained_at may postdate the split — see manifest; treat as leak-tainted)")

    if not args.skip_clean_retrain:
        horizons = [int(h) for h in sorted(arch["horizon_day"].unique())]
        print("retraining clean residual models on train slice only…")
        variants = {
            "v1": FEAT_COLS_V1,
            "v2": FEAT_COLS_V2,
            "v3": FEAT_COLS_V3,
        }
        trained: dict[str, dict[int, dict]] = {}
        for name, cols in variants.items():
            mdl = train_clean(train, horizons, cols)
            if not mdl:
                continue
            trained[name] = mdl
            members[f"nwm_residual_{name}"] = predict_residual(test, mdl)
            print(f"  nwm_residual_{name}: h={sorted(mdl.keys())}")
            if args.save_clean_models:
                outdir = Path(args.save_clean_models) / name
                outdir.mkdir(parents=True, exist_ok=True)
                for h, b in mdl.items():
                    with open(outdir / f"h{h}.pkl", "wb") as f:
                        pickle.dump(b, f)

        if len(trained) > 1:
            stack = np.mean([np.log1p(members[f"nwm_residual_{n}"]) for n in trained], axis=0)
            members["nwm_residual_avg"] = np.expm1(stack)
        if len(trained) > 1 and not args.no_cv:
            print("  forward-chaining CV selection inside train window…")
            picks = cv_select(train, horizons, variants)
            print(f"  cv picks: {picks}")
            best = np.array(members["nwm_residual_avg"])  # default
            hcol = test["horizon_day"].to_numpy()
            for h, v in picks.items():
                src = members["nwm_residual_avg" if v == "avg" else f"nwm_residual_{v}"]
                best[hcol == h] = src[hcol == h]
            members["nwm_residual_best"] = best
            payload_picks = picks

    # Production parity: how do the key members score AS SERVED, i.e. after
    # forecast.py's anchor-to-observed? decay_h=7 is production's setting;
    # decay_h=2 is the candidate for the residual member, which already
    # self-anchors through its d_anchor / obs_t0 features.
    if "nwm_corrected" in members:
        members["nwm_corr_anch7"] = anchor_member(test, members["nwm_corrected"])
    for resid_name in ("nwm_residual_avg", "nwm_residual_clean"):
        if resid_name in members:
            members["resid_anch7"] = anchor_member(test, members[resid_name])
            members["resid_anch2"] = anchor_member(test, members[resid_name], decay_h=2)
            break

    scores = score(test, members)
    member_names = list(members.keys())
    print_table(scores, member_names)

    sota = score_sota(test, members)
    if sota:
        print("\nSOTA metrics (median across stations, pooled over horizons):")
        print(f"{'member':>20} {'NSE':>8} {'KGE':>8} {'logNSE':>8} {'PBIAS%':>8}")
        for name in member_names:
            s = sota.get(name, {})
            print(f"{name:>20} {s.get('nse', float('nan')):>8.3f} "
                  f"{s.get('kge', float('nan')):>8.3f} {s.get('log_nse', float('nan')):>8.3f} "
                  f"{s.get('pct_bias', float('nan')):>8.1f}")

    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "label": args.label,
        "ran_at": pd.Timestamp.utcnow().isoformat(),
        "train_end": args.train_end,
        "archive_issue_days": len(issue_dates),
        "n_test_rows": int(len(test)),
        "n_test_stations": int(test["station_id"].nunique()),
        "members": member_names,
        "cv_picks": ({int(k): v for k, v in payload_picks.items()}
                     if payload_picks else None),
        "scores": scores,
        "sota_metrics": sota,
    }
    out = OUT_DIR / f"nwm_backtest_{args.label}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  ({time.time() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
