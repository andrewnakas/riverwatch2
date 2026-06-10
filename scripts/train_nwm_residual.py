#!/usr/bin/env python3
"""v15.9: train the NWM residual ensemble from the v14.2 nwm-archive.

Honest rebuild after BACKTEST_REPORT.md. The v15.1 trainer had four bugs
that made its manifest ratios unearned:
  1. station_id parsed as int → every leading-zero (region 01-09) gauge
     silently failed the USGS-records join and fell out of training.
  2. Trained with q_obs_t0 backfilled to 0, served with the real value
     (train/serve feature skew).
  3. Validation baseline was RAW NWM (empty bias_scale_used → 1.0), but
     the live `nwm` member it's displayed against is bias-corrected.
  4. The "chronological" 10% split was by file order, pooled across
     stations, so recent target dates could leak into training.

This version:
  - reads station_id as a string (fix 1),
  - reconstructs q_obs_t0 from cached USGS records for backfilled rows
    and bias_scale from the trailing h=1 forecast-vs-obs window — the
    same information the live build has at issuance (fixes 2 and 3),
  - splits by issued_date with the validation block at the end and
    drops any training row whose target_date reaches it (fix 4),
  - trains the three feature variants defined in app.nwm_residual
    (single source of truth — serving computes identical features) and
    reports the honest val MAE of their log-space average against the
    bias-corrected baseline,
  - then retrains on the full archive for shipping, and writes
    sidecar.json with the per-station trailing-skill features v2/v3
    need at serve time.

Outputs under data/nwm_residual_models/:
  v1/h{N}.pkl  v2/h{N}.pkl  v3/h{N}.pkl  sidecar.json  manifest.json

The manifest's val_mae_baseline_cfs / val_mae_learned_cfs feed
app.forecast._load_resid_scale (clamped [0.50, 1.05]) — they are real
held-out numbers under this protocol, not the model's own training val.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import subprocess
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.nwm_residual import VARIANT_COLS  # noqa: E402

USGS_RECORDS_DIR = ROOT / "data" / "cache" / "usgs_records"
MODELS_DIR = ROOT / "data" / "nwm_residual_models"
N_MIN_PAIRS = 5000  # per-horizon minimum
HORIZONS = list(range(1, 15))
VAL_FRACTION = 0.15  # final issue-days held out for the manifest metrics

# Mirror app.nwm.hindcast_skill's guardrails: >=7 overlap days, clip [0.5, 2].
BIAS_MIN_OVERLAP = 7
BIAS_LOOKBACK_DAYS = 30
BIAS_CLIP = (0.5, 2.0)


def _restore_archive(workdir: Path) -> Path:
    """Worktree-checkout the nwm-archive branch and return the archive
    dir. Caller is responsible for cleanup."""
    subprocess.run(
        ["git", "fetch", "origin", "nwm-archive", "--depth=1"],
        cwd=ROOT, check=True,
    )
    target = workdir / "nwm-archive"
    subprocess.run(
        ["git", "worktree", "add", str(target), "FETCH_HEAD"],
        cwd=ROOT, check=True,
    )
    return target / "archive"


def _load_archive(archive_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(archive_dir).rglob("*.csv.gz")):
        with gzip.open(p, "rt") as f:
            frames.append(pd.read_csv(f, dtype={"station_id": str}))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["issued_date"] = pd.to_datetime(out["issued_date"]).dt.date
    out["target_date"] = pd.to_datetime(out["target_date"]).dt.date
    out["horizon_day"] = out["horizon_day"].astype(int)
    out["q_cfs_raw"] = pd.to_numeric(out["q_cfs_raw"], errors="coerce")
    out["q_cfs_obs_today"] = pd.to_numeric(out["q_cfs_obs_today"], errors="coerce")
    out = out.dropna(subset=["q_cfs_raw"])
    return out[out["q_cfs_raw"] >= 0].reset_index(drop=True)


def _load_obs(stations: list[str], records_dir: Path) -> dict[str, dict[str, float]]:
    """{station: {iso_date: cfs}} from the cached USGS daily records."""
    out: dict[str, dict[str, float]] = {}
    for sid in stations:
        f = records_dir / f"{sid}.json"
        if not f.exists():
            continue
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        rows = {d: float(q) for d, q in (rec.get("rows") or {}).items() if q is not None}
        if rows:
            out[sid] = rows
    return out


def _reconstruct_bias(arch: pd.DataFrame, obs: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Per (station, issued_date): multiplicative bias scale and trailing
    |log error| from h=1 forecasts whose targets precede issuance. Only
    information available at issue time."""
    h1 = arch[arch["horizon_day"] == 1]
    pairs: dict[str, list[tuple[date, float, float]]] = {}
    for s, t, qr in zip(h1["station_id"], h1["target_date"], h1["q_cfs_raw"]):
        qo = (obs.get(s) or {}).get(t.isoformat())
        if qo is None:
            continue
        pairs.setdefault(s, []).append((t, float(qr), qo))
    issue_dates = sorted(arch["issued_date"].unique())
    rows = []
    for s, plist in pairs.items():
        plist.sort()
        tds = [p[0] for p in plist]
        for D in issue_dates:
            i0 = np.searchsorted(tds, D - timedelta(days=BIAS_LOOKBACK_DAYS), side="left")
            i1 = np.searchsorted(tds, D - timedelta(days=1), side="right")
            window = plist[i0:i1]
            if len(window) < BIAS_MIN_OVERLAP:
                continue
            fc = np.array([w[1] for w in window])
            ob = np.array([w[2] for w in window])
            scale = 1.0
            if fc.mean() > 1e-3 and ob.mean() > 1e-3:
                scale = float(np.clip(ob.mean() / fc.mean(), *BIAS_CLIP))
            logmae = float(np.mean(np.abs(np.log1p(np.clip(ob, 0, None))
                                          - np.log1p(np.clip(fc, 0, None)))))
            rows.append((s, D, scale, logmae))
    return pd.DataFrame(rows, columns=["station_id", "issued_date",
                                       "bias_scale", "trail_h1_logmae"])


def _build_panel(arch: pd.DataFrame, obs: dict[str, dict[str, float]],
                 bias: pd.DataFrame) -> pd.DataFrame:
    panel = arch.copy()
    panel["q_obs"] = [
        (obs.get(s) or {}).get(t.isoformat())
        for s, t in zip(panel["station_id"], panel["target_date"])
    ]
    # Obs at issuance: live rows recorded what the build actually used;
    # backfilled rows reconstruct it from the records (no fillna(0)).
    t0 = []
    for s, D, snap in zip(panel["station_id"], panel["issued_date"], panel["q_cfs_obs_today"]):
        if np.isfinite(snap):
            t0.append(float(snap))
            continue
        srec = obs.get(s) or {}
        v = srec.get(D.isoformat())
        if v is None:
            v = srec.get((D - timedelta(days=1)).isoformat())
        t0.append(v if v is not None else np.nan)
    panel["q_obs_t0"] = t0

    # Issuance-time obs lags / trailing mean.
    keys = panel[["station_id", "issued_date"]].drop_duplicates()
    lag_rows = []
    for s, D in zip(keys["station_id"], keys["issued_date"]):
        srec = obs.get(s) or {}
        lag3 = srec.get((D - timedelta(days=3)).isoformat())
        lag7 = srec.get((D - timedelta(days=7)).isoformat())
        trail = [srec.get((D - timedelta(days=k)).isoformat()) for k in range(0, 30)]
        trail = [v for v in trail if v is not None]
        lag_rows.append((s, D, lag3, lag7, float(np.mean(trail)) if trail else np.nan))
    lags = pd.DataFrame(lag_rows, columns=["station_id", "issued_date",
                                           "q_obs_lag3", "q_obs_lag7", "q_obs_trail30"])
    panel = panel.merge(lags, on=["station_id", "issued_date"], how="left")
    panel = panel.merge(bias, on=["station_id", "issued_date"], how="left")
    panel["bias_scale"] = panel["bias_scale"].fillna(1.0)
    panel["trail_h1_logmae"] = panel["trail_h1_logmae"].fillna(
        panel["trail_h1_logmae"].median()
    )
    panel["q_corrected"] = (panel["q_cfs_raw"] * panel["bias_scale"]).clip(lower=0.0)
    panel = panel.dropna(subset=["q_obs", "q_obs_t0"]).reset_index(drop=True)

    td = pd.to_datetime(panel["target_date"])
    log_t0 = np.log1p(panel["q_obs_t0"].clip(lower=0))
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

    # trail_resid_h: trailing 30d mean signed log residual of this
    # station+horizon over rows whose target precedes issuance.
    panel = panel.sort_values(["station_id", "horizon_day", "target_date"],
                              kind="stable").reset_index(drop=True)
    resid = panel["target_log1p_residual"].to_numpy(dtype=np.float64)
    trail_col = np.zeros(len(panel))
    for _, idx in panel.groupby(["station_id", "horizon_day"]).indices.items():
        tds = panel["target_date"].to_numpy()[idx]
        iss = panel["issued_date"].to_numpy()[idx]
        csum = np.concatenate([[0.0], np.cumsum(resid[idx])])
        for j, D in enumerate(iss):
            i1 = np.searchsorted(tds, D, side="left")
            i0 = np.searchsorted(tds, D - timedelta(days=30), side="left")
            if i1 > i0:
                trail_col[idx[j]] = (csum[i1] - csum[i0]) / (i1 - i0)
    panel["trail_resid_h"] = trail_col
    return panel


def _train_variant(df: pd.DataFrame, horizon: int, feat_cols: list[str],
                   *, seed: int = 7) -> Optional[dict]:
    sub = df[df["horizon_day"] == horizon].sort_values("issued_date")
    if len(sub) < N_MIN_PAIRS:
        return None
    try:
        import lightgbm as lgb
    except ImportError:
        print("lightgbm not installed", file=sys.stderr)
        return None
    X = sub[feat_cols].to_numpy(dtype=np.float32)
    y = sub["target_log1p_residual"].to_numpy(dtype=np.float32)
    n_es = max(1, len(X) // 10)  # chronological tail for early stopping only
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.04, num_leaves=31,
        min_data_in_leaf=200, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5,
        objective="regression_l1", verbose=-1, random_state=seed,
    )
    model.fit(X[:-n_es], y[:-n_es], eval_set=[(X[-n_es:], y[-n_es:])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return {"model": model, "feature_cols": feat_cols}


def _predict(panel: pd.DataFrame, models: dict[int, dict]) -> np.ndarray:
    pred = panel["q_corrected"].to_numpy(dtype=np.float64).copy()
    for h, bundle in models.items():
        mask = (panel["horizon_day"] == h).to_numpy()
        if not mask.any():
            continue
        X = panel.loc[mask, bundle["feature_cols"]].to_numpy(dtype=np.float32)
        base = np.log1p(panel.loc[mask, "q_corrected"].to_numpy(dtype=np.float64))
        pred[mask] = np.clip(np.expm1(base + bundle["model"].predict(X)), 0.0, None)
    return pred


def _ensemble(preds: list[np.ndarray]) -> np.ndarray:
    return np.expm1(np.mean([np.log1p(p) for p in preds], axis=0))


def _build_sidecar(panel: pd.DataFrame) -> dict:
    """Per-station trailing stats as of the latest issuance, for serving."""
    last = panel["issued_date"].max()
    snap = panel[panel["issued_date"] == last]
    out: dict = {"_as_of": last.isoformat(),
                 "_default": {"trail_h1_logmae": float(snap["trail_h1_logmae"].median())}}
    for s, g in snap.groupby("station_id"):
        out[s] = {
            "trail_h1_logmae": float(g["trail_h1_logmae"].iloc[0]),
            "trail_resid_h": {str(int(h)): float(v) for h, v in
                              zip(g["horizon_day"], g["trail_resid_h"])},
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", default="",
                   help="Use this dir instead of fetching the nwm-archive branch.")
    p.add_argument("--out-dir", default=str(MODELS_DIR))
    p.add_argument("--usgs-records-dir", default=str(USGS_RECORDS_DIR))
    args = p.parse_args()

    if args.archive_dir:
        archive_dir = Path(args.archive_dir)
        cleanup = None
    else:
        td = tempfile.mkdtemp(prefix="rw2-nwm-arch-")
        archive_dir = _restore_archive(Path(td))
        cleanup = (Path(td) / "nwm-archive")
    try:
        t0 = time.time()
        arch = _load_archive(archive_dir)
        print(f"loaded {len(arch):,} archive rows in {time.time()-t0:.1f}s")
        if arch.empty:
            print("archive is empty; nothing to train", file=sys.stderr)
            return 1
        obs = _load_obs(sorted(arch["station_id"].unique()),
                        Path(args.usgs_records_dir))
        print(f"USGS records: {len(obs)}/{arch['station_id'].nunique()} stations cached")
        bias = _reconstruct_bias(arch, obs)
        panel = _build_panel(arch, obs, bias)
        print(f"labeled panel: {len(panel):,}/{len(arch):,} rows "
              f"({100 * len(panel) / max(len(arch), 1):.1f}%)")

        # --- honest validation: hold out a block of recent issue-days ---
        # Only issue-days whose FULL h=1..14 fan already has observable
        # truth qualify for val; otherwise long horizons would be scored
        # on a tiny biased remnant (issuances from the last 2 weeks can't
        # have h=14 truth yet).
        days = sorted(panel["issued_date"].unique())
        max_obs_target = panel["target_date"].max()
        full_days = [d for d in days
                     if d + timedelta(days=max(HORIZONS)) <= max_obs_target]
        if not full_days:
            print("no issue-days with full-horizon truth; cannot validate",
                  file=sys.stderr)
            return 1
        n_val_days = max(3, int(len(full_days) * VAL_FRACTION))
        val_days = set(full_days[-n_val_days:])
        val_start = min(val_days)
        train = panel[(panel["issued_date"] < val_start)
                      & (panel["target_date"] < val_start)]
        val = panel[panel["issued_date"].isin(val_days)]
        print(f"split: train={len(train):,} (issued+target < {val_start}), "
              f"val={len(val):,} ({n_val_days} issue-days "
              f"{val_start} → {max(val_days)}, full-horizon truth)")

        manifest: dict = {
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema": "v15.9",
            "n_archive_rows": int(len(arch)),
            "n_labeled_rows": int(len(panel)),
            "val_protocol": (
                f"temporal holdout: final {n_val_days} issue-days "
                f"(≥ {val_start}); baseline = bias-corrected NWM "
                "(trailing-h1 reconstruction); learned = log-space mean of "
                "v1/v2/v3 variants; shipped models retrained on full archive"
            ),
            "horizons": {},
        }
        split_models: dict[str, dict[int, dict]] = {v: {} for v in VARIANT_COLS}
        for h in HORIZONS:
            for v, cols in VARIANT_COLS.items():
                r = _train_variant(train, h, cols)
                if r is not None:
                    split_models[v][h] = r
            got = [v for v in VARIANT_COLS if h in split_models[v]]
            if not got:
                print(f"  h{h}: insufficient rows — skipped")
                continue
            vsub = val[val["horizon_day"] == h]
            if not len(vsub):
                continue
            preds = [_predict(vsub, {h: split_models[v][h]}) for v in got]
            learned = _ensemble(preds)
            q_obs = vsub["q_obs"].to_numpy()
            base = vsub["q_corrected"].to_numpy()
            mae_base = float(np.mean(np.abs(base - q_obs)))
            mae_learn = float(np.mean(np.abs(learned - q_obs)))
            err_b = pd.Series(np.abs(base - q_obs), index=vsub.index)
            err_l = pd.Series(np.abs(learned - q_obs), index=vsub.index)
            med_base = float(err_b.groupby(vsub["station_id"]).mean().median())
            med_learn = float(err_l.groupby(vsub["station_id"]).mean().median())
            print(f"  h{h}: n_train={len(train[train['horizon_day'] == h]):,} "
                  f"val cfs MAE base={mae_base:,.0f} learn={mae_learn:,.0f} "
                  f"({100*(mae_learn-mae_base)/max(mae_base,1e-6):+.0f}%) | "
                  f"median-station base={med_base:,.0f} learn={med_learn:,.0f}")
            manifest["horizons"][str(h)] = {
                "n_train": int(len(train[train["horizon_day"] == h])),
                "n_val": int(len(vsub)),
                "variants": got,
                "val_mae_baseline_cfs": mae_base,
                "val_mae_learned_cfs": mae_learn,
                "val_mae_baseline_cfs_median_station": med_base,
                "val_mae_learned_cfs_median_station": med_learn,
            }

        # --- ship: retrain every variant on the full labeled panel ---
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Remove legacy flat pickles so the loader doesn't double-count v1.
        for stale in out_dir.glob("h*.pkl"):
            stale.unlink()
        n_shipped = 0
        for v, cols in VARIANT_COLS.items():
            vdir = out_dir / v
            vdir.mkdir(exist_ok=True)
            for h in HORIZONS:
                r = _train_variant(panel, h, cols)
                if r is None:
                    continue
                with open(vdir / f"h{h}.pkl", "wb") as f:
                    pickle.dump(r, f)
                n_shipped += 1
        (out_dir / "sidecar.json").write_text(json.dumps(_build_sidecar(panel)))
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nWrote {n_shipped} models across {len(VARIANT_COLS)} variants "
              f"+ sidecar + manifest → {out_dir}")
    finally:
        if cleanup is not None:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", str(cleanup), "--force"],
                    cwd=ROOT, check=False,
                )
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
