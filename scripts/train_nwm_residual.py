#!/usr/bin/env python3
"""v15.1: train a learned NWM residual model from the v14.2 nwm-archive.

The v14.2 NWM forecast archive accumulates one row per
(issued_date × station × target_date × horizon_day) tuple, with the raw
NWM medium_range_blend forecast (q_cfs_raw), the observed flow at
issuance time (q_cfs_obs_today), and the multiplicative bias scale that
v14.1 derived from the analysis_assimilation overlap.

This script:
  1. Restores the nwm-archive branch into a worktree
  2. Joins the archive with cached USGS daily records to attach the
     ground-truth target flow at each (target_date, station)
  3. Builds per-horizon training matrices (h=1..14)
  4. Trains a LightGBM model per horizon predicting the residual
        log1p(q_obs) - log1p(q_nwm_corrected)
     where q_nwm_corrected = q_cfs_raw * bias_scale_used (if available).
     Working in log-space stops huge-flow stations from dominating loss.
  5. Saves per-horizon models to data/nwm_residual_models/h{N}.pkl,
     committed to git so the build pipeline can pick them up.

Why per-horizon: NWM error grows non-linearly with horizon, and the
forcing-uncertainty signal (h=8+) looks different from the
initial-condition signal (h=1-2). One model per horizon keeps each
small enough to fit on CPU and fast enough to score in <1ms.

Why log-space: matches the asinh transform our other ML members use.

Threshold for training: at least N_MIN_PAIRS labeled rows per horizon.
Below that, we skip the horizon and the inference module falls back to
plain v14.1-corrected NWM (no learned residual).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
USGS_RECORDS_DIR = ROOT / "data" / "cache" / "usgs_records"
MODELS_DIR = ROOT / "data" / "nwm_residual_models"
N_MIN_PAIRS = 5000  # per-horizon minimum
HORIZONS = list(range(1, 15))


def _restore_archive(workdir: Path) -> Path:
    """Worktree-checkout the nwm-archive branch and return the archive
    dir. Caller is responsible for cleanup."""
    subprocess.run(
        ["git", "fetch", "origin", "nwm-archive", "--depth=1"],
        cwd=ROOT, check=True,
    )
    target = workdir / "nwm-archive"
    subprocess.run(
        ["git", "worktree", "add", str(target), "nwm-archive"],
        cwd=ROOT, check=True,
    )
    return target / "archive"


def _load_archive(archive_dir: Path) -> pd.DataFrame:
    """Concatenate every per-day CSV in the archive into one DataFrame."""
    frames = []
    for p in sorted(archive_dir.rglob("*.csv.gz")):
        with gzip.open(p, "rt") as f:
            df = pd.read_csv(f)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Coerce types
    out["issued_date"] = pd.to_datetime(out["issued_date"]).dt.date
    out["target_date"] = pd.to_datetime(out["target_date"]).dt.date
    out["horizon_day"] = out["horizon_day"].astype(int)
    out["station_id"] = out["station_id"].astype(str)
    out["q_cfs_raw"] = pd.to_numeric(out["q_cfs_raw"], errors="coerce")
    out["q_cfs_obs_today"] = pd.to_numeric(out["q_cfs_obs_today"], errors="coerce")
    out["bias_scale_used"] = pd.to_numeric(out["bias_scale_used"], errors="coerce")
    return out


def _attach_targets(arch: pd.DataFrame) -> pd.DataFrame:
    """For each (station_id, target_date) in the archive, look up the
    observed cfs from the cached USGS records. Drop rows we can't
    label."""
    if arch.empty:
        return arch
    arch = arch.copy()
    obs_lookup: dict[tuple[str, str], float] = {}
    needed_stations = arch["station_id"].unique()
    for sid in needed_stations:
        f = USGS_RECORDS_DIR / f"{sid}.json"
        if not f.exists():
            continue
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        rows = rec.get("rows") or {}
        for d, q in rows.items():
            if q is None:
                continue
            obs_lookup[(sid, d)] = float(q)
    arch["q_cfs_obs"] = [
        obs_lookup.get((s, d.isoformat()))
        for s, d in zip(arch["station_id"], arch["target_date"])
    ]
    return arch.dropna(subset=["q_cfs_obs", "q_cfs_raw"]).reset_index(drop=True)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight features that are cheap to compute at inference time."""
    bs = df["bias_scale_used"].fillna(1.0).clip(0.5, 2.0)
    q_corr = (df["q_cfs_raw"] * bs).clip(lower=0.0)
    out = pd.DataFrame({
        "log1p_q_nwm_raw": np.log1p(df["q_cfs_raw"].clip(lower=0)),
        "log1p_q_nwm_corrected": np.log1p(q_corr),
        "log1p_q_obs_t0": np.log1p(df["q_cfs_obs_today"].clip(lower=0).fillna(0)),
        "bias_scale": bs,
        "horizon_day": df["horizon_day"],
        "doy": pd.to_datetime(df["target_date"]).dt.dayofyear,
        "month": pd.to_datetime(df["target_date"]).dt.month,
    })
    out["target_log1p_residual"] = (
        np.log1p(df["q_cfs_obs"].clip(lower=0)) - out["log1p_q_nwm_corrected"]
    )
    out["station_id"] = df["station_id"]
    return out


def _train_one(df: pd.DataFrame, horizon: int) -> Optional[dict]:
    sub = df[df["horizon_day"] == horizon]
    if len(sub) < N_MIN_PAIRS:
        print(f"  h{horizon}: only {len(sub)} pairs (< {N_MIN_PAIRS}) — skipping")
        return None
    try:
        import lightgbm as lgb
    except ImportError:
        print("lightgbm not installed", file=sys.stderr)
        return None
    feat_cols = [
        "log1p_q_nwm_raw", "log1p_q_nwm_corrected", "log1p_q_obs_t0",
        "bias_scale", "doy", "month",
    ]
    X = sub[feat_cols].to_numpy(dtype=np.float32)
    y = sub["target_log1p_residual"].to_numpy(dtype=np.float32)
    # Hold last 10% chronologically for validation (rough — relies on
    # archive being roughly chronological by issued_date)
    n = len(X)
    n_val = max(1, n // 10)
    Xtr, Xv = X[:-n_val], X[-n_val:]
    ytr, yv = y[:-n_val], y[-n_val:]
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        objective="regression_l1",  # MAE — matches our headline metric
        verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(30, verbose=False)])
    pred_v = model.predict(Xv)
    mae_v = float(np.mean(np.abs(pred_v - yv)))
    mae_base = float(np.mean(np.abs(yv)))
    # cfs-space MAE: invert log1p to get the real-world headline number.
    # log1p_q_obs = log1p_q_nwm_corrected + residual; reconstruct both
    # the v14.1 baseline (predict 0 residual) and v15.1 learned forecast.
    log1p_q_corr_v = sub["log1p_q_nwm_corrected"].to_numpy(dtype=np.float64)[-n_val:]
    log1p_q_obs_v = log1p_q_corr_v + yv.astype(np.float64)
    q_obs_cfs = np.expm1(log1p_q_obs_v)
    q_base_cfs = np.expm1(log1p_q_corr_v)
    q_learn_cfs = np.expm1(log1p_q_corr_v + pred_v.astype(np.float64))
    mae_base_cfs = float(np.mean(np.abs(q_base_cfs - q_obs_cfs)))
    mae_learn_cfs = float(np.mean(np.abs(q_learn_cfs - q_obs_cfs)))
    print(
        f"  h{horizon}: n={n:>7d}  log1p MAE base={mae_base:.4f} learn={mae_v:.4f} ({100*(mae_v-mae_base)/mae_base:+.0f}%) | "
        f"cfs MAE base={mae_base_cfs:,.0f} learn={mae_learn_cfs:,.0f} ({100*(mae_learn_cfs-mae_base_cfs)/max(mae_base_cfs,1e-6):+.0f}%)"
    )
    return {
        "model": model,
        "feature_cols": feat_cols,
        "n_train": int(n - n_val),
        "n_val": int(n_val),
        "val_mae_baseline": mae_base,
        "val_mae_learned": mae_v,
        "val_mae_baseline_cfs": mae_base_cfs,
        "val_mae_learned_cfs": mae_learn_cfs,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", default="",
                   help="Use this dir instead of fetching the nwm-archive branch.")
    p.add_argument("--out-dir", default=str(MODELS_DIR))
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
        labeled = _attach_targets(arch)
        print(f"labeled {len(labeled):,}/{len(arch):,} rows ({100*len(labeled)/len(arch):.1f}%)")
        feats = _build_features(labeled)
        print(f"feature matrix: {len(feats):,} rows")

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict = {
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_archive_rows": int(len(arch)),
            "n_labeled_rows": int(len(labeled)),
            "horizons": {},
        }
        for h in HORIZONS:
            r = _train_one(feats, h)
            if r is None:
                continue
            with open(out_dir / f"h{h}.pkl", "wb") as f:
                pickle.dump({
                    "model": r["model"],
                    "feature_cols": r["feature_cols"],
                }, f)
            manifest["horizons"][str(h)] = {
                "n_train": r["n_train"],
                "n_val": r["n_val"],
                "val_mae_baseline_log1p": r["val_mae_baseline"],
                "val_mae_learned_log1p": r["val_mae_learned"],
                "val_mae_baseline_cfs": r["val_mae_baseline_cfs"],
                "val_mae_learned_cfs": r["val_mae_learned_cfs"],
            }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nWrote {len(manifest['horizons'])} per-horizon models → {out_dir}")
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
