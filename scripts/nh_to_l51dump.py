#!/usr/bin/env python3
"""LEDGER 53 — convert a NeuralHydrology <period>_results.p into a LEDGER-51
day-1 dump + provenance sidecar, so `analysis/l51_withq_score.py` can score an
NH member next to the MB-LSTM members.

L51 dump schema (from scripts/train_mblstm.py --dump-day1):
  station_id,t0,h,truth,ylo,ymed,yhi     flows in cfs, target date = t0 + h
For a point model ylo == ymed == yhi == sim, h == 1, t0 = target date − 1 day.
The scorer inner-joins members on (station_id, t0) at h == 1 and asserts the
truths agree, so the DATE CONVENTION here is load-bearing: NH's results are
indexed by the target date; we emit t0 = date − 1 (LEDGER 51 measured
target = t0 + h for the MB-LSTM dumps, PREREG_v2.md §LEDGER 51).

NH results are in specific discharge (mm/day); cfs = mm/day × area_km2 / 2.446576
(the inverse of scripts/corpus_to_nh_multi.py). Provenance for the sidecar is
read from the run's config.yml, not from argv.

usage: nh_to_l51dump.py --results <results.p> --run-dir <dir> --member nhar
                        --seed 501 --frame val1|test1 --out <csv.gz>
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFS_TO_MMDAY_PER_KM2 = 0.0283168 * 86400 / 1e6 * 1000   # = 2.446576
GUARD = {"train_start_date": "01/10/1999", "train_end_date": "30/09/2008",
         "validation_start_date": "01/10/1980", "validation_end_date": "30/09/1989",
         "test_start_date": "01/10/1989", "test_end_date": "30/09/1999"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--member", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--frame", required=True, choices=["val1", "test1"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--allow-partial", action="store_true", help="SMOKE ONLY: skip the 531-basin assert")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(Path(args.run_dir) / "config.yml"))
    for k, v in GUARD.items():
        got = str(cfg.get(k))
        assert got == v, f"config {k}={got!r} != {v!r} — not the Nearing split, refusing"
    assert cfg["model"] == "arlstm", cfg["model"]
    assert int(cfg["seed"]) == args.seed, (cfg["seed"], args.seed)
    assert cfg["autoregressive_inputs"] == ["q_mm_shift1"], cfg["autoregressive_inputs"]
    period = "validation" if args.frame == "val1" else "test"
    exp_start = cfg[f"{period}_start_date"]
    exp_end = cfg[f"{period}_end_date"]

    attrs = json.loads((ROOT / "data" / "camels_attrs.json").read_text())
    ids = set(str(x).zfill(8) for x in json.load(open(ROOT / "data" / "camels_gauge_ids.json"))["531"])
    res = pickle.load(open(args.results, "rb"))

    parts, n_basins = [], 0
    for bid, per in res.items():
        sid = str(bid).zfill(8)
        if sid not in ids:
            continue
        area = attrs[sid]["area_gages2"]
        assert area and np.isfinite(area) and area > 0, sid
        xr_ = per["1D"]["xr"]
        obs = xr_["q_mm_obs"].isel(time_step=0).to_series()
        sim = xr_["q_mm_sim"].isel(time_step=0).to_series()
        dates = pd.to_datetime(obs.index)
        to_cfs = area / CFS_TO_MMDAY_PER_KM2
        d = pd.DataFrame({
            "station_id": sid,
            "t0": (dates - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "h": 1,
            "truth": obs.to_numpy(float) * to_cfs,
            "y": sim.to_numpy(float) * to_cfs,
        })
        d = d[np.isfinite(d.y)]
        parts.append(d)
        n_basins += 1
    df = pd.concat(parts, ignore_index=True)
    df["ylo"] = df["y"]; df["ymed"] = df["y"]; df["yhi"] = df["y"]
    df = df[["station_id", "t0", "h", "truth", "ylo", "ymed", "yhi"]].sort_values(["station_id", "t0"])
    # sanity: dates inside the period, no leak
    tgt = pd.to_datetime(df.t0) + pd.Timedelta(days=1)
    lo, hi = pd.to_datetime(exp_start, dayfirst=True), pd.to_datetime(exp_end, dayfirst=True)
    assert tgt.min() >= lo and tgt.max() <= hi, (tgt.min(), tgt.max(), lo, hi)
    if not args.allow_partial:
        assert n_basins == 531, f"{n_basins} basins in results, expected 531"
    assert len(df) > 1000, len(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")
    ckpt = Path(args.run_dir) / f"model_epoch{args.epoch:03d}.pt"
    side = {
        "member": args.member, "seed": args.seed, "frame": args.frame, "rows": int(len(df)),
        "ckpt": str(ckpt), "ckpt_md5": hashlib.md5(ckpt.read_bytes()).hexdigest(),
        "train_start": "1999-10-01", "train_end": "2008-09-30", "protocol": "nearing2022",
        "bytes": os.path.getsize(out), "engine": "neuralhydrology", "nh_run_dir": str(Path(args.run_dir).resolve()),
        "model": cfg["model"], "hidden_size": cfg["hidden_size"], "epoch": args.epoch,
        "train_holdout": cfg.get("random_holdout_from_dynamic_features"),
        "eval_holdout": None, "results": str(Path(args.results).resolve()),
        "target_convention": "t0 = target_date - 1 day, h = 1",
        "basins": n_basins, "period": f"{exp_start}..{exp_end}",
    }
    json.dump(side, open(str(out).replace(".csv.gz", ".json"), "w"), indent=1)
    cov = df.truth.notna().mean()
    print(f"DUMP OK {args.frame}: {len(df):,} rows, {n_basins} basins, target {tgt.min().date()}..{tgt.max().date()}, "
          f"truth coverage {cov:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
