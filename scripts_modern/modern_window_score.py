#!/usr/bin/env python3
"""SINGLE-ARM modern-window score: serve one CAMELS-trained checkpoint its OWN
forcing over 2016-2025 and report per-basin NSE against fresh USGS discharge.

Unlike the paired point-vs-areal A/B, this produces an ABSOLUTE number, which is
only interpretable when the corpus matches what the checkpoint was trained on:

  * NLDAS  — modern corpus reproduces CAMELS NLDAS at r=1.000 on
             precip/srad/vp/tmean, so NO version and NO registration mismatch.
             ⚠️ MUST pass --tmax-tmin-as-mean: CAMELS NLDAS has tmax==tmin
             (diurnal range 0.0000 C) while ours is real (7.02 C). Serving real
             extremes shifts 2 of ~5 dynamic inputs.
  * DAYMET — carries a V2/V3->V4R1 version gap AND a one-day precip registration
             offset. Its number is a LOWER bound, not a clean read.

Normalisation: NH reloads `train_data/train_data_scaler.yml` from the run dir and
builds the dataset with is_train=False, so TRAIN statistics are reused and never
refit on 2016-2025. Verified by md5 in the earlier A/B work.

Never points NH at the real run dir — evaluation writes into `run_dir/test/` and
would pollute the ledger. A scratch clone is used.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path("/home/nakas/riverwatch2")
NHPY = ROOT / "gpu1080/.venv/bin/python"


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def shift_cols(rows, ci, cols, n):
    """Assign row[t-n]ics values to row[t] for the named columns.

    Daymet V4R1 labels precipitation (and tmax) one day EARLIER than the CAMELS
    data these checkpoints were trained on, so V4[d] corresponds to CAMELS[d+1].
    To restore the training convention we need new[d] = V4[d-n] with n=1, i.e.
    shift the series LATER by one day. The first n rows have no predecessor and
    are blanked; the corpus starts 2014 while scoring starts 2016, so they fall
    in warmup and cost nothing.
    """
    src = [[r[ci[c]] for c in cols] for r in rows]
    for t in range(len(rows) - 1, -1, -1):
        for k, c in enumerate(cols):
            rows[t][ci[c]] = src[t - n][k] if t - n >= 0 else ""
    return rows


def stage_corpus(src: Path, dst: Path, tmax_tmin_as_mean: bool, limit: int,
                 basins_only: set | None, shift_days: int = 0,
                 shift_cols_list: tuple = ()):
    """Copy the corpus, optionally collapsing tmax/tmin to tmean and/or
    shifting named columns later by `shift_days` to fix a registration offset."""
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.csv.gz"))
    if basins_only:
        files = [f for f in files if f.name.split(".")[0] in basins_only]
    if limit:
        files = files[:limit]
    n = 0
    for f in files:
        out = dst / f.name
        if not tmax_tmin_as_mean and not shift_days:
            shutil.copy2(f, out)
            n += 1
            continue
        with gzip.open(f, "rt") as fh:
            hdr = fh.readline().rstrip("\n").split(",")
            ci = {c: i for i, c in enumerate(hdr)}
            rows = [l.rstrip("\n").split(",") for l in fh if l.strip()]
        if shift_days:
            rows = shift_cols(rows, ci, shift_cols_list, shift_days)
        if tmax_tmin_as_mean:
            ix, ia, im = (ci["temperature_2m_max"], ci["temperature_2m_mean"],
                          ci["temperature_2m_min"])
            for r in rows:
                r[ix] = r[ia]
                r[im] = r[ia]
        with gzip.open(out, "wt") as fh:
            fh.write(",".join(hdr) + "\n")
            for r in rows:
                fh.write(",".join(r) + "\n")
        n += 1
    return n, [f.name.split(".")[0] for f in files]


def clone_run(run_dir: Path, epoch: int, scratch: Path, data_dir: Path,
              basin_file: Path, start: str, end: str):
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "train_data").mkdir(exist_ok=True)
    cfg = yaml.safe_load((run_dir / "config.yml").read_text())
    ck = run_dir / f"model_epoch{epoch:03d}.pt"
    if not ck.exists():
        sys.exit(f"missing checkpoint {ck}")
    shutil.copy2(ck, scratch / ck.name)
    sc = run_dir / "train_data/train_data_scaler.yml"
    if not sc.exists():
        sys.exit(f"missing scaler {sc}")
    shutil.copy2(sc, scratch / "train_data/train_data_scaler.yml")
    scaler_md5 = md5(sc)

    cfg["data_dir"] = str(data_dir)
    cfg["run_dir"] = str(scratch)
    cfg["train_dir"] = str(scratch / "train_data")
    cfg["test_start_date"] = start
    cfg["test_end_date"] = end
    for k in ("train_basin_file", "validation_basin_file", "test_basin_file"):
        cfg[k] = str(basin_file)
    cfg["device"] = "cpu"
    cfg["num_workers"] = 0
    (scratch / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg, scaler_md5


def nse(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[m], sim[m]
    if len(o) < 365 or o.std() == 0:
        return np.nan
    return float(1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--test-start", default="01/01/2016")
    ap.add_argument("--test-end", default="31/12/2025")
    ap.add_argument("--tmax-tmin-as-mean", action="store_true",
                    help="collapse tmax/tmin to tmean (REQUIRED for CAMELS-NLDAS "
                         "checkpoints, whose training data has tmax==tmin)")
    ap.add_argument("--shift-days", type=int, default=0,
                    help="shift --shift-cols LATER by N days to undo a "
                         "registration offset (Daymet V4R1 needs 1)")
    ap.add_argument("--shift-cols", default="precipitation_sum",
                    help="comma-separated columns to shift")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--work", default=str(ROOT / "data/modern_score"))
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    work = Path(a.work) / a.label
    if work.exists():
        shutil.rmtree(work)
    staged = work / "corpus"
    nhdata = work / "nh"
    scratch = work / "run"

    scols = tuple(c.strip() for c in a.shift_cols.split(",") if c.strip())
    n, basins = stage_corpus(Path(a.corpus), staged, a.tmax_tmin_as_mean,
                             a.limit, None, a.shift_days, scols)
    print(f"[{a.label}] staged {n} basins"
          + ("  (tmax=tmin=tmean applied)" if a.tmax_tmin_as_mean else "")
          + (f"  (shifted {scols} later by {a.shift_days}d)" if a.shift_days else ""))

    r = subprocess.run([str(NHPY), str(ROOT / "scripts/corpus_to_nh.py"),
                        "--forcing", "daymet", "--corpus-dir", str(staged),
                        "--out", str(nhdata)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"corpus_to_nh failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    print(f"[{a.label}] converted -> {nhdata}")

    # ⚠️ Use the CONVERTER's basin list, not the staged one. corpus_to_nh.py
    # filters to the 531 eval subset, so a staged basin outside it produces no
    # .nc and no attributes row — and NH then dies with the unhelpful
    # "Some basins are missing static attributes."
    conv_bf = nhdata / "basins.txt"
    produced = [b.strip() for b in conv_bf.read_text().split() if b.strip()] \
        if conv_bf.exists() else []
    have_ts = {p.stem for p in (nhdata / "time_series").glob("*.nc")}
    basins = [b for b in produced if b in have_ts]
    if not basins:
        sys.exit("converter produced no usable basins")
    if len(basins) != n:
        print(f"[{a.label}] note: staged {n}, converter kept {len(basins)} "
              f"(it filters to the 531 eval subset)")
    bf = work / "basins.txt"
    bf.write_text("\n".join(basins) + "\n")
    cfg, scaler_md5 = clone_run(run_dir, a.epoch, scratch, nhdata, bf,
                                a.test_start, a.test_end)
    print(f"[{a.label}] inputs={cfg.get('dynamic_inputs')}  "
          f"seq={cfg.get('seq_length')}  scaler_md5={scaler_md5[:12]}")

    r = subprocess.run([str(NHPY), "-m", "neuralhydrology.nh_run", "evaluate",
                        "--run-dir", str(scratch), "--epoch", str(a.epoch),
                        "--period", "test", "--gpu", "-1"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"nh evaluate failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    # scaler must be untouched by evaluation
    assert md5(scratch / "train_data/train_data_scaler.yml") == scaler_md5, \
        "scaler changed during evaluation - train statistics were NOT reused"

    res = list((scratch / "test").rglob("*results.p"))
    if not res:
        sys.exit("no results file produced")
    import pickle
    with open(res[0], "rb") as fh:
        R = pickle.load(fh)

    rows = []
    for b, d in R.items():
        try:
            xr_ = d["1D"]["xr"]
            obs = np.asarray(xr_["q_mm_obs"]).squeeze()
            sim = np.asarray(xr_["q_mm_sim"]).squeeze()
        except Exception:
            continue
        v = nse(obs, sim)
        if not math.isnan(v):
            rows.append((b, v, int(np.isfinite(obs).sum())))

    if not rows:
        sys.exit("no scorable basins")
    vals = np.array([r[1] for r in rows])
    print()
    print("=" * 66)
    print(f"  MODERN-WINDOW SCORE — {a.label}   {a.test_start}..{a.test_end}")
    print("=" * 66)
    print(f"  basins scored : {len(rows)}")
    print(f"  MEDIAN NSE    : {np.median(vals):.4f}")
    print(f"  mean NSE      : {np.mean(vals):.4f}")
    for q in (10, 25, 75, 90):
        print(f"  p{q:<12}: {np.percentile(vals, q):.4f}")
    print(f"  NSE > 0.0     : {(vals > 0).mean()*100:.1f}%")
    print(f"  NSE > 0.5     : {(vals > 0.5).mean()*100:.1f}%")
    print(f"  worst / best  : {vals.min():.4f} / {vals.max():.4f}")

    out = work.parent / f"modern_score_{a.label}.json"
    json.dump({"label": a.label, "run_dir": str(run_dir), "epoch": a.epoch,
               "corpus": a.corpus, "tmax_tmin_as_mean": a.tmax_tmin_as_mean,
               "shift_days": a.shift_days, "shift_cols": list(scols),
               "window": [a.test_start, a.test_end], "n": len(rows),
               "median_nse": float(np.median(vals)),
               "per_basin": {b: v for b, v, _ in rows}},
              open(out, "w"), indent=1)
    print(f"\n  wrote {out}")
    if not a.keep:
        shutil.rmtree(staged, ignore_errors=True)


if __name__ == "__main__":
    main()
