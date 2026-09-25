#!/usr/bin/env python3
"""A/B: does serving a CAMELS-trained LSTM AREAL forcings instead of POINT
forcings actually improve NSE against real 2015-2025 observations?

THE QUESTION
------------
`data/modern_corpus/camels_corpus_daymet_modern` was built by sampling the
single Daymet pixel AT THE GAUGE (point). The models were trained on CAMELS
`basin_mean_forcing` (areal, exact-extract over the catchment polygon). We have
already measured the FORCING difference (areal precip ~+7% vs point, up to +33%
in steep basins; point runs ~+0.8 C warmer). This script measures whether that
difference changes SKILL, by running the SAME checkpoint with the SAME train
scalers over both corpora and computing the paired per-basin NSE delta.

WHY THIS IS A CLEAN PAIRED TEST (and where it is not) -- read before quoting a number
-------------------------------------------------------------------------------------
Held identical between the two arms, by construction here:
  * the checkpoint (same model_epochNNN.pt)
  * the normalisation (NH reloads `train_data/train_data_scaler.yml` from the
    run dir; BaseDataset is constructed with is_train=False, so _compute_scaler
    is False and the TRAIN statistics are reused -- it never re-fits on 2015-2025)
  * the static attributes (same camels_attrs.json, same basin subset => the
    NaN->column-mean imputation in corpus_to_nh.py is byte-identical)
  * the target: q_cfs is IDENTICAL in both corpora (verified max|dq| = 0 over 60
    basins) and q_mm = q_cfs * 2.446576 / area_gages2 uses the same area
  * the evaluation dates (see --gap-policy below)

NOT held identical, and therefore a real confound on the LEVEL (not the sign):
  * DAYMET VERSION. The training corpus is CAMELS basin_mean_forcing, built from
    the CIDA-era Daymet V2/V3. Both modern arms are Daymet V4R1. V4 is wetter.
    Measured here: areal/train precip ratio 1.077, point/train 1.011 -- i.e. the
    point corpus looks "closer" to the training climatology in precip only
    because two errors (point-dry bias, V4-wet bias) happen to cancel. Do NOT
    read a negative areal delta as "point sampling is fine"; read it as "the
    model has absorbed a V3-areal bias calibration". The version confound is
    common to both arms only in the sense that both are V4 -- it is NOT common
    to the model.
  * CALENDAR. Daymet uses a 365-day calendar (leap years end Dec 30). The AREAL
    builder fills Dec-31 of 2016/2020/2024 by copying Dec-30; the POINT corpus
    simply omits those 3 rows. corpus_to_nh.py reindexes to a full daily range,
    so POINT ends up with 3 all-NaN rows -- and NH's _validate_samples drops any
    sample whose 365-day lookback touches a NaN, which would silently delete
    ~1095 evaluation days from the POINT arm ONLY. That alone could manufacture
    a spurious delta. Handled two ways, both applied by default:
      1. --fill-gaps (default on) forward-fills forcing gaps of <=1 day in BOTH
         arms, mirroring exactly what the areal builder did. No-op on areal.
      2. --gap-policy intersect (default) computes NSE only on (basin, date)
         pairs present and finite in BOTH arms.

  * SAMPLING BIAS OF THE PARTIAL AREAL BUILD. The areal build walks basins in
    id order. Until it finishes, the available subset is HUC 01/02/03 only
    (New England / Mid-Atlantic / South Atlantic): median slope_mean 25.7 vs
    29.0 for all 531, median elev 330 m vs 437 m, and ZERO western/mountain
    basins -- exactly the basins where the point-sampling defect was measured
    at -28% to -35% precip. A number produced before the build finishes
    UNDERSTATES the effect and is not CONUS-representative. Either wait, or
    report the delta stratified by slope_mean/elev_mean.
  * SEED NOISE. One checkpoint's per-basin deltas are idiosyncratic. Repeat over
    the three complete ls-daymet seeds and average the paired deltas:
      for R in rw2ls_daymet_lstm_mm_s111_2207_111933 \
               rw2ls_daymet_lstm_mm_s222_2207_194041 \
               rw2ls_daymet_lstm_mm_s333_2307_033647; do
        gpu1080/.venv/bin/python scripts_modern/ab_point_vs_areal_nse.py \
          --run-dir gpu1080/nh_runs/$R --skip-convert \
          --work-dir data/ab_point_vs_areal --out-csv data/ab_$R.csv
      done
    (--skip-convert reuses the already-built NH dirs; only inference reruns.)

VERIFIED on a 2-basin smoke (2026-08-15), do not re-derive:
  * this script's NSE reproduces NH's own test_metrics.csv exactly
  * train_data_scaler.yml md5 is identical in both arms AND identical to the
    source run dir after evaluation => NH reused train scalers, never refit
  * model_epoch030.pt and attributes.csv md5-identical across arms
  * both arms scored 3650 paired days (2016-01-01..2025-12-31 minus the 3
    leap Dec-31s, which carry no observation)
  * cost 29 s/basin/arm at 1 CPU thread, ~17 s at 2

Usage (SMOKE first -- this is CPU-only by design; GPUs are busy):
  gpu1080/.venv/bin/python scripts_modern/ab_point_vs_areal_nse.py --limit 3 --threads 2
Full run over whatever the areal build has finished:
  gpu1080/.venv/bin/python scripts_modern/ab_point_vs_areal_nse.py --threads 2

Cost: measured ~17 s/basin/arm at 2 CPU threads (nn.LSTM h=256, seq=365,
~3650 samples/basin) => ~150 basins x 2 arms ~= 1.5 h wall.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from ruamel.yaml import YAML

ROOT = Path("/home/nakas/riverwatch2")
VENV_PY = ROOT / "gpu1080" / ".venv" / "bin" / "python"      # has torch + neuralhydrology
CONVERTER = ROOT / "scripts" / "corpus_to_nh.py"

DEFAULT_RUN = ROOT / "gpu1080/nh_runs/rw2ls_daymet_lstm_mm_s222_2207_194041"
DEFAULT_POINT = ROOT / "data/modern_corpus/camels_corpus_daymet_modern"
DEFAULT_AREAL = ROOT / "data/modern_corpus/camels_corpus_daymet_modern_areal"


# ---------------------------------------------------------------- conversion
def _load_converter():
    """Import scripts/corpus_to_nh.py as a module so BOTH arms go through the
    byte-identical conversion (q_cfs->q_mm, attrs, netCDF layout)."""
    spec = importlib.util.spec_from_file_location("corpus_to_nh", CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick_basins(c2nh, point_dir: Path, areal_dir: Path, train_dir: Path,
                limit: int) -> list[str]:
    """Basins usable by BOTH arms AND by the checkpoint (in the trained 531 and
    in the run's basin file)."""
    ids531 = set(c2nh.load_camels_531(ROOT))
    have_p = {p.name.split(".")[0] for p in point_dir.glob("*.csv.gz")
              if not p.name.startswith("._")}
    have_a = {p.name.split(".")[0] for p in areal_dir.glob("*.csv.gz")
              if not p.name.startswith("._")}
    trained = {p.stem for p in (train_dir / "time_series").glob("*.nc")}
    attrs = json.loads((ROOT / "data" / "camels_attrs.json").read_text())
    ok_area = {b for b in ids531
               if (attrs.get(b, {}).get("area_gages2") or 0) > 0}
    basins = sorted(ids531 & have_p & have_a & trained & ok_area)
    if limit:
        basins = basins[:limit]
    return basins


def convert_arm(c2nh, corpus_dir: Path, out_dir: Path, basins: list[str],
                fill_gaps: bool) -> dict:
    """corpus -> NH GenericDataset dir, restricted to `basins`.

    Uses corpus_to_nh.basin_to_nc so the q_cfs->q_mm conversion, the column
    mapping and the netCDF layout are literally the same code for both arms.
    """
    attrs = json.loads((ROOT / "data" / "camels_attrs.json").read_text())
    (out_dir / "time_series").mkdir(parents=True, exist_ok=True)
    written, n_filled, rows = [], 0, []
    for i, bid in enumerate(basins, 1):
        src = corpus_dir / f"{bid}.csv.gz"
        nc = out_dir / "time_series" / f"{bid}.nc"
        area = attrs.get(bid, {}).get("area_gages2")
        if not c2nh.basin_to_nc(src, nc, area):
            print(f"  WARN {bid}: conversion refused (missing cols/area)")
            continue
        if fill_gaps:
            n_filled += _fill_forcing_gaps(nc, c2nh.DYNAMIC_INPUTS)
        written.append(bid)
        row = {"gauge_id": bid}
        a = attrs.get(bid, {})
        for k in c2nh.STATIC_ATTRS:
            v = a.get(k)
            row[k] = float(v) if v is not None and np.isfinite(v) else np.nan
        rows.append(row)
        if i % 25 == 0:
            print(f"  [{i}/{len(basins)}] {out_dir.name}", flush=True)

    adf = pd.DataFrame(rows).set_index("gauge_id")
    adf = adf.fillna(adf.mean(numeric_only=True))
    (out_dir / "attributes").mkdir(parents=True, exist_ok=True)
    adf.to_csv(out_dir / "attributes" / "attributes.csv")
    (out_dir / "basins.txt").write_text("\n".join(written) + "\n")
    return {"basins": written, "filled_days": n_filled}


def _fill_forcing_gaps(nc: Path, dyn: list[str], limit: int = 1) -> int:
    """Forward-fill forcing gaps of <= `limit` days IN PLACE. Leaves the target
    untouched (a filled day contributes forcing history but no observation).

    Exists because the POINT corpus omits Dec-31 of leap years (Daymet's 365-day
    calendar) while the AREAL builder fills them from Dec-30. Without this, one
    NaN row poisons the 365-day lookback of ~365 samples in the POINT arm only.
    """
    ds = xr.open_dataset(nc).load()
    ds.close()
    df = ds.to_dataframe()
    before = int(df[dyn].isna().any(axis=1).sum())
    if before:
        df[dyn] = df[dyn].ffill(limit=limit)
    after = int(df[dyn].isna().any(axis=1).sum())
    if before != after:
        out = xr.Dataset(coords={"date": df.index.values})
        for c in df.columns:
            out[c] = ("date", df[c].to_numpy(dtype="float32"))
        tmp = nc.with_suffix(".tmp.nc")
        out.to_netcdf(tmp)
        tmp.replace(nc)
    return before - after


# ---------------------------------------------------------------- run dirs
def make_run_dir(src_run: Path, dst_run: Path, epoch: int, data_dir: Path,
                 basin_file: Path, start: str, end: str, device: str) -> None:
    """Minimal inference-only clone of a trained run dir.

    NH's Tester needs exactly three things from the run dir:
      config.yml, model_epochNNN.pt, train_data/train_data_scaler.yml
    We copy those and patch the config. We never touch the original run dir --
    NH writes results back into run_dir/test/, and polluting a real run dir
    would corrupt the ledger.
    """
    if dst_run.exists():
        shutil.rmtree(dst_run)
    (dst_run / "train_data").mkdir(parents=True)
    ckpt = src_run / f"model_epoch{epoch:03d}.pt"
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    scaler = src_run / "train_data" / "train_data_scaler.yml"
    if not scaler.is_file():                       # NH also accepts the old .p
        scaler = src_run / "train_data" / "train_data_scaler.p"
    if not scaler.is_file():
        raise SystemExit(f"no train scaler in {src_run}/train_data -- inference "
                         f"would have to refit statistics; refusing")
    shutil.copy2(ckpt, dst_run / ckpt.name)
    shutil.copy2(scaler, dst_run / "train_data" / scaler.name)

    yaml = YAML()
    with (src_run / "config.yml").open() as fh:
        cfg = yaml.load(fh)
    cfg["data_dir"] = str(data_dir)
    cfg["run_dir"] = str(dst_run)
    cfg["train_dir"] = str(dst_run / "train_data")
    cfg["img_log_dir"] = str(dst_run / "img_log")
    cfg["device"] = device
    for p in ("train", "validation", "test"):
        cfg[f"{p}_basin_file"] = str(basin_file)
    cfg["test_start_date"] = start
    cfg["test_end_date"] = end
    cfg["num_workers"] = 0
    cfg["metrics"] = ["NSE"]
    with (dst_run / "config.yml").open("w") as fh:
        yaml.dump(cfg, fh)


def run_eval(run_dir: Path, epoch: int, threads: int, device: str) -> Path:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), CUDA_VISIBLE_DEVICES=""
               if device == "cpu" else env.get("CUDA_VISIBLE_DEVICES", ""))
    cmd = [str(VENV_PY), "-m", "neuralhydrology.nh_run", "evaluate",
           "--run-dir", str(run_dir), "--epoch", str(epoch), "--period", "test"]
    if device == "cpu":
        cmd += ["--gpu", "-1"]
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    res = run_dir / "test" / f"model_epoch{epoch:03d}" / "test_results.p"
    if not res.is_file():
        raise SystemExit(f"evaluation produced no results at {res}")
    return res


# ---------------------------------------------------------------- scoring
def load_series(results_p: Path) -> dict[str, pd.DataFrame]:
    with results_p.open("rb") as fh:
        res = pickle.load(fh)
    out = {}
    for bid, per in res.items():
        da = per["1D"]["xr"]
        obs = da["q_mm_obs"].isel(time_step=0).to_series()
        sim = da["q_mm_sim"].isel(time_step=0)
        if "samples" in getattr(sim, "dims", ()):
            sim = sim.median(dim="samples")
        sim = sim.to_series()
        df = pd.DataFrame({"obs": obs.values, "sim": sim.values},
                          index=pd.to_datetime(obs.index))
        out[str(bid).zfill(8)] = df
    return out


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((sim - obs) ** 2) / denom)


def score(point: dict, areal: dict, gap_policy: str, min_days: int) -> pd.DataFrame:
    rows = []
    for bid in sorted(set(point) & set(areal)):
        p, a = point[bid], areal[bid]
        if gap_policy == "intersect":
            idx = p.index.intersection(a.index)
            p, a = p.loc[idx], a.loc[idx]
            ok = p.notna().all(axis=1) & a.notna().all(axis=1)
            p, a = p[ok], a[ok]
            if len(p) < min_days:
                continue
            rows.append(dict(basin=bid, n_days=len(p),
                             nse_point=nse(p.obs.values, p.sim.values),
                             nse_areal=nse(a.obs.values, a.sim.values)))
        else:  # 'independent' -- each arm scored on its own valid days
            p = p.dropna(); a = a.dropna()
            if len(p) < min_days or len(a) < min_days:
                continue
            rows.append(dict(basin=bid, n_days_point=len(p), n_days_areal=len(a),
                             nse_point=nse(p.obs.values, p.sim.values),
                             nse_areal=nse(a.obs.values, a.sim.values)))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["delta"] = df.nse_areal - df.nse_point
    return df


def report(df: pd.DataFrame, gap_policy: str) -> None:
    print("\n" + "=" * 72)
    print("A/B  AREAL vs POINT daymet forcings, same checkpoint, same train scalers")
    print("=" * 72)
    if df.empty:
        print("NO SCORABLE BASINS -- check the smoke output above.")
        return
    d = df.delta.dropna()
    print(f"basins scored          : {len(df)}")
    if "n_days" in df:
        print(f"paired days/basin      : median {df.n_days.median():.0f} "
              f"(min {df.n_days.min():.0f}, max {df.n_days.max():.0f})  "
              f"[gap-policy={gap_policy}]")
    else:
        print(f"days/basin point/areal : median {df.n_days_point.median():.0f} / "
              f"{df.n_days_areal.median():.0f}   [gap-policy={gap_policy}]")
        print("  !! unpaired day sets -- a delta here is NOT interpretable. "
              "Use --gap-policy intersect.")
    print(f"median NSE  POINT      : {df.nse_point.median():+.4f}")
    print(f"median NSE  AREAL      : {df.nse_areal.median():+.4f}")
    print(f"difference of medians  : {df.nse_areal.median() - df.nse_point.median():+.4f}"
          "   (rank statistic -- NOT the paired effect)")
    print("-" * 72)
    print(f"PAIRED delta (areal-point)")
    print(f"  median               : {d.median():+.6f}")
    print(f"  mean                 : {d.mean():+.6f}")
    print(f"  breadth (% improved) : {100.0 * (d > 0).mean():.1f}%  "
          f"({int((d > 0).sum())}/{len(d)})")
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d.values, len(d), replace=True).mean()
                     for _ in range(2000)])
    print(f"  95% CI on mean delta : [{np.percentile(boot, 2.5):+.6f}, "
          f"{np.percentile(boot, 97.5):+.6f}]")
    print("-" * 72)
    worst = df.nsmallest(5, "delta")[["basin", "nse_point", "nse_areal", "delta"]]
    best = df.nlargest(5, "delta")[["basin", "nse_point", "nse_areal", "delta"]]
    print("most helped by AREAL:\n" + best.to_string(index=False))
    print("most hurt by AREAL:\n" + worst.to_string(index=False))


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN),
                    help="trained NH run dir (daymet-only single-forcing)")
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--point-corpus", default=str(DEFAULT_POINT))
    ap.add_argument("--areal-corpus", default=str(DEFAULT_AREAL))
    ap.add_argument("--work-dir", default=str(ROOT / "data/ab_point_vs_areal"))
    ap.add_argument("--test-start", default="01/01/2016",
                    help="dd/mm/yyyy. Default leaves 2015 as the 365-day "
                         "warmup; the corpus starts 2015-01-01 so an earlier "
                         "start silently drops the first year anyway.")
    ap.add_argument("--test-end", default="31/12/2025")
    ap.add_argument("--limit", type=int, default=0, help="cap basins (smoke test)")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch CPU threads. Box has 4 cores and a training job "
                         "is using them -- keep this at 1-2.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda:0"])
    ap.add_argument("--allow-gpu", action="store_true",
                    help="required to pass --device cuda:0 (GPU jobs run here)")
    ap.add_argument("--gap-policy", default="intersect",
                    choices=["intersect", "independent"])
    ap.add_argument("--no-fill-gaps", action="store_true",
                    help="do NOT forward-fill <=1-day forcing gaps. Leaves the "
                         "POINT arm missing ~1095 days that AREAL has.")
    ap.add_argument("--min-days", type=int, default=365)
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not args.allow_gpu:
        raise SystemExit("refusing to touch the GPU: pass --allow-gpu if the "
                         "training jobs really are finished")

    src_run = Path(args.run_dir)
    work = Path(args.work_dir)
    point_dir, areal_dir = Path(args.point_corpus), Path(args.areal_corpus)
    c2nh = _load_converter()

    train_data_dir = Path(YAML(typ="safe").load(
        (src_run / "config.yml").read_text())["data_dir"])
    basins = pick_basins(c2nh, point_dir, areal_dir, train_data_dir, args.limit)
    print(f"checkpoint : {src_run.name} epoch {args.epoch}")
    print(f"basins     : {len(basins)} "
          f"(in CAMELS-531 AND point AND areal AND trained)")
    print(f"period     : {args.test_start} .. {args.test_end}  "
          f"(warmup consumes the 365 d before the start)")
    print(f"device     : {args.device} ({args.threads} threads)")
    if args.dry_run or not basins:
        return 0 if basins else 1

    arms = {"point": (point_dir, work / "nh_point", work / "run_point"),
            "areal": (areal_dir, work / "nh_areal", work / "run_areal")}
    series = {}
    for name, (corpus, data_dir, run_dir) in arms.items():
        print(f"\n--- arm: {name}")
        if not args.skip_convert:
            info = convert_arm(c2nh, corpus, data_dir, basins,
                               fill_gaps=not args.no_fill_gaps)
            print(f"  converted {len(info['basins'])} basins, "
                  f"forward-filled {info['filled_days']} forcing-gap days")
            if sorted(info["basins"]) != basins:
                raise SystemExit("arm basin sets diverged -- the attribute "
                                 "imputation would differ between arms; abort")
        make_run_dir(src_run, run_dir, args.epoch, data_dir,
                     data_dir / "basins.txt", args.test_start, args.test_end,
                     args.device)
        res = (run_dir / "test" / f"model_epoch{args.epoch:03d}" / "test_results.p")
        if not args.skip_eval or not res.is_file():
            res = run_eval(run_dir, args.epoch, args.threads, args.device)
        series[name] = load_series(res)
        print(f"  {len(series[name])} basins returned predictions")

    df = score(series["point"], series["areal"], args.gap_policy, args.min_days)
    report(df, args.gap_policy)
    out = Path(args.out_csv) if args.out_csv else work / "ab_per_basin.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nper-basin table -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
