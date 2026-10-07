#!/usr/bin/env python
"""Build AREAL GFS lead-1 decoder forcing for the day-1 forecast-decay measurement.

Registered question (decision (c)): what does day-1 skill cost when the target-day forcing is a
1-day-ahead FORECAST instead of an analysis?

Design, and why each part:
  * The 00Z run on day D is the serving setup: observations through yesterday, today's run. Its
    f006..f024 records cover calendar day D, and load_gfs_windows() maps lead_day 1 -> day D with
    issue date t0 = D-1. So day D's own forcing is a genuine 1-day-ahead forecast.
  * _parse_forcing_file() DROPS any station without all 14 leads, so each file carries 14. Lead 1 is
    the GFS forecast; leads 2-14 are the OBSERVED corpus values. The decoder is MEASURABLY CAUSAL
    (analysis/m1_decoder_causality.py: an input at lead k moves only leads >= k, exact zeros below),
    so the observed fill cannot touch the lead-1 number this measures.
  * Areal, not point-sampled. The archived data/mblstm/gfs_fcst is point-sampled at gauge
    coordinates, which runs ~53% dry versus the areal mean in mountain terrain while the whole
    corpus is areal. Weights from build_gfs_weights.py (area check median 0.14%).

⚠️ Aggregation is stated, not hidden: GFS TMP/SPFH are instantaneous at each valid hour, so the
daily mean is the mean over f006/f012/f018/f024; TMAX/TMIN are the max/min of the 6-hour bucket
extrema, which is coarser than a true daily extremum; DSWRF is a period-average flux converted to
MJ/m2/day. These are approximations of the corpus's own definitions and the registration screen
below is what decides whether they are good enough to believe.
"""
from __future__ import annotations
import argparse, gzip, io, json, os, pickle, sys, tempfile
from pathlib import Path

import numpy as np, pandas as pd
import s3fs, xarray as xr

ROOT = Path("/home/nakas/riverwatch2")
W = ROOT / "data/modern_corpus/gfs_basin_weights.pkl"
OUTDIR = ROOT / "data/mblstm/gfs_fcst_areal"
B = "noaa-gfs-bdp-pds"
HOURS = (6, 12, 18, 24)
VARS6 = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
         "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]

WANT = {  # (grib short name in the .idx, level string fragment)
    "TMP": "2 m above ground", "TMAX": "2 m above ground", "TMIN": "2 m above ground",
    "SPFH": "2 m above ground", "APCP": "surface", "DSWRF": "surface", "PRES": "surface",
}


def idx_ranges(fs, day, hour):
    p = f"{B}/gfs.{day}/00/atmos"
    base = f"gfs.t00z.pgrb2.0p25.f{hour:03d}"
    grb, idx = f"{p}/{base}", f"{p}/{base}.idx"
    try:
        lines = fs.cat(idx).decode().strip().split("\n")
    except Exception:
        return None, None
    out = []
    for i, ln in enumerate(lines):
        f = ln.split(":")
        if len(f) < 5:
            continue
        var, lev = f[3], f[4]
        if var in WANT and WANT[var] in lev:
            s = int(f[1])
            nxt = lines[i + 1].split(":")[1] if i + 1 < len(lines) else None
            if nxt is None:
                continue
            out.append((var, lev, ln, s, int(nxt) - s))
    return grb, out


def decode(fs, grb, recs):
    """-> {(var, lev_line): 2D global array}"""
    got = {}
    for var, lev, ln, start, n in recs:
        blob = fs.read_block(grb, start, n)
        if blob[:4] != b"GRIB":
            continue
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as t:
            t.write(blob); tmp = t.name
        try:
            ds = xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""})
            v = list(ds.data_vars)[0]
            got.setdefault(var, []).append((ln, np.asarray(ds[v].values, dtype="float64")))
        except Exception as e:
            print(f"    decode failed {var}: {type(e).__name__}", file=sys.stderr)
        finally:
            os.unlink(tmp)
    return got


class Reducer:
    def __init__(self):
        w = pickle.load(open(W, "rb"))
        g = w["grid"]
        self.ny, self.nx = g["ny"], g["nx"]
        self.r0, self.r1 = g["row_idx"]
        self.c0, self.c1 = g["col_idx"]
        self.row0 = g["row0"]
        self.basins = sorted(w["weights"])
        bidx, cid, cov = [], [], []
        for i, b in enumerate(self.basins):
            c, v = w["weights"][b]
            bidx.append(np.full(len(c), i, np.int32)); cid.append(c); cov.append(v)
        self.bidx = np.concatenate(bidx); self.cid = np.concatenate(cid)
        self.cov = np.concatenate(cov)
        self.wsum = np.bincount(self.bidx, weights=self.cov, minlength=len(self.basins))

    def __call__(self, field_global):
        assert field_global.shape == (721, 1440), field_global.shape
        sub = field_global[self.r0:self.r1 + 1, self.c0:self.c1 + 1]
        assert sub.shape == (self.ny, self.nx), (sub.shape, (self.ny, self.nx))
        # weights were computed in NORTH-first space; GFS decodes NORTH-first, so no flip.
        assert self.row0 == "north"
        flat = sub.ravel()
        num = np.bincount(self.bidx, weights=self.cov * flat[self.cid],
                          minlength=len(self.basins))
        return num / np.maximum(self.wsum, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-05-01")
    ap.add_argument("--end", default="2025-09-17")
    ap.add_argument("--stride", type=int, default=28, help="sample every Nth day")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    red = Reducer()
    fs = s3fs.S3FileSystem(anon=True)
    days = pd.date_range(a.start, a.end, freq="D")[::a.stride]
    if a.limit:
        days = days[:a.limit]
    print(f"{len(days)} sampled init dates, {len(red.basins)} basins -> {OUTDIR}")

    done = fail = 0
    for d in days:
        day = d.strftime("%Y%m%d")
        out = OUTDIR / f"{d.strftime('%Y-%m-%d')}.gfsraw.csv.gz"
        if out.exists():
            done += 1; continue
        acc = {}
        ok = True
        for h in HOURS:
            grb, recs = idx_ranges(fs, day, h)
            if not recs:
                ok = False; break
            acc[h] = decode(fs, grb, recs)
        if not ok or not acc:
            print(f"  {day}: unavailable"); fail += 1; continue
        try:
            rows = {}
            # precipitation: sum the 6-hour APCP buckets over the day
            pr = np.zeros(len(red.basins))
            for h in HOURS:
                cands = acc[h].get("APCP", [])
                if not cands: continue
                # prefer the SHORTEST accumulation window (the 6h bucket), not the 0-h total
                ln, arr = sorted(cands, key=lambda x: len(x[0]))[0]
                pr = pr + red(arr)
            rows["precipitation_sum"] = pr
            tmps = [red(acc[h]["TMP"][0][1]) for h in HOURS if acc[h].get("TMP")]
            assert tmps, "no TMP"
            rows["temperature_2m_mean"] = np.mean(tmps, axis=0) - 273.15
            mx = [red(acc[h]["TMAX"][0][1]) for h in HOURS if acc[h].get("TMAX")]
            mn = [red(acc[h]["TMIN"][0][1]) for h in HOURS if acc[h].get("TMIN")]
            rows["temperature_2m_max"] = (np.max(mx, axis=0) - 273.15) if mx else rows["temperature_2m_mean"]
            rows["temperature_2m_min"] = (np.min(mn, axis=0) - 273.15) if mn else rows["temperature_2m_mean"]
            sw = [red(acc[h]["DSWRF"][0][1]) for h in HOURS if acc[h].get("DSWRF")]
            rows["shortwave_radiation_sum"] = (np.mean(sw, axis=0) * 86400 / 1e6) if sw else np.full(len(red.basins), np.nan)
            q = [red(acc[h]["SPFH"][0][1]) for h in HOURS if acc[h].get("SPFH")]
            pp = [red(acc[h]["PRES"][0][1]) for h in HOURS if acc[h].get("PRES")]
            if q and pp:
                qm, pm = np.mean(q, axis=0), np.mean(pp, axis=0)
                rows["vapor_pressure"] = qm * pm / (0.622 + 0.378 * qm)
            else:
                rows["vapor_pressure"] = np.full(len(red.basins), np.nan)
            df = pd.DataFrame({"station_id": red.basins, **{k: rows[k] for k in VARS6}})
            df.to_csv(out, index=False, compression="gzip")
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(days)} ... {day}")
        except Exception as e:
            print(f"  {day}: build failed {type(e).__name__}: {e}"); fail += 1
    print(f"\ndone={done} fail={fail} -> {OUTDIR}")


if __name__ == "__main__":
    main()
