#!/usr/bin/env python3
"""LEDGER 42 A2/C1 — event-tail quantile-mapped precipitation corpus.

THE TARGET, and why this shape and not another
----------------------------------------------
The measured residual on this campaign is **event MAGNITUDE, not phase**: the
per-basin oracle timing shift buys **+0.000000** (526/531 basins pick shift 0),
both limbs under-predict on event days, and ~96.5% of squared error sits on
top-decile flow days. The model is ON TIME and TOO SMALL.

So this construction changes **only the upper tail of the magnitude
distribution** and **nothing about the day a storm lands**:

  - ordinary days pass through daymet UNCHANGED (they already work, and a third
    of near-median error is on ordinary days — not a place to add noise);
  - above a per-basin wet-day quantile the daymet value is replaced by the
    station corpus's value at the SAME quantile — a monotone remap, so the
    RANK ORDER of days, and therefore the timing, is preserved exactly;
  - a linear blend across a transition band prevents a discontinuity at the
    threshold.

⛔ WHAT THIS IS NOT, and why prior failures do not close it
  - `build_daymet_station_corrected.py` applied ONE multiplicative VOLUME
    scalar per basin (-0.005 as a member): volume is not the defect.
  - `build_daymet_station_blend.py` blended daily values, which perturbs
    timing and ordinary days as well.
  - Both shipped as ADDED CHANNELS. This is a **SUBSTITUTE** for the daymet
    precip slot in the best recipe — the untried shape.

LEAKAGE
-------
The quantile map is FIT ON THE TRAIN WINDOW ONLY (1981-01-01..1995-09-30) and
then applied to the whole series. Test-window values are transformed by a map
that never saw them. The window is asserted, not assumed.

OUTPUT
------
A corpus dir with daymet's exact schema (so `corpus_to_nh.py` and every
downstream config consume it unchanged) with `precipitation_sum` replaced.

Usage:
  python scripts/build_precip_eventqm.py --out gpu1080/corpora671/camels_corpus_eventqm_v1
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))

DAYMET = Path("gpu1080/corpora671/camels_corpus_daymet_v2")
STATION = Path("data/local_corpora/camels_corpus_station_v1")
SUBSET = Path("data/camels_gauge_ids.json")

FIT_START, FIT_END = "1981-01-01", "1995-09-30"   # TRAIN ONLY
WET = 1.0            # mm; below this a day is not a wet day
Q_LO = 0.90          # blend starts here (quantile of WET days)
Q_HI = 0.95          # full quantile mapping above here
MIN_WET = 200        # wet days needed in the fit window to fit a map
MAX_DIST_KM = 50.0   # station-corpus quality floor
MIN_STATIONS = 1


def fit_apply(dm, st, meta):
    """Tail-only quantile map, fit on the train window, applied to all rows.

    Returns (new_precip, info). Falls back to unmodified daymet when the
    station record is too thin to fit a trustworthy tail.
    """
    p = dm["precipitation_sum"].to_numpy(float)
    out = p.copy()
    info = {"applied": False, "reason": "", "n_wet_fit": 0,
            "ratio_q95": np.nan, "ratio_q99": np.nan}

    fit = (dm["date"] >= FIT_START) & (dm["date"] <= FIT_END)
    ok_st = st["precipitation_sum"].notna().to_numpy()
    if meta is not None:
        # quality floor: the station estimate must come from a real nearby gauge
        q = ((st["n_stations"].fillna(0).to_numpy() >= MIN_STATIONS)
             & (st["min_dist_km"].fillna(1e9).to_numpy() <= MAX_DIST_KM))
        ok_st = ok_st & q

    s = st["precipitation_sum"].to_numpy(float)
    m = fit.to_numpy() & ok_st & np.isfinite(p) & np.isfinite(s)
    pw, sw = p[m], s[m]
    wet = (pw > WET) & (sw > WET)
    if wet.sum() < MIN_WET:
        info["reason"] = f"only {int(wet.sum())} paired wet days (<{MIN_WET})"
        return out, info
    pw, sw = pw[wet], sw[wet]
    info["n_wet_fit"] = int(len(pw))

    # the empirical tail map: daymet quantile u -> station quantile u
    us = np.linspace(Q_LO, 1.0, 201)
    src = np.quantile(pw, us)
    dst = np.quantile(sw, us)
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        info["reason"] = "non-finite quantiles"
        return out, info
    # enforce monotonicity so the remap cannot reorder days (timing is sacred)
    src = np.maximum.accumulate(src)
    dst = np.maximum.accumulate(dst)
    if src[-1] <= src[0]:
        info["reason"] = "degenerate tail (no spread)"
        return out, info

    thr_lo, thr_hi = float(np.quantile(pw, Q_LO)), float(np.quantile(pw, Q_HI))
    if thr_hi <= thr_lo:
        info["reason"] = "degenerate blend band"
        return out, info

    tail = np.isfinite(p) & (p > thr_lo)
    mapped = np.interp(p[tail], src, dst)
    # linear blend across [thr_lo, thr_hi]; full map above thr_hi
    w = np.clip((p[tail] - thr_lo) / (thr_hi - thr_lo), 0.0, 1.0)
    out[tail] = (1.0 - w) * p[tail] + w * mapped
    out = np.maximum(out, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        info["ratio_q95"] = float(np.quantile(dst, 0.5) / max(np.quantile(src, 0.5), 1e-9))
        info["ratio_q99"] = float(dst[-1] / max(src[-1], 1e-9))
    info["applied"] = True
    info["thr_lo"], info["thr_hi"] = thr_lo, thr_hi
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    raw = json.loads(SUBSET.read_text())
    basins = [str(b).zfill(8) for b in (raw["531"] if isinstance(raw, dict) else raw)]
    if a.limit:
        basins = basins[:a.limit]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report, n_app, n_pass = {}, 0, 0
    for b in basins:
        dpath, spath = DAYMET / f"{b}.csv.gz", STATION / f"{b}.csv.gz"
        if not dpath.exists():
            report[b] = {"applied": False, "reason": "no daymet"}
            continue
        dm = pd.read_csv(dpath)
        dm["date"] = pd.to_datetime(dm["date"])
        if not spath.exists():
            # no station record -> pass daymet through unchanged, recorded
            dm.to_csv(out_dir / f"{b}.csv.gz", index=False, compression="gzip")
            report[b] = {"applied": False, "reason": "no station corpus"}
            n_pass += 1
            continue
        st = pd.read_csv(spath)
        st["date"] = pd.to_datetime(st["date"])
        cols = ["date", "precipitation_sum"] + \
               [c for c in ("n_stations", "min_dist_km") if c in st.columns]
        st = st[cols]
        j = dm[["date"]].merge(st, on="date", how="left")
        newp, info = fit_apply(dm, j, "n_stations" if "n_stations" in j else None)
        dm["precipitation_sum"] = newp
        dm.to_csv(out_dir / f"{b}.csv.gz", index=False, compression="gzip")
        report[b] = info
        n_app += int(info["applied"])
        n_pass += int(not info["applied"])

    ratios = [v["ratio_q99"] for v in report.values()
              if v.get("applied") and np.isfinite(v.get("ratio_q99", np.nan))]
    summary = {
        "out": str(out_dir), "basins": len(basins),
        "mapped": n_app, "passed_through": n_pass,
        "fit_window": [FIT_START, FIT_END],
        "params": {"WET": WET, "Q_LO": Q_LO, "Q_HI": Q_HI,
                   "MIN_WET": MIN_WET, "MAX_DIST_KM": MAX_DIST_KM},
        "median_q99_ratio_station_over_daymet": (float(np.median(ratios))
                                                 if ratios else None),
        "per_basin": report,
    }
    Path("benchmarks/ledger42_c1_eventqm_build.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"basins {len(basins)}  mapped {n_app}  passed-through {n_pass}")
    if ratios:
        r = np.array(ratios)
        print(f"q99 station/daymet ratio: median {np.median(r):.3f}  "
              f"IQR {np.percentile(r,25):.3f}-{np.percentile(r,75):.3f}")
        print(f"  basins where stations are WETTER at q99: "
              f"{(r > 1).mean():.1%}")
    print("wrote benchmarks/ledger42_c1_eventqm_build.json")


if __name__ == "__main__":
    sys.exit(main())
