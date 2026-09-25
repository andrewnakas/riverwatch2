#!/usr/bin/env python3
"""LEDGER 42 A2/C2 — station-anchored EVENT-DAY SUBSTITUTION precipitation.

WHAT C1 GOT WRONG, MEASURED
---------------------------
C1 quantile-mapped daymet's event tail onto the station corpus's tail. It
FAILED A1-3 at event-day R² = 0.9894 vs daymet (98.7% of basins over the bar),
while the RAW station corpus PASSES at 0.7991 (22.3%).

The diagnosis is exact: a quantile map is MONOTONE, so it preserves daymet's
rank order of days. It can change how big every event is, but never WHICH event
is biggest — and the station data's genuinely new information is precisely that
per-event disagreement. C1 kept the 20% of the tail that was systematic and
threw away the part that was independent. (This also retro-explains the −0.005
volume-scalar member: a scalar is the degenerate monotone map.)

⭐ "Preserve the timing" and "preserve the rank order of magnitudes" are the
SAME operation on one channel, and only the first was ever intended.

WHAT C2 DOES INSTEAD
--------------------
On event days, take the station estimate ITSELF (not a remap of daymet), so the
construction is free to disagree with daymet about which storm was largest:

  event days   : precip <- station value, where the station record is
                 trustworthy (>= MIN_STATIONS gauges within MAX_DIST_KM and a
                 non-null value); daymet is retained where it is not
  ordinary days: daymet, UNCHANGED (they already work; a third of near-median
                 error is on ordinary days and is not a place to add noise)
  transition   : days adjacent to an event blend linearly, so a storm is not
                 spliced from two products mid-hydrograph

Event days are defined from PRECIPITATION, never from discharge — using
observed q to choose which days get corrected would leak the target into an
input channel. The threshold is a per-basin wet-day quantile of daymet precip.

⛔ Timing is NOT touched: only the VALUE on a day changes, never which day it
lands on. The measured residual is magnitude, and the per-basin oracle timing
shift is +0.000000.

LEAKAGE: thresholds are fit on the TRAIN window only and then applied
throughout. Station values are observations, not model output, so substituting
them on the test window is a forcing change, not a target read.

Usage:
  python scripts/build_precip_event_rescale.py --out gpu1080/corpora671/camels_corpus_evsub_v1
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

FIT_START, FIT_END = "1981-01-01", "1995-09-30"   # threshold fit: TRAIN ONLY
WET = 1.0
Q_EVENT = 0.90        # daymet wet-day quantile above which a day is an "event"
MIN_WET = 200
MAX_DIST_KM = 50.0
MIN_STATIONS = 1
BLEND_DAYS = 1        # +/- days around an event that blend rather than switch


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
        p = dm["precipitation_sum"].to_numpy(float)

        info = {"applied": False, "reason": "", "n_events": 0,
                "n_substituted": 0, "frac_days_changed": 0.0,
                "median_ratio_on_events": np.nan}

        if not spath.exists():
            dm.to_csv(out_dir / f"{b}.csv.gz", index=False, compression="gzip")
            report[b] = {**info, "reason": "no station corpus"}
            n_pass += 1
            continue

        st = pd.read_csv(spath)
        st["date"] = pd.to_datetime(st["date"])
        keep = ["date", "precipitation_sum"] + \
               [c for c in ("n_stations", "min_dist_km") if c in st.columns]
        j = dm[["date"]].merge(st[keep], on="date", how="left")
        s = j["precipitation_sum"].to_numpy(float)

        trust = np.isfinite(s)
        if "n_stations" in j:
            trust &= (j["n_stations"].fillna(0).to_numpy() >= MIN_STATIONS)
        if "min_dist_km" in j:
            trust &= (j["min_dist_km"].fillna(1e9).to_numpy() <= MAX_DIST_KM)

        fit = ((dm["date"] >= FIT_START) & (dm["date"] <= FIT_END)).to_numpy()
        wet_fit = p[fit & np.isfinite(p)]
        wet_fit = wet_fit[wet_fit > WET]
        if wet_fit.size < MIN_WET:
            dm.to_csv(out_dir / f"{b}.csv.gz", index=False, compression="gzip")
            report[b] = {**info, "reason": f"only {wet_fit.size} wet days"}
            n_pass += 1
            continue

        thr = float(np.quantile(wet_fit, Q_EVENT))
        ev = np.isfinite(p) & (p > thr)
        info["n_events"] = int(ev.sum())

        # weight 1.0 on an event day, tapering over BLEND_DAYS on either side
        w = ev.astype(float)
        for k in range(1, BLEND_DAYS + 1):
            f = 1.0 - k / (BLEND_DAYS + 1.0)
            w[k:] = np.maximum(w[k:], f * ev[:-k])
            w[:-k] = np.maximum(w[:-k], f * ev[k:])
        w[~trust] = 0.0                       # never substitute untrusted values

        out = np.where(np.isfinite(s), (1 - w) * p + w * s, p)
        out = np.maximum(out, 0.0)
        changed = np.abs(out - p) > 1e-9
        info["n_substituted"] = int(changed.sum())
        info["frac_days_changed"] = float(changed.mean())
        m = ev & trust & (p > 0)
        if m.sum() > 10:
            info["median_ratio_on_events"] = float(np.median(s[m] / p[m]))
        info["applied"] = bool(changed.any())
        info["threshold_mm"] = thr

        dm["precipitation_sum"] = out
        dm.to_csv(out_dir / f"{b}.csv.gz", index=False, compression="gzip")
        report[b] = info
        n_app += int(info["applied"])
        n_pass += int(not info["applied"])

    fr = [v["frac_days_changed"] for v in report.values() if v.get("applied")]
    ra = [v["median_ratio_on_events"] for v in report.values()
          if v.get("applied") and np.isfinite(v.get("median_ratio_on_events", np.nan))]
    summary = {"out": str(out_dir), "basins": len(basins),
               "substituted": n_app, "passed_through": n_pass,
               "fit_window": [FIT_START, FIT_END],
               "params": {"WET": WET, "Q_EVENT": Q_EVENT, "MIN_WET": MIN_WET,
                          "MAX_DIST_KM": MAX_DIST_KM, "BLEND_DAYS": BLEND_DAYS},
               "median_frac_days_changed": float(np.median(fr)) if fr else None,
               "median_station_over_daymet_on_events": float(np.median(ra)) if ra else None,
               "per_basin": report}
    Path("benchmarks/ledger42_c2_evsub_build.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"basins {len(basins)}  substituted {n_app}  passed-through {n_pass}")
    if fr:
        print(f"median fraction of days changed: {np.median(fr):.3%}")
    if ra:
        print(f"median station/daymet ratio ON EVENT DAYS: {np.median(ra):.3f}")
    print("wrote benchmarks/ledger42_c2_evsub_build.json")


if __name__ == "__main__":
    sys.exit(main())
