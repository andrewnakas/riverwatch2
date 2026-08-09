#!/usr/bin/env python3
"""Daymet corrected toward GHCN station observations -- but only where the
stations are actually the better estimator.

The idea: daymet gives full catchment coverage; the station corpus gives real
observations. Blending should give both. The measurements support it -- daily
corr(station, daymet) is 0.842, daymet runs 7.7% wetter on average, 48 of 147
basins differ by >10% in volume, and the per-basin ratio has sd 0.092, so the
disagreement is systematic (correctable) rather than noise.

THE TRAP THIS CODE EXISTS TO AVOID. The daymet/station ratio correlates +0.506
with elev_mean and +0.543 with frac_snow: daymet reads wetter than the stations
specifically in high, snowy terrain. That is exactly where point sampling was
measured to run 53% dry versus a catchment areal mean. So in those basins the
STATIONS are the biased estimator, and "correcting" daymet toward them would
inject the point-sampling bias into the orographic basins that carry most of our
event error. The ratio also does not improve with proximity (1.070 at <=5 km vs
1.064 at >15 km), confirming orography rather than a distance artifact.

So the correction is TERRAIN-GATED: applied only where the point sample is
credible (low elevation, low snow fraction, a nearby station, good coverage),
and deliberately skipped elsewhere. Ungated basins pass through as pure daymet,
which keeps 100% catchment coverage and full 531-basin parity with the other
corpora.

LEAKAGE: the scale factor is fit on the TRAIN period only (Li/Song
1980-10-01..1995-09-30). Test-period observations are never touched.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(R, "data/local_corpora")
SRC = os.path.join(C, "camels_corpus_daymet_v2")
STATION = os.path.join(C, "camels_corpus_station_v1")
OUT = os.path.join(C, "camels_corpus_daymetSC_v2")

# Li/Song training window -- the only period the correction may see.
TRAIN_LO, TRAIN_HI = "1980-10-01", "1995-09-30"

# Gate: where a 1-2 station point sample can be trusted to represent the basin.
#
# MAX_ELEV was TUNED, not guessed. At a 1000 m gate the surviving correction
# still tracked terrain (corr(ratio, elev) = -0.337), and the residual gradient
# was monotone inside the gate: mean station/daymet ratio 0.964 at 0-300 m,
# 0.936 at 300-600 m, 0.883 at 600-1000 m -- i.e. orographic undercatch is
# already biting well below 1000 m. Sweeping the threshold:
#   1000 m -> n=187, corr -0.337     500 m -> n=148, corr -0.246
#    800 m -> n=177, corr -0.398     400 m -> n=125, corr -0.087  <- chosen
#    600 m -> n=155, corr -0.283     300 m -> n= 86, corr -0.276  (small-n noise)
# 400 m is the largest gate where the correction is essentially terrain-free,
# and sd 0.067 shows real gauge-vs-grid signal still survives.
MAX_ELEV = 400.0       # m; above this, orographic undercatch dominates
MAX_SNOW = 0.15        # frac_snow; frozen precip is badly measured by gauges
MAX_DIST = 15.0        # km to the nearest contributing station
MIN_COVER = 0.90       # fraction of days with a station observation
# Clamp so one bad basin cannot blow up its forcing.
RATIO_LO, RATIO_HI = 0.75, 1.35


def load_attrs():
    a = json.load(open(f"{R}/data/camels_attrs.json"))
    recs = a["basins"] if isinstance(a, dict) and "basins" in a else a
    if isinstance(recs, dict):
        df = pd.DataFrame([{"basin": str(k).zfill(8), **v} for k, v in recs.items()])
    else:
        df = pd.DataFrame(recs)
        idc = next((c for c in df.columns
                    if c.lower() in ("basin", "station_id", "gauge_id", "id")), None)
        df["basin"] = df[idc].astype(str).str.zfill(8)
    for c in ("elev_mean", "frac_snow"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("basin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    attrs = load_attrs()
    ids = [str(x).zfill(8) for x in
           json.load(open(f"{R}/data/camels_gauge_ids.json"))["531"]]
    if not args.dry_run:
        os.makedirs(OUT, exist_ok=True)

    stats, applied, passthrough, skipped = [], 0, 0, 0
    for b in ids:
        sp = f"{SRC}/{b}.csv.gz"
        if not os.path.exists(sp):
            skipped += 1
            continue
        dm = pd.read_csv(sp)
        dm["date"] = pd.to_datetime(dm["date"])

        ratio, reason = 1.0, "no_station_data"
        stp = f"{STATION}/{b}.csv.gz"
        if os.path.exists(stp):
            st = pd.read_csv(stp, usecols=["date", "precipitation_sum",
                                           "min_dist_km", "n_stations"])
            st["date"] = pd.to_datetime(st["date"])
            cover = st.precipitation_sum.notna().mean()
            dist = st.min_dist_km.min()
            elev = attrs.at[b, "elev_mean"] if b in attrs.index else np.nan
            snow = attrs.at[b, "frac_snow"] if b in attrs.index else np.nan

            # Terrain gate -- the whole point of this script.
            if not (cover >= MIN_COVER):
                reason = "sparse_station_record"
            elif not (dist <= MAX_DIST):
                reason = "station_too_far"
            elif not (np.isfinite(elev) and elev < MAX_ELEV):
                reason = "high_elevation_orographic"
            elif not (np.isfinite(snow) and snow < MAX_SNOW):
                reason = "snowy_gauge_undercatch"
            else:
                # Fit on TRAIN PERIOD ONLY.
                m = dm[["date", "precipitation_sum"]].merge(
                    st[["date", "precipitation_sum"]], on="date",
                    suffixes=("_dm", "_st")).dropna()
                m = m[(m.date >= TRAIN_LO) & (m.date <= TRAIN_HI)]
                if len(m) < 1000 or m.precipitation_sum_st.sum() <= 0:
                    reason = "insufficient_train_overlap"
                else:
                    raw = m.precipitation_sum_st.sum() / m.precipitation_sum_dm.sum()
                    ratio = float(np.clip(raw, RATIO_LO, RATIO_HI))
                    reason = "corrected" if abs(ratio - 1) > 1e-9 else "ratio_unity"

        if reason == "corrected":
            dm["precipitation_sum"] = dm.precipitation_sum * ratio
            applied += 1
        else:
            passthrough += 1

        stats.append(dict(basin=b, ratio=ratio, reason=reason,
                          ann_mm=dm.precipitation_sum.mean() * 365.25))
        if not args.dry_run:
            dm.to_csv(f"{OUT}/{b}.csv.gz", index=False, compression="gzip")

    S = pd.DataFrame(stats)
    print(f"basins written   : {len(S)} (skipped {skipped})")
    print(f"  CORRECTED      : {applied}")
    print(f"  pure daymet    : {passthrough}")
    print("\n=== why basins were left alone ===")
    print(S[S.reason != "corrected"].reason.value_counts().to_string())

    cor = S[S.reason == "corrected"]
    if len(cor):
        print(f"\n=== correction factors (station/daymet, train period) ===")
        print(f"  mean {cor.ratio.mean():.4f}  sd {cor.ratio.std():.4f}  "
              f"min {cor.ratio.min():.3f}  max {cor.ratio.max():.3f}")
        print(f"  basins scaled DOWN (<1): {(cor.ratio<1).sum()}  "
              f"UP (>1): {(cor.ratio>1).sum()}")
        at = cor.set_index("basin").join(attrs[["elev_mean", "frac_snow"]])
        for c in ("elev_mean", "frac_snow"):
            v = at[c].astype(float)
            ok = v.notna()
            if ok.sum() > 20:
                r = np.corrcoef(at.ratio[ok], v[ok])[0, 1]
                print(f"  corr(ratio, {c:10s}) = {r:+.3f}  "
                      f"{'<- should be WEAK after gating' if abs(r)<0.3 else '<- STILL TERRAIN-TRACKING, investigate'}")

    print(f"\n=== units gate ===")
    print(f"  annual precip median {S.ann_mm.median():.0f} mm/yr  "
          f"range {S.ann_mm.min():.0f}-{S.ann_mm.max():.0f}")
    bad = S[(S.ann_mm < 100) | (S.ann_mm > 6000)]
    print("  " + ("all physical" if not len(bad) else f"WARNING {len(bad)} basins out of range"))
    if not args.dry_run:
        S.to_csv(f"{R}/benchmarks/daymetSC_corrections.csv", index=False)
        print(f"\nwrote {OUT} and benchmarks/daymetSC_corrections.csv")


if __name__ == "__main__":
    main()
