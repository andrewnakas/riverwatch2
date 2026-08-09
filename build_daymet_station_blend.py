#!/usr/bin/env python3
"""Daymet blended DAILY with GHCN station observations, terrain-gated.

Why this and not the per-basin scalar. `build_daymet_station_corrected.py`
applies one multiplicative factor per basin, which leaves daily correlation with
daymet at exactly 1.000 -- it fixes VOLUME, not timing. Our own decomposition
says bias^2 is 0.3% of ensemble MSE while the residual event-magnitude term is
88.2%, so a volume-only correction attacks almost none of the error.

A DAILY blend can reach it. Measured on gate-eligible basins, replacing daymet
with 0.5*daymet + 0.5*station moves:

    band        mean |change|   as % of daymet   signed
    light            0.83 mm         35.4%       -0.15
    moderate         2.32 mm         21.7%       -0.83
    heavy            4.67 mm         16.2%       -0.85
    extreme          9.39 mm         13.5%       +0.47

Heavy and extreme days move by several mm -- that is the event-magnitude term.
The signed column matters too: the blend is slightly DRIER on heavy days but
WETTER on extremes, and our ensemble is known to under-predict storm peaks, so
the extreme-day direction is the helpful one.

SAME TERRAIN GATE AS THE SCALAR, for the same reason. The station/daymet ratio
tracks elevation (corr -0.337 at a 1000 m gate) because point gauges under-catch
in orographic terrain; a threshold sweep put the correction essentially
terrain-free at 400 m (corr -0.116). Blending outside that gate would inject the
53% point-vs-areal dry bias into precisely the mountain basins that dominate our
event error. Ungated basins pass through as pure daymet.

WEIGHTING. Station weight rises with confidence: more contributing stations and
a closer nearest station both raise w, capped at W_MAX so daymet always retains
a majority share on any single day. Days with no station observation fall back to
pure daymet, so coverage stays 100%.

LEAKAGE: the blend uses only observations, never discharge, and the gate is
decided on static attributes. Nothing is fit on test-period targets.
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
OUT = os.path.join(C, "camels_corpus_daymetSB_v2")

MAX_ELEV = 400.0    # tuned; see build_daymet_station_corrected.py for the sweep
MAX_SNOW = 0.15
MAX_DIST = 15.0
MIN_COVER = 0.90
W_MAX = 0.50        # station never outweighs daymet on a given day


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


def station_weight(n_stations, dist_km):
    """Confidence in the point sample: more stations and closer is better."""
    n = np.clip(np.nan_to_num(n_stations, nan=0.0), 0, 4) / 4.0      # 0..1
    d = np.clip((MAX_DIST - np.nan_to_num(dist_km, nan=MAX_DIST)) / MAX_DIST, 0, 1)
    return W_MAX * (0.5 * n + 0.5 * d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    attrs = load_attrs()
    ids = [str(x).zfill(8) for x in
           json.load(open(f"{R}/data/camels_gauge_ids.json"))["531"]]
    if not args.dry_run:
        os.makedirs(OUT, exist_ok=True)

    stats, blended = [], 0
    for b in ids:
        sp = f"{SRC}/{b}.csv.gz"
        if not os.path.exists(sp):
            continue
        dm = pd.read_csv(sp)
        dm["date"] = pd.to_datetime(dm["date"])
        reason, wmean, moved = "no_station_data", 0.0, 0.0

        stp = f"{STATION}/{b}.csv.gz"
        if os.path.exists(stp):
            st = pd.read_csv(stp, usecols=["date", "precipitation_sum",
                                           "min_dist_km", "n_stations"])
            st["date"] = pd.to_datetime(st["date"])
            cover = st.precipitation_sum.notna().mean()
            dist = st.min_dist_km.min()
            elev = attrs.at[b, "elev_mean"] if b in attrs.index else np.nan
            snow = attrs.at[b, "frac_snow"] if b in attrs.index else np.nan

            if cover < MIN_COVER:
                reason = "sparse_station_record"
            elif not (dist <= MAX_DIST):
                reason = "station_too_far"
            elif not (np.isfinite(elev) and elev < MAX_ELEV):
                reason = "high_elevation_orographic"
            elif not (np.isfinite(snow) and snow < MAX_SNOW):
                reason = "snowy_gauge_undercatch"
            else:
                m = dm.merge(st, on="date", how="left", suffixes=("", "_st"))
                w = station_weight(m.n_stations.to_numpy(),
                                   m.min_dist_km.to_numpy())
                obs = m.precipitation_sum_st.to_numpy()
                w = np.where(np.isfinite(obs), w, 0.0)   # no obs -> pure daymet
                base = m.precipitation_sum.to_numpy()
                new = (1 - w) * base + w * np.nan_to_num(obs)
                moved = float(np.mean(np.abs(new - base)))
                wmean = float(w[w > 0].mean()) if (w > 0).any() else 0.0
                dm["precipitation_sum"] = new
                reason = "blended"
                blended += 1

        stats.append(dict(basin=b, reason=reason, w_mean=wmean,
                          mean_abs_change_mm=moved,
                          ann_mm=dm.precipitation_sum.mean() * 365.25))
        if not args.dry_run:
            dm.to_csv(f"{OUT}/{b}.csv.gz", index=False, compression="gzip")

    S = pd.DataFrame(stats)
    print(f"basins written : {len(S)}")
    print(f"  BLENDED      : {blended}")
    print(f"  pure daymet  : {len(S)-blended}")
    print("\n=== why basins were left alone ===")
    print(S[S.reason != "blended"].reason.value_counts().to_string())
    bl = S[S.reason == "blended"]
    if len(bl):
        print(f"\n=== blend strength ===")
        print(f"  mean station weight   : {bl.w_mean.mean():.3f} (cap {W_MAX})")
        print(f"  mean |daily change|   : {bl.mean_abs_change_mm.mean():.3f} mm")
    print(f"\n=== units gate ===")
    print(f"  annual precip median {S.ann_mm.median():.0f} mm/yr  "
          f"range {S.ann_mm.min():.0f}-{S.ann_mm.max():.0f}")
    bad = S[(S.ann_mm < 100) | (S.ann_mm > 6000)]
    print("  " + ("all physical" if not len(bad) else f"WARNING {len(bad)} out of range"))
    if not args.dry_run:
        S.to_csv(f"{R}/benchmarks/daymetSB_blend.csv", index=False)
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
