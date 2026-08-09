#!/usr/bin/env python3
"""Per-basin observed PEAK discharge — the half of the extrapolation test we own.

THE TEST
--------
A basin's high flows required rating-curve EXTRAPOLATION if its observed peak
exceeds the largest DIRECT field measurement ever made at that gauge. Coxon 2015
found this was so common in the UK that 44% of high-flow uncertainties could not
even be computed. Nobody has published the equivalent for CAMELS-531, and it is
the single number that decides whether the ~104 flashy basins' headroom is real
(see headroom-where-effort-pays).

This script produces the LEFT side of that comparison from data already on disk:
per basin, the max observed daily discharge and the flow statistics that predict
extrapolation risk. The RIGHT side (max field-measured discharge per USGS site)
needs a working USGS field-measurement endpoint.

Emits CSV so the join is trivial once the USGS side lands:
    station_id, q_max, q_p99, q_median, flashiness, n_days, cohort
"""
import argparse
import glob

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump",
                    default="gpu1080/dumps/camels531ls_multi_nhlstm_s111.csv.gz",
                    help="any TEST dump; only truth is used")
    ap.add_argument("--train-glob",
                    default="gpu1080/dumps/camels531ls_multi_nhlstm_TRAIN_s111.csv.gz",
                    help="TRAIN dump, so the peak covers the whole record")
    ap.add_argument("--out", default="basin_peaks.csv")
    a = ap.parse_args()

    frames = []
    for path in [a.dump] + glob.glob(a.train_glob):
        try:
            d = pd.read_csv(path, usecols=["station_id", "t0", "h", "truth"])
        except Exception as e:                       # noqa: BLE001
            print(f"  skip {path}: {e}")
            continue
        # h==1 gives one row per calendar day without double counting windows
        frames.append(d[d.h == 1][["station_id", "t0", "truth"]])
        print(f"  loaded {path.split('/')[-1]}: {len(frames[-1]):,} day-1 rows")

    if not frames:
        raise SystemExit("no dumps loaded")

    df = pd.concat(frames).drop_duplicates(subset=["station_id", "t0"])
    df = df.dropna(subset=["truth"])
    print(f"\ncombined: {len(df):,} unique basin-days, "
          f"{df.station_id.nunique()} basins")

    rows = []
    for sid, g in df.groupby("station_id"):
        q = g["truth"].to_numpy(float)
        q = q[np.isfinite(q)]
        if q.size < 100:
            continue
        pos = q[q > 0]
        if pos.size < 50:
            continue
        rows.append((
            sid,
            float(q.max()),
            float(np.percentile(pos, 99)),
            float(np.median(pos)),
            float(np.percentile(pos, 99) / np.median(pos)),
            int(q.size),
        ))

    out = pd.DataFrame(rows, columns=["station_id", "q_max", "q_p99",
                                      "q_median", "flashiness", "n_days"])
    thr = out.flashiness.median()
    out["cohort"] = np.where(out.flashiness > thr, "flashy", "steady")

    # peak-to-p99 ratio: how far the single largest event sits above the
    # routine high flows. A large ratio means the peak is far outside the
    # range a rating curve is likely to have been measured in.
    out["peak_over_p99"] = out.q_max / out.q_p99

    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(out)} basins)\n")

    print(f"{'cohort':<10}{'n':>5}{'med q_max':>12}{'med flashiness':>16}"
          f"{'med peak/p99':>14}")
    print("-" * 57)
    for c in ("steady", "flashy"):
        s = out[out.cohort == c]
        print(f"{c:<10}{len(s):>5}{s.q_max.median():>12.0f}"
              f"{s.flashiness.median():>16.1f}{s.peak_over_p99.median():>14.2f}")
    s = out
    print(f"{'ALL':<10}{len(s):>5}{s.q_max.median():>12.0f}"
          f"{s.flashiness.median():>16.1f}{s.peak_over_p99.median():>14.2f}")

    print("\npeak_over_p99 is the extrapolation-risk proxy we can compute WITHOUT")
    print("USGS: the higher it is, the further the record's largest event sits")
    print("above routine high flows, and the less likely a direct gauging exists")
    print("near it. The real test still needs max field-measured discharge.")


if __name__ == "__main__":
    main()
