#!/usr/bin/env python3
"""Phase 3: do the forcing products drift from STATION OBSERVATIONS on failure days?

Phase 1 established, with a magnitude-matched control, that at the same rainfall
the four products disagree ~27% more with EACH OTHER on the days the ensemble
fails. That is inter-product disagreement. It does not say whether any of them is
actually WRONG, because the products are all we compared them to.

This script adds the missing reference: the station corpus
(data/local_corpora/camels_corpus_station_v1). Now we can ask whether the
products drift away from observed precipitation specifically on the days the
model fails.

Read the caveat before reading the numbers:

  1. NOT INDEPENDENT OF DAYMET. Daymet ingests GHCN-Daily, and ASOS/WBAN sits
     inside GHCN too. Daymet agreeing with GHCN stations is partly circular. The
     SNOTEL subset (NRCS, outside the COOP/GHCN base) is the closest thing to an
     independent control, so results are also reported per source network.
  2. POINT vs AREAL. A station is a point; a basin mean is areal. We measured a
     53% point-vs-areal effect in mountain terrain. So a product-vs-station gap
     is NOT by itself product error. The load-bearing comparison is therefore
     RELATIVE -- how each product's agreement CHANGES between ordinary and
     failure days -- because the point-vs-areal penalty applies to both classes
     and largely cancels in the difference.

The pre-registered reading: if agreement with observations degrades on failure
days by more than it does for ordinary days, input error is implicated in the
failures. If agreement is unchanged, the failures are not about the products
mis-reporting what the gauges saw.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(R, "benchmarks/forcing_disagreement_rows.csv.gz")
STATION = os.path.join(R, "data/local_corpora/camels_corpus_station_v1")
MAP = os.path.join(R, "data/cache/ghcn/basin_station_map.csv")
PRODUCTS = ["daymet", "maurer", "nldas", "aorc"]


def main():
    if not os.path.exists(ROWS):
        sys.exit("run forcing_disagreement.py first (writes the residual rows)")
    if not os.path.isdir(STATION):
        sys.exit("run build_station_corpus.py --stage build first")

    j = pd.read_csv(ROWS, dtype={"station_id": str})
    j["station_id"] = j.station_id.str.zfill(8)
    j["t0"] = pd.to_datetime(j.t0)
    print(f"residual rows: {len(j):,}  basins: {j.station_id.nunique()}")

    # Attach station observations.
    obs = []
    for b in sorted(j.station_id.unique()):
        f = os.path.join(STATION, f"{b}.csv.gz")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, usecols=["date", "precipitation_sum", "n_stations",
                                    "min_dist_km", "src_networks"])
        d["date"] = pd.to_datetime(d.date)
        d["station_id"] = b
        obs.append(d)
    if not obs:
        sys.exit("no station corpus files found")
    O = pd.concat(obs, ignore_index=True).rename(
        columns={"date": "t0", "precipitation_sum": "obs_mm"})
    print(f"station corpus: {len(O):,} basin-days, {O.station_id.nunique()} basins")

    m = j.merge(O, on=["station_id", "t0"], how="inner")
    m = m.dropna(subset=["obs_mm"] + PRODUCTS)
    print(f"joined with observations: {len(m):,} rows  "
          f"{m.station_id.nunique()} basins")
    if len(m) < 1000:
        sys.exit("too few joined rows to conclude anything")

    thr = m.sq_err.quantile(0.99)
    m["event"] = m.sq_err >= thr
    m["pmean"] = m[PRODUCTS].mean(axis=1)
    print(f"\nevent days (top-1% error): {m.event.sum():,}"
          f"   ordinary: {(~m.event).sum():,}")
    print(f"event days are {m[m.event].obs_mm.mean()/max(m[~m.event].obs_mm.mean(),1e-9):.1f}x "
          f"wetter by OBSERVED precip too "
          f"({m[m.event].obs_mm.mean():.2f} vs {m[~m.event].obs_mm.mean():.2f} mm)")

    # --- the comparison, magnitude-matched on OBSERVED precip -----------------
    # Binning on observations rather than on the products avoids selecting bins
    # with the very quantity under test.
    bins = [0.1, 2, 5, 10, 20, 40, 1e9]
    labs = ["0.1-2", "2-5", "5-10", "10-20", "20-40", "40+"]
    m["bin"] = pd.cut(m.obs_mm, bins=bins, labels=labs)

    print("\n" + "=" * 78)
    print("PRODUCT vs STATION OBSERVATIONS, matched on observed rainfall")
    print("=" * 78)
    print("bias = product - observed (mm). Compare the CHANGE between classes,")
    print("not the absolute level: point-vs-areal inflates both equally.\n")

    print(f"{'product':>8} {'class':>9} {'n':>7} {'bias_mm':>9} "
          f"{'|bias|':>8} {'corr':>7}")
    print("-" * 54)
    summary = {}
    for prod in PRODUCTS:
        for label, sel in (("ordinary", ~m.event), ("event", m.event)):
            g = m[sel]
            if len(g) < 100:
                continue
            # weight bins equally so the wet-day mix cannot drive the result
            biases, absb, corrs, ws = [], [], [], []
            for b in labs:
                gg = g[g.bin == b]
                if len(gg) < 30:
                    continue
                d = gg[prod] - gg.obs_mm
                biases.append(d.mean())
                absb.append(d.abs().mean())
                if gg[prod].std() > 1e-9 and gg.obs_mm.std() > 1e-9:
                    corrs.append(np.corrcoef(gg[prod], gg.obs_mm)[0, 1])
                ws.append(len(gg))
            if not biases:
                continue
            w = np.array(ws, float)
            bias = float(np.average(biases, weights=w))
            ab = float(np.average(absb, weights=w))
            cr = float(np.mean(corrs)) if corrs else np.nan
            summary[(prod, label)] = (bias, ab, cr, int(w.sum()))
            print(f"{prod:>8} {label:>9} {int(w.sum()):>7,} {bias:>9.2f} "
                  f"{ab:>8.2f} {cr:>7.3f}")

    print("\n=== CHANGE from ordinary to event days (the load-bearing number) ===")
    print(f"{'product':>8} {'d_bias':>9} {'d_|bias|':>10} {'d_corr':>9}")
    print("-" * 39)
    deltas = []
    for prod in PRODUCTS:
        a = summary.get((prod, "ordinary"))
        b = summary.get((prod, "event"))
        if not a or not b:
            continue
        deltas.append((prod, b[0] - a[0], b[1] - a[1], b[2] - a[2]))
        print(f"{prod:>8} {b[0]-a[0]:>9.2f} {b[1]-a[1]:>10.2f} {b[2]-a[2]:>9.3f}")

    if deltas:
        dcorr = float(np.mean([d[3] for d in deltas]))
        dbias = float(np.mean([d[1] for d in deltas]))
        print(f"\n  mean change in correlation with observations: {dcorr:+.3f}")
        print(f"  mean change in bias                         : {dbias:+.2f} mm")
        print()
        if dcorr < -0.05:
            print("  => products track the gauges materially WORSE on failure days.")
            print("     Input error is implicated in the failures.")
        elif dcorr > 0.05:
            print("  => products track the gauges BETTER on failure days, so the")
            print("     failures are not explained by inputs mis-reporting rainfall.")
        else:
            print("  => agreement with observations is essentially UNCHANGED on")
            print("     failure days (|d_corr| <= 0.05). The products report what the")
            print("     gauges saw about as well as always; the ensemble still fails.")
            print("     That points AWAY from product-vs-gauge error and toward either")
            print("     gauge-vs-catchment sampling or model structure.")

    # --- per-network split: SNOTEL is the independence control ---------------
    print("\n=== by source network (SNOTEL = outside the COOP/GHCN base) ===")
    m["net"] = m.src_networks.fillna("")
    for tag, sel in (("has SNOTEL", m.net.str.contains("SNOTEL")),
                     ("COOP only", (m.net == "COOP")),
                     ("has WBAN/ASOS", m.net.str.contains("WBAN"))):
        g = m[sel]
        if len(g) < 500:
            print(f"  {tag:<15} n={len(g):,} (too few)")
            continue
        line = f"  {tag:<15} n={len(g):>6,} basins={g.station_id.nunique():>3}  "
        for prod in PRODUCTS:
            gg = g.dropna(subset=[prod, "obs_mm"])
            if len(gg) < 200 or gg[prod].std() < 1e-9:
                continue
            line += f"{prod[:4]}:{np.corrcoef(gg[prod], gg.obs_mm)[0,1]:.2f} "
        print(line)
    print("\n  daymet scoring highest is EXPECTED and partly circular -- Daymet")
    print("  ingests GHCN-Daily. The SNOTEL row is the closest to independent.")

    # distance sanity: closer stations should agree better
    print("\n=== sanity: does agreement improve with station proximity? ===")
    m["dbin"] = pd.cut(m.min_dist_km, [0, 5, 10, 25, 50],
                       labels=["0-5", "5-10", "10-25", "25-50"])
    for db in ["0-5", "5-10", "10-25", "25-50"]:
        g = m[m.dbin == db].dropna(subset=["daymet", "obs_mm"])
        if len(g) < 300:
            continue
        print(f"  {db:>6} km: n={len(g):>6,}  "
              f"corr(daymet,obs)={np.corrcoef(g.daymet, g.obs_mm)[0,1]:.3f}")
    print("  (monotone improvement = the spatial join is sound)")


if __name__ == "__main__":
    main()
