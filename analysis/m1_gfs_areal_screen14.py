#!/usr/bin/env python3
"""REGISTRATION + SKILL SCREEN for the 14-lead areal GFS forcing, per LEAD DAY.

The lead-1 screen (`m1_gfs_areal_screen.py`) answers "is the forecast-day
bookkeeping right at lead 1". With 14 leads built from four 6-hour buckets each,
that question has to be asked fourteen times, because a bucket-window error at
one lead is invisible at the others -- and would masquerade as forecast error,
which is exactly what a decay measurement would then mistake for signal.

A one-day precip label offset was worth +0.229 NSE in this campaign and was
invisible to every mean/bias check, so the same two tests apply at every lead:

 1. LAG SCAN against the corpus's OBSERVED precipitation on the same basins and
    target dates. ⭐ Alignment is the one property that must NOT decay with
    lead: skill falls away, but a forecast for day D must still correlate best
    with the OBSERVED day D at every lead. A peak at ±1 anywhere is a
    bookkeeping error wearing forecast error's clothes.
 2. MAGNITUDE per lead. A 300 mm/day basin maximum is what the earlier
    bucket double-counting produced.

Plus two tests the lead-1 screen cannot express:
 3. r at k=0 must be non-increasing in lead (tolerance +0.02). A longer lead
    that BEATS a shorter one is a bookkeeping error, not skill.
 4. The spatial sd per lead must differ. Identical sds would mean one lead's
    field was broadcast into another's rows.

CONVENTION, from the lead-1 builder: a 00z run on init date D has lead 1 =
calendar day D, so lead L covers D + (L-1).

Registered band in PREREG_v2.md GATE 4's "what has already been validated"
section; this script is what validates it.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

GFS = "data/mblstm/gfs_fcst_areal14"
REF = "data/modern_corpus/v3/camels_corpus_gridmetfix3_m3"
VARS = ["precipitation_sum", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "shortwave_radiation_sum", "vapor_pressure"]
KS = (-2, -1, 0, 1, 2)

#: Registered thresholds. The lead-1 screen's own passing values are the
#: anchor: precip best_k 0, ratio 0.869, r at 0 >= 0.5.
RATIO_LO, RATIO_HI = 0.7, 1.4
R0_MIN = 0.5
MONOTONE_TOL = 0.02
#: temperature_2m_min peaks at k=+1 in the lead-1 screen and that deviation is
#: already registered: UTC 6-hour TMIN buckets do not align with the corpus's
#: local-day minimum. Carried forward rather than silently re-passed.
TMIN_ALLOWED_K = {0, 1}


def main() -> int:
    files = sorted(glob.glob(f"{GFS}/*.csv.gz"))
    if not files:
        print(f"no files in {GFS}", file=sys.stderr)
        return 1
    print(f"{len(files)} GFS init dates")

    rows = []
    for f in files:
        d = os.path.basename(f).split(".")[0]
        g = pd.read_csv(f, dtype={"station_id": str})
        g["init"] = pd.to_datetime(d, format="%Y%m%d")
        rows.append(g)
    G = pd.concat(rows, ignore_index=True)
    G["station_id"] = G.station_id.str.zfill(8)
    # lead L covers init + (L-1)
    G["target"] = G["init"] + pd.to_timedelta(G["lead_day"] - 1, unit="D")
    print(f"{len(G):,} (basin, init, lead) rows, {G.station_id.nunique()} basins, "
          f"leads {sorted(G.lead_day.unique())[0]}..{sorted(G.lead_day.unique())[-1]}")

    # observed, once, for every basin we need
    obs = {}
    for sid in sorted(G.station_id.unique()):
        p = f"{REF}/{sid}.csv.gz"
        if not os.path.exists(p):
            continue
        o = pd.read_csv(p, usecols=["date"] + VARS)
        o["date"] = pd.to_datetime(o["date"])
        obs[sid] = o.set_index("date")
    print(f"observed loaded for {len(obs)} basins")
    if not obs:
        print(f"no observed corpus at {REF}", file=sys.stderr)
        return 1

    out = {"n_dates": len(files), "n_basins": int(G.station_id.nunique()), "by_lead": {}}
    fails = []
    print(f"\n{'lead':>4} {'var':<24}{'best_k':>7}{'r@0':>8}{'ratio':>8}{'gfs_max':>10}{'obs_max':>10}")
    for L in sorted(G.lead_day.unique()):
        gl = G[G.lead_day == L]
        lead_rec = {}
        for v in VARS:
            # paired series per lag
            per_k = {k: ([], []) for k in KS}
            for sid, grp in gl.groupby("station_id"):
                o = obs.get(sid)
                if o is None:
                    continue
                for k in KS:
                    idx = grp["target"] + pd.to_timedelta(k, unit="D")
                    ov = o[v].reindex(idx).to_numpy(dtype=float)
                    gv = grp[v].to_numpy(dtype=float)
                    m = np.isfinite(ov) & np.isfinite(gv)
                    if m.sum() >= 3:
                        per_k[k][0].append(gv[m])
                        per_k[k][1].append(ov[m])
            rs = {}
            for k in KS:
                a, b = per_k[k]
                if not a:
                    continue
                A, B = np.concatenate(a), np.concatenate(b)
                if A.std() > 0 and B.std() > 0:
                    rs[k] = float(np.corrcoef(A, B)[0, 1])
            if not rs:
                continue
            best_k = max(rs, key=rs.get)
            A = np.concatenate(per_k[0][0]) if per_k[0][0] else np.array([np.nan])
            B = np.concatenate(per_k[0][1]) if per_k[0][1] else np.array([np.nan])
            ratio = float(A.mean() / B.mean()) if B.mean() else float("nan")
            lead_rec[v] = {"best_k": int(best_k), "r": rs,
                           "r0": rs.get(0), "ratio": ratio,
                           "gfs_mean": float(A.mean()), "obs_mean": float(B.mean()),
                           "gfs_max": float(A.max()), "obs_max": float(B.max())}
            print(f"{L:>4} {v:<24}{best_k:>7}{rs.get(0, float('nan')):>8.3f}"
                  f"{ratio:>8.3f}{A.max():>10.2f}{B.max():>10.2f}")

            # --- the registered checks
            allowed = TMIN_ALLOWED_K if v == "temperature_2m_min" else {0}
            if best_k not in allowed:
                fails.append(f"lead {L} {v}: best_k={best_k}, allowed {sorted(allowed)}")
            if v == "precipitation_sum":
                if not (RATIO_LO <= ratio <= RATIO_HI):
                    fails.append(f"lead {L} precip ratio {ratio:.3f} outside "
                                 f"[{RATIO_LO},{RATIO_HI}]")
                if rs.get(0, 0) < R0_MIN and L == 1:
                    fails.append(f"lead 1 precip r@0 {rs.get(0)} < {R0_MIN}")
                if A.max() > 1.3 * B.max():
                    fails.append(f"lead {L} precip max {A.max():.1f} > 1.3x observed "
                                 f"{B.max():.1f}")
        out["by_lead"][int(L)] = lead_rec

    # --- r must not IMPROVE with lead
    r0s = {L: (rec.get("precipitation_sum") or {}).get("r0")
           for L, rec in out["by_lead"].items()}
    seq = [r0s[L] for L in sorted(r0s) if r0s[L] is not None]
    for i in range(1, len(seq)):
        if seq[i] > seq[i - 1] + MONOTONE_TOL:
            fails.append(f"precip r@0 RISES from lead {i} ({seq[i-1]:.3f}) to "
                         f"{i+1} ({seq[i]:.3f}) by more than {MONOTONE_TOL}")
    out["precip_r0_by_lead"] = seq

    # --- no lead may be a broadcast of another
    sds = G.groupby("lead_day").precipitation_sum.std().round(6)
    out["precip_spatial_sd_by_lead"] = {int(k): float(v) for k, v in sds.items()}
    if len(set(sds.values)) != len(sds):
        fails.append("two leads share an identical precip spatial sd -- one may "
                     "be a broadcast of another")

    out["failures"] = fails
    out["screen_passes"] = not fails
    print("\n" + "=" * 78)
    if fails:
        print(f"⛔ SCREEN FAILS ({len(fails)}):")
        for f in fails[:12]:
            print(f"   - {f}")
        print("   Do NOT report any per-lead number from this build until resolved.")
    else:
        print("✅ SCREEN PASSES at every lead: alignment peaks at k=0, magnitudes sane,")
        print("   r does not improve with lead, and no lead duplicates another.")
    json.dump(out, open("benchmarks/m1_gfs_areal_screen14.json", "w"), indent=2)
    print("-> benchmarks/m1_gfs_areal_screen14.json")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
