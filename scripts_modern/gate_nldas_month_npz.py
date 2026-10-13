#!/usr/bin/env python3
"""Artifact gate for an NLDAS-2 areal monthly checkpoint.

⚠️ LIVES IN THE REPO, NOT /tmp. The first version of this sat in /tmp and
gpu1080/chain_gfs14.sh referenced it by that path; a reboot cleared /tmp and the
chain's gate step would have failed on its next run, reporting a gate problem
where there was only a missing file. "/tmp is cleared by reboot -- keep helper
scripts in the repo" was already a recorded lesson here and I repeated it.

BOUNDS ARE CALIBRATED against the verified 1980-2025 months of this same
corpus (120 months sampled), not invented. The first version used a 50 Pa floor
for vapour pressure and REJECTED a perfectly good January: at a basin minimum
of -32.7 degC the saturation vapour pressure is about 39 Pa, and the verified
corpus itself reaches 25.8 Pa. A gate whose threshold is tighter than the data
it validates reports a defect that is not there.

Observed over the verified months:
  precip 0 .. 308.96    tmean -34.3 .. 37.8   tmax -30.2 .. 47.5
  tmin  -38.2 .. 30.2   srad    1.06 .. 37.74  vp    25.8 .. 3605.2

The bounds below sit outside those with headroom, so they still catch a
latitude flip (wrong cells on every basin), a unit error, or an accumulation
double-count, without rejecting real weather.
"""
import sys

import numpy as np

BOUNDS = {
    "precip": (0.0, 500.0),
    "tmean": (-60.0, 60.0),
    "tmax": (-60.0, 60.0),
    "tmin": (-60.0, 60.0),
    "srad": (0.0, 45.0),
    "vp": (10.0, 7000.0),
}


def main() -> int:
    path = sys.argv[1]
    ndays_expected = int(sys.argv[2]) if len(sys.argv) > 2 else None
    d = np.load(path, allow_pickle=True)
    b, days = d["basins"], d["days"]
    print("%s: %d basins, %d days, %s -> %s"
          % (path.split("/")[-1], len(b), len(days), days[0], days[-1]))
    ok = True
    for k, (lo, hi) in BOUNDS.items():
        a = d[k]
        mn, mx, nn = float(np.nanmin(a)), float(np.nanmax(a)), int(np.isnan(a).sum())
        good = (mn >= lo and mx <= hi and nn == 0)
        ok = ok and good
        if not good:
            print("  %-7s min %9.3f max %9.3f nan %d  *** OUTSIDE [%g, %g]"
                  % (k, mn, mx, nn, lo, hi))
    order = bool((d["tmin"] <= d["tmean"]).all() and (d["tmean"] <= d["tmax"]).all())
    if not order:
        print("  *** tmin <= tmean <= tmax VIOLATED")
    nb = len(b) == 671
    if not nb:
        print("  *** expected 671 basins, got %d" % len(b))
    nd = True if ndays_expected is None else len(days) == ndays_expected
    if not nd:
        print("  *** expected %d days, got %d" % (ndays_expected, len(days)))
    verdict = ok and order and nb and nd
    print("GATE:", "PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
