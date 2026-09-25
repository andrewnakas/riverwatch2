#!/usr/bin/env python3
"""LEDGER 45 / T2b — IS THE CCI SM PASS REAL, OR IS IT THE AVAILABILITY RAMP?

cci_combined PASSED the conjunction screen (3.94x noise). But its valid fraction
ramps 0.193 -> 0.609 across 1980-2008 (3.2x) and it has measured value breaks at
1987-08 (z=+3.21, INSIDE TRAIN) and 2002-07 (z=+2.44, INSIDE TEST).

So before believing it, ask what else could produce that partial correlation.
Diagnostic channels, screened by the SAME conjunction_screen code path:

  C_sm_vfrac    the VALID FRACTION alone -- availability, no soil-moisture values
  C_date        a pure linear time trend -- placebo; must be at the null
  C_sm_deseas   SM minus its monthly climatology, linearly detrended per basin
                -- strips the slow drift the ramp could imprint
  C_sm_comb_raw the original, re-run here as the reference point

If C_sm_vfrac or C_date scores near the SM channel, the PASS is a time artifact.
If C_sm_deseas keeps it, the information is hydrological.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path.home() / "riverwatch2"))
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path.home() / "riverwatch2" / "analysis"))
import ledger45_satellite_screen as S   # noqa: E402

ROOT = Path.home() / "riverwatch2"


def main():
    gids = sorted(json.load(open(ROOT / "data/camels_gauge_ids.json"))["531"])
    cols = ["prcp_daymet", "prcp_nldas", "prcp_maurer", "tmax_daymet",
            "tmin_daymet", "srad_daymet", "vp_daymet", "tmax_nldas",
            "srad_nldas", "vp_nldas", "q_mm"]
    print("loading forcings ...", flush=True)
    forc = S.load_forcing(gids, cols)
    val, vf = S.load_product("cci_combined")

    t = np.arange(len(val.index), dtype=float)
    t = (t - t.mean()) / t.std()
    mon = val.index.month

    chans = {}
    for g in gids:
        v = val[g]
        if v.std(skipna=True) == 0 or not np.isfinite(v.std(skipna=True)):
            continue
        df = pd.DataFrame(index=val.index)
        df["C_sm_comb_raw"] = v
        df["C_sm_vfrac"] = vf[g]
        df["C_date"] = t
        clim = v.groupby(mon).transform("mean")
        a = v - clim
        ok = np.isfinite(a.to_numpy()) & np.isfinite(t)
        if ok.sum() > 500:                    # remove the linear trend too
            b = np.polyfit(t[ok], a.to_numpy()[ok], 1)
            a = a - (b[0] * t + b[1])
        df["C_sm_deseas"] = a
        chans[g] = df

    print(f"diagnostic channels built for {len(chans)} basins")
    print("\n=== conjunction screen — SAME code path as the main screen ===")
    scr = S.conjunction_screen(chans, forc)
    out = {"question": "does the CCI SM conjunction PASS survive controls for the "
                       "3.2x availability ramp and the time trend?",
           "rows": scr}
    p = ROOT / "benchmarks/ledger45_cci_confound.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {p} md5 {hashlib.md5(p.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
