"""GATE 4 STOP re-check: compare each arm's DECODER FORCING with the areal NLDAS-2
rows of the encoder table (truth-like), at target-date shifts -1/0/+1. No discharge."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "analysis"); sys.path.insert(0, ".")
import gate4_run as G
from app import camels_attrs
coh = set(camels_attrs.cohort())
inits = sorted(Path(p).name[:10] for p in glob.glob("data/gate4/omarch/**/*.csv.gz", recursive=True))
om = pd.concat([G._om_arm("data/gate4/omarch", d).assign(init=d) for d in inits])
gf = pd.concat([G._gfs_arm("data/gate4/gfs15", d).assign(init=d) for d in inits])
for a in (om, gf):
    a["lead"] = (a.date - pd.to_datetime(a.init)).dt.days
truth = []
for sid in sorted(set(om.station_id) & coh):
    p = Path(f"data/gate4/tables/{sid}.csv.gz")
    if p.exists():
        t = pd.read_csv(p, parse_dates=["date"]); t = t[t.src == "nldas_areal"]
        truth.append(t.assign(station_id=sid))
truth = pd.concat(truth)
print("truth through", truth.date.max().date(), "basins", truth.station_id.nunique())
print("rows om", len(om), "gfs", len(gf), "| NaN share om", om[G.COLS].isna().mean().round(3).to_dict())
print("NaN share gfs", gf[G.COLS].isna().mean().round(3).to_dict())
for lead in (1, 2, 3, 7):
    print(f"\n--- lead {lead}")
    for name, a in (("om", om), ("gfs", gf)):
        a1 = a[(a.lead == lead) & a.station_id.isin(coh)]
        for sh in (-1, 0, 1):
            t = truth.assign(date=truth.date + pd.Timedelta(days=sh))
            m = a1.merge(t, on=["station_id", "date"], suffixes=("", "_t")).dropna()
            out = []
            for c in ("precipitation_sum", "temperature_2m_mean", "shortwave_radiation_sum", "vapor_pressure"):
                r = np.corrcoef(m[c], m[c + "_t"])[0, 1]
                out.append(f"{c.split('_')[0][:6]} r={r:.3f} bias={np.mean(m[c]-m[c+'_t']):+.2f} mae={np.mean(abs(m[c]-m[c+'_t'])):.2f}")
            print(f"{name:3} shift{sh:+d} n={len(m):6d} inits={m.init.nunique():2d} | " + " | ".join(out))
