#!/usr/bin/env python3
"""Convert the raw areal GFS daily files into the 14-lead decoder-forcing schema.

_parse_forcing_file() requires every station to carry all 14 leads or it DROPS that station, so
each file must have 14 rows per basin. Lead 1 is the real GFS forecast for day D; leads 2-14 are
the OBSERVED corpus values for D+1..D+13.

That is sound because the decoder is MEASURABLY CAUSAL (analysis/m1_decoder_causality.py: a decoder
input at lead k moves only leads >= k, exact zeros below), so the observed fill cannot affect the
lead-1 prediction this measurement is about. Anything read off leads 2-14 from these files would be
perfect-forcing and must NOT be reported as forecast skill -- hence the explicit note written into
each file's companion marker.

Output goes to data/mblstm/gfs_fcst/ because load_gfs_windows() reads GFS_DIR, and the file name's
first dot-separated field is the INIT date (a 00z run on D has lead 1 = calendar day D, matching
t0 = D-1).
"""
import glob, json, os
from pathlib import Path
import numpy as np, pandas as pd

RAW = Path("data/mblstm/gfs_fcst_areal")
OUT = Path("data/mblstm/gfs_fcst")
REF = Path("data/modern_corpus/v3/camels_corpus_gridmetfix3_m3")
VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
HORIZON = 14

OUT.mkdir(parents=True, exist_ok=True)
cohort = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]]
cset = set(cohort)

# load each cohort basin's observed series once
obs = {}
for sid in cohort:
    p = REF / f"{sid}.csv.gz"
    if p.exists():
        c = pd.read_csv(p, dtype={"date": str}).set_index("date")
        obs[sid] = c[VARS]
print(f"observed series for {len(obs)} basins")

files = sorted(glob.glob(str(RAW / "*.csv.gz")))
print(f"{len(files)} raw GFS dates -> {OUT}")
written = skipped = 0
for f in files:
    init = os.path.basename(f).split(".")[0]
    g = pd.read_csv(f, dtype={"station_id": str})
    g["station_id"] = g.station_id.str.zfill(8)
    g = g[g.station_id.isin(cset)]
    d0 = pd.Timestamp(init)
    out_rows = []
    for sid, grp in g.groupby("station_id"):
        c = obs.get(sid)
        if c is None:
            continue
        # lead 1 = the GFS forecast for day D itself
        rec = {"station_id": sid, "lead_day": 1}
        row = grp.iloc[0]
        for v in VARS:
            rec[v] = float(row[v])
        block = [rec]
        # leads 2..14 = OBSERVED for D+1..D+13
        okall = True
        for L in range(2, HORIZON + 1):
            key = (d0 + pd.Timedelta(days=L - 1)).strftime("%Y-%m-%d")
            if key not in c.index:
                okall = False; break
            o = c.loc[key]
            r2 = {"station_id": sid, "lead_day": L}
            for v in VARS:
                r2[v] = float(o[v])
            block.append(r2)
        if okall and all(np.isfinite(list(b[v] for v in VARS)).all() for b in block):
            out_rows.extend(block)
    if not out_rows:
        skipped += 1; continue
    df = pd.DataFrame(out_rows)
    n_st = df.station_id.nunique()
    assert len(df) == n_st * HORIZON, (len(df), n_st)
    df.to_csv(OUT / f"{init}.csv.gz", index=False, compression="gzip")
    written += 1
print(f"\nwritten={written} skipped={skipped}")
# the marker that says what these files are, so no one reads leads 2-14 as forecast skill
(OUT / "README_FORCING.txt").write_text(
    "Areal GFS lead-1 decoder forcing for the MODERN-1 day-1 decay measurement.\n"
    "lead_day 1  = real GFS 00z forecast for the init date itself (1-day-ahead)\n"
    "lead_day 2-14 = OBSERVED corpus values (camels_corpus_gridmetfix3_m3)\n"
    "The decoder is measurably causal, so the observed fill cannot affect the lead-1 number.\n"
    "⛔ Nothing read off leads 2-14 of these files is forecast skill -- it is perfect forcing.\n")
ex = sorted(glob.glob(str(OUT / "*.csv.gz")))[:1]
if ex:
    d = pd.read_csv(ex[0], dtype={"station_id": str})
    print(f"example {os.path.basename(ex[0])}: {len(d)} rows, "
          f"{d.station_id.nunique()} basins x {d.lead_day.nunique()} leads, cols {list(d.columns)}")
