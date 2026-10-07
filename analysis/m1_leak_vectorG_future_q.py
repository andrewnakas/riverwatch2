#!/usr/bin/env python3
"""VECTOR G -- can the TARGET-DAY (or later) observed discharge reach the model's inputs?

This is the one leak that would make a discharge-assimilating record worthless: if the encoder or
decoder input for target day T contains q(T) or later, the model is reading its own answer.

Reading the code says it cannot: Corpus.sample() takes the encoder over [t0-364 .. t0], the decoder
forcing over t0+1..t0+14, targets y over t0+1..t0+14, and the --dec-q channel is q_n[-1] = q(t0).
But I have asserted direction from code and been wrong 22 times in this campaign, so this measures
it instead, and measures it on the INPUTS rather than the outputs: leakage is a property of the
input tensor. If q(T) is absent from x_enc and x_dec, nothing downstream can use it, for any
architecture or weight set.

Method: perturb the station's normalized discharge at ONE absolute index t0+d, rebuild the sample,
and take max|delta| over x_enc and x_dec. Expected: NONZERO for d <= 0 (the encoder legitimately
assimilates history through t0) and EXACTLY ZERO for d >= +1 (the target region). A nonzero reading
at d >= +1 is a leak; a zero reading at d == 0 would mean assimilation is not reaching the model.
Run with dec_q on and off, since --dec-q is the channel that most plausibly carries it.
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import train_mblstm as T

CORP = "data/modern_corpus/v3/camels_corpus_gridmet_m3_cohort"
ENC = T.VAR_SETS["camels1f"] if hasattr(T, "VAR_SETS") else None
enc_vars = ENC if ENC else ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                            "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
from pathlib import Path
ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:3]
stations = []
for g in ids:
    st = T.load_station(Path(f"{CORP}/{g}.csv.gz"), enc_vars)
    if st is not None:
        stations.append(st)
assert stations, "no stations loaded"
print(f"loaded {len(stations)} stations; enc_vars={len(enc_vars)}")

import importlib
gages2 = importlib.import_module("app.gages2") if False else None
try:
    from app import gages2 as _g2
    enrich = _g2.enrich_station_attrs
except Exception:
    enrich = lambda d: d
reg = {}
try:
    reg = json.loads(Path("data/station_registry.json").read_text())
except Exception:
    pass
attrs_by_id = {s["id"]: enrich(dict(reg.get(s["id"], {"id": s["id"]}))) for s in stations}
cam = json.loads(Path(T.CAMELS_ATTRS_PATH).read_text())
for gid, rec in attrs_by_id.items():
    for k, v in cam.get(gid, {}).items():
        if v is not None:
            rec[k] = v

OFFS = [-3, -2, -1, 0, 1, 2, 3, 7, 14]
results = {}
for dec_q in (False, True):
    c = T.Corpus(stations, attrs_by_id, pd.Timestamp("2014-09-30"), enc_vars, enc_vars,
                 stats=None, q_transform="linear", static_feats=T.CAMELS_STATIC_FEATS,
                 head="quantile", load_vars=enc_vars, dec_q=dec_q)
    c.no_q_input = False
    c.ar_mask_p = 0.0
    W = c.window_index(pd.Timestamp("2014-10-01"), pd.Timestamp("2018-09-30"))
    assert len(W), "no windows"
    sel = W[np.linspace(0, len(W) - 1, 12).astype(int)]
    print(f"\n=== dec_q={dec_q} : {len(W)} windows, probing {len(sel)} ===")
    print(f"  {'offset d (index t0+d)':<26}{'max|d x_enc|':>14}{'max|d x_dec|':>14}   status")
    rows = {}
    for d in OFFS:
        me, md = 0.0, 0.0
        for si, t0 in sel:
            si, t0 = int(si), int(t0)
            st = c.stations[si]
            idx = t0 + d
            if idx < 0 or idx >= len(st["q_n"]):
                continue
            base = c.sample(si, t0, None)
            xe0, xd0 = np.array(base[0], float), np.array(base[1], float)
            old = st["q_n"][idx]
            st["q_n"][idx] = old + 10.0          # a perturbation no normalization can hide
            try:
                pert = c.sample(si, t0, None)
                xe1, xd1 = np.array(pert[0], float), np.array(pert[1], float)
            finally:
                st["q_n"][idx] = old              # restore before the next probe
            me = max(me, float(np.nanmax(np.abs(xe1 - xe0))))
            md = max(md, float(np.nanmax(np.abs(xd1 - xd0))))
        legit = d <= 0
        if legit:
            status = "✅ encoder history (expected)" if (me > 0 or md > 0) else "⚠️ NO assimilation!"
        else:
            status = "⛔ LEAK" if (me > 1e-12 or md > 1e-12) else "✅ no future q in inputs"
        print(f"  d = {d:<22}{me:14.6f}{md:14.6f}   {status}")
        rows[d] = {"max_abs_dx_enc": me, "max_abs_dx_dec": md}
    results[f"dec_q={dec_q}"] = rows

leaks = [(k, d) for k, r in results.items() for d, v in r.items()
         if d >= 1 and (v["max_abs_dx_enc"] > 1e-12 or v["max_abs_dx_dec"] > 1e-12)]
assim = [(k, d) for k, r in results.items() for d, v in r.items()
         if d == 0 and v["max_abs_dx_enc"] <= 1e-12 and v["max_abs_dx_dec"] <= 1e-12]
print("\n" + "=" * 78)
if leaks:
    print(f"⛔ VECTOR G FIRES: future discharge reaches the inputs at {leaks}")
elif assim:
    print(f"⚠️ q(t0) does NOT reach the inputs at {assim} -- assimilation is not working")
else:
    print("✅ VECTOR G CLEAN: perturbing observed discharge at t0+1 or later changes NEITHER")
    print("   x_enc nor x_dec by any amount, with --dec-q on and off, while q(t0) and earlier")
    print("   do move them. The model cannot see its own target day.")
json.dump(results, open("benchmarks/m1_leakaudit_vectorG_future_q.json", "w"), indent=2)
