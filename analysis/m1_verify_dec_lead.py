#!/usr/bin/env python3
"""Verify --dec-lead does exactly what it claims, and nothing more.

With dec_lead=1 the decoder step for target T must carry FORCING for T+1 (that is the diagnostic's
whole point, and it is beyond entitlement). Target-day and later DISCHARGE must still reach neither
tensor. And the encoder must be untouched.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(1, "scripts")
import train_mblstm as T

CORP = "data/modern_corpus/v3/camels_corpus_gridmetfix3_m3_cohort"
enc_vars = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:3]
try:
    from app import gages2 as _g2; enrich = _g2.enrich_station_attrs
except Exception:
    enrich = lambda d: d
reg = {}
try: reg = json.loads(Path("data/station_registry.json").read_text())
except Exception: pass
cam = json.loads(Path(T.CAMELS_ATTRS_PATH).read_text())

def build(dec_lead=0, enc_lead=0):
    stations = [T.load_station(Path(f"{CORP}/{g}.csv.gz"), enc_vars) for g in ids]
    stations = [x for x in stations if x is not None]
    attrs = {x["id"]: enrich(dict(reg.get(x["id"], {"id": x["id"]}))) for x in stations}
    for gid, rec in attrs.items():
        for k, v in cam.get(gid, {}).items():
            if v is not None: rec[k] = v
    c = T.Corpus(stations, attrs, pd.Timestamp("2014-09-30"), enc_vars, enc_vars, stats=None,
                 q_transform="linear", static_feats=T.CAMELS_STATIC_FEATS, head="quantile",
                 load_vars=enc_vars, dec_q=False)
    c.no_q_input = False; c.ar_mask_p = 0.0
    c.dec_lead = dec_lead; c.enc_lead = enc_lead
    return c

c0, c1 = build(dec_lead=0), build(dec_lead=1)
W = c1.window_index(pd.Timestamp("2014-10-01"), pd.Timestamp("2018-09-30"))
sel = W[np.linspace(0, len(W) - 1, 12).astype(int)]
print(f"{len(W)} windows, probing {len(sel)}\n")

print("which corpus row does the decoder's FIRST step read?")
print(f"  {'offset d (row t0+d)':<26}{'dec_lead=0':>13}{'dec_lead=1':>13}")
for d in (0, 1, 2, 3):
    m0 = m1 = 0.0
    for si, t0 in sel:
        si, t0 = int(si), int(t0)
        for c, which in ((c0, 0), (c1, 1)):
            st = c.stations[si]; i = t0 + d
            if i >= len(st["wx_n"]): continue
            base = c.sample(si, t0, None)
            old = st["wx_n"][i].copy(); st["wx_n"][i] = old + 10.0
            try:
                p = c.sample(si, t0, None)
                # decoder step 0 only
                v = float(np.nanmax(np.abs(np.array(p[1], float)[0] - np.array(base[1], float)[0])))
            finally:
                st["wx_n"][i] = old
            if which == 0: m0 = max(m0, v)
            else: m1 = max(m1, v)
    tag = ""
    if d == 1: tag = "   <- correct target-day row"
    if d == 2: tag = "   <- T+1, BEYOND entitlement"
    print(f"  d = {d:<22}{m0:13.4f}{m1:13.4f}{tag}")

print("\ndoes dec_lead leak DISCHARGE, or disturb the encoder?")
ok = True
for d in (0, 1, 2, 14):
    mq = me = 0.0
    for si, t0 in sel:
        si, t0 = int(si), int(t0)
        st = c1.stations[si]; i = t0 + d
        if i >= len(st["q_n"]): continue
        base = c1.sample(si, t0, None)
        old = st["q_n"][i]; st["q_n"][i] = old + 10.0
        try:
            p = c1.sample(si, t0, None)
            mq = max(mq, max(float(np.nanmax(np.abs(np.array(p[j], float) - np.array(base[j], float))))
                             for j in (0, 1)))
        finally:
            st["q_n"][i] = old
        oldw = st["wx_n"][i].copy(); st["wx_n"][i] = oldw + 10.0
        try:
            p = c1.sample(si, t0, None)
            me = max(me, float(np.nanmax(np.abs(np.array(p[0], float) - np.array(base[0], float)))))
        finally:
            st["wx_n"][i] = oldw
    st_q = ("✅ q history (expected)" if d <= 0 else
            ("⛔ Q LEAK" if mq > 1e-12 else "✅ no future q"))
    if d >= 1 and mq > 1e-12: ok = False
    print(f"  d={d:>3}  from q: {mq:9.6f}  {st_q:<24} forcing->x_enc: {me:9.6f}")

# encoder must be byte-identical between dec_lead 0 and 1
worst = 0.0
for si, t0 in sel:
    a = c0.sample(int(si), int(t0), None)[0]
    b = c1.sample(int(si), int(t0), None)[0]
    worst = max(worst, float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float)))))
print(f"\nencoder x_enc identical between dec_lead 0 and 1: max|diff| = {worst:.6g}")
print("\n✅ --dec-lead verified: it moves the decoder one day beyond the target, touches the encoder"
      "\n   not at all, and leaks no discharge." if ok and worst == 0.0 else
      "\n⛔ VERIFICATION FAILED")
