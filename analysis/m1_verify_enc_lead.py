#!/usr/bin/env python3
"""Two checks that must both pass before --enc-lead is used for anything.

1. DEFAULT OFF IS BYTE-IDENTICAL. Loads the pre-patch snapshot of train_mblstm.py and the patched
   file as two separate modules, builds the same Corpus from each, and compares x_enc/x_dec/y/m
   byte-for-byte over many windows. Verified against the actual previous code, not by inspection --
   a default-off claim is exactly the kind of thing that passes a reading and fails in fact.

2. WITH THE FLAG ON, TARGET-DAY DISCHARGE STILL CANNOT REACH THE INPUTS. Re-runs vector G's
   perturbation probe at enc_lead=1: perturbing observed q at t0+1.. must move NOTHING, while
   target-day FORCING must now move x_enc (that is the point of the flag).
"""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

# The snapshot lives outside the repo, so it does not get the repo root on sys.path the way
# scripts/train_mblstm.py does when run directly. Both modules import `app`, so put the repo
# root first explicitly before either is loaded.
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.getcwd(), "scripts"))

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

OLD = load("scripts/.train_mblstm_prepatch.py", "tm_old")
NEW = load("scripts/train_mblstm.py", "tm_new")

CORP = "data/modern_corpus/v3/camels_corpus_gridmet_m3_cohort"
enc_vars = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:4]
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

def build(M, dec_q=False, enc_lead=0):
    stations = [M.load_station(Path(f"{CORP}/{g}.csv.gz"), enc_vars) for g in ids]
    stations = [s for s in stations if s is not None]
    attrs = {s["id"]: enrich(dict(reg.get(s["id"], {"id": s["id"]}))) for s in stations}
    cam = json.loads(Path(M.CAMELS_ATTRS_PATH).read_text())
    for gid, rec in attrs.items():
        for k, v in cam.get(gid, {}).items():
            if v is not None: rec[k] = v
    c = M.Corpus(stations, attrs, pd.Timestamp("2014-09-30"), enc_vars, enc_vars, stats=None,
                 q_transform="linear", static_feats=M.CAMELS_STATIC_FEATS, head="quantile",
                 load_vars=enc_vars, dec_q=dec_q)
    c.no_q_input = False
    c.ar_mask_p = 0.0
    if enc_lead:
        c.enc_lead = enc_lead
    return c

print("=== CHECK 1: default off must be byte-identical to the pre-patch code ===")
worst = 0.0
nchk = 0
for dq in (False, True):
    A, B = build(OLD, dec_q=dq), build(NEW, dec_q=dq)
    W = B.window_index(pd.Timestamp("2014-10-01"), pd.Timestamp("2018-09-30"))
    sel = W[np.linspace(0, len(W) - 1, 40).astype(int)]
    for si, t0 in sel:
        sa, sb = A.sample(int(si), int(t0), None), B.sample(int(si), int(t0), None)
        assert len(sa) == len(sb), (len(sa), len(sb))
        for k, (x, y) in enumerate(zip(sa, sb)):
            if isinstance(x, np.ndarray):
                assert x.shape == y.shape, (dq, k, x.shape, y.shape)
                d = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                worst = max(worst, d); nchk += 1
    print(f"  dec_q={dq}: {len(sel)} windows compared")
print(f"  max |difference| over {nchk} arrays = {worst:.6g}")
assert worst == 0.0, f"NOT byte-identical: {worst}"
print("  ✅ byte-identical with --enc-lead absent\n")

print("=== CHECK 2: with --enc-lead 1, target-day DISCHARGE must still not reach the inputs ===")
ok = True
for dq in (False, True):
    c = build(NEW, dec_q=dq, enc_lead=1)
    assert c.enc_lead == 1
    W = c.window_index(pd.Timestamp("2014-10-01"), pd.Timestamp("2018-09-30"))
    sel = W[np.linspace(0, len(W) - 1, 10).astype(int)]
    print(f"  dec_q={dq}")
    for d in (-1, 0, 1, 2, 14):
        mq = mf = 0.0
        for si, t0 in sel:
            si, t0 = int(si), int(t0)
            st = c.stations[si]
            i = t0 + d
            if i < 0 or i >= len(st["q_n"]): continue
            base = c.sample(si, t0, None)
            # perturb DISCHARGE at t0+d
            old = st["q_n"][i]; st["q_n"][i] = old + 10.0
            try:
                p = c.sample(si, t0, None)
                mq = max(mq, max(float(np.nanmax(np.abs(np.array(p[j], float) - np.array(base[j], float))))
                                 for j in (0, 1)))
            finally:
                st["q_n"][i] = old
            # perturb FORCING at t0+d
            oldw = st["wx_n"][i].copy(); st["wx_n"][i] = oldw + 10.0
            try:
                p = c.sample(si, t0, None)
                mf = max(mf, float(np.nanmax(np.abs(np.array(p[0], float) - np.array(base[0], float)))))
            finally:
                st["wx_n"][i] = oldw
        if d >= 1:
            st_q = "⛔ Q LEAK" if mq > 1e-12 else "✅ no future q"
            if mq > 1e-12: ok = False
        else:
            st_q = "✅ q history (expected)"
        print(f"    d={d:>3}  max|dx| from q = {mq:9.6f}  {st_q:<28} "
              f"from forcing into x_enc = {mf:9.6f}")
print()
# the flag's PURPOSE: target-day forcing must now reach x_enc, where at lead 0 it did not
c0, c1 = build(NEW, enc_lead=0), build(NEW, enc_lead=1)
W = c1.window_index(pd.Timestamp("2014-10-01"), pd.Timestamp("2018-09-30"))
sel = W[np.linspace(0, len(W) - 1, 10).astype(int)]
m0 = m1 = 0.0
for si, t0 in sel:
    si, t0 = int(si), int(t0)
    for c, acc in ((c0, "0"), (c1, "1")):
        st = c.stations[si]; i = t0 + 1
        base = c.sample(si, t0, None)
        oldw = st["wx_n"][i].copy(); st["wx_n"][i] = oldw + 10.0
        try:
            p = c.sample(si, t0, None)
            v = float(np.nanmax(np.abs(np.array(p[0], float) - np.array(base[0], float))))
        finally:
            st["wx_n"][i] = oldw
        if acc == "0": m0 = max(m0, v)
        else: m1 = max(m1, v)
print(f"target-day FORCING -> x_enc:  enc_lead=0 gives {m0:.6f}   enc_lead=1 gives {m1:.6f}")
if ok and m1 > 1e-9 and m0 <= 1e-12:
    print("\n✅ BOTH CHECKS PASS: off is byte-identical; on, the target day's FORCING reaches the")
    print("   encoder (which at enc_lead=0 it did not) while target-day DISCHARGE still cannot.")
else:
    print("\n⛔ CHECK FAILED -- do not use --enc-lead")
