#!/usr/bin/env python3
"""VECTOR I -- are the normalization statistics computed over TEST-period data?

This is the most common leak in the streamflow literature: z-scoring forcings and discharge with
mu/sd taken over the whole record leaks the test period's mean and variance into every training
sample. It was previously marked clean in this audit by READING Corpus.__init__ (it takes a
train_end and is supposed to mask on it). That is the same kind of evidence that has been wrong 22
times in this campaign, so this measures it.

Method, as in vector G: perturb raw forcing and discharge values far inside the VAL and TEST windows
and check whether any normalization constant moves. mu/sd must be invariant to anything after
train_end and must move for data before it -- the second half matters, because a statistic that
never moves is not being computed from the data at all and the test would pass vacuously.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import train_mblstm as T

CORP = "data/modern_corpus/v3/camels_corpus_gridmet_m3_cohort"
enc_vars = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
TRAIN_END = pd.Timestamp("2014-09-30")
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
cam = json.loads(Path(T.CAMELS_ATTRS_PATH).read_text())

def build(raw_edit=None):
    """Load stations, optionally mutate the RAW csv values at given dates, build a Corpus,
    and return every normalization constant it derived."""
    stations = []
    for g in ids:
        st = T.load_station(Path(f"{CORP}/{g}.csv.gz"), enc_vars)
        if st is None: continue
        if raw_edit is not None:
            # load_station may hand back read-only views; make them writable first
            for k in ("wx", "q", "q_raw"):
                if k in st and isinstance(st[k], np.ndarray):
                    st[k] = np.array(st[k], copy=True)
            raw_edit(st)
        stations.append(st)
    attrs = {s["id"]: enrich(dict(reg.get(s["id"], {"id": s["id"]}))) for s in stations}
    for gid, rec in attrs.items():
        for k, v in cam.get(gid, {}).items():
            if v is not None: rec[k] = v
    c = T.Corpus(stations, attrs, TRAIN_END, enc_vars, enc_vars, stats=None,
                 q_transform="linear", static_feats=T.CAMELS_STATIC_FEATS,
                 head="quantile", load_vars=enc_vars, dec_q=False)
    out = {}
    # the real attribute names, read off Corpus.__init__: forcing z-stats, static z-stats,
    # and the per-station discharge z-stats
    for k in ("wx_mean", "wx_std", "static_median", "static_mean", "static_std"):
        v = getattr(c, k, None)
        if v is not None:
            out[k] = np.asarray(v, dtype=float).copy()
    for i, s in enumerate(c.stations):
        for k in ("mu_q", "sd_q"):
            if k in s: out[f"st{i}.{k}"] = float(s[k])
    return out

def edit_at(lo, hi, cols, delta):
    def f(st):
        d = st["dates"]
        m = (d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))
        if not m.any(): return
        for j, cn in enumerate(enc_vars):
            if cn in cols and j < st["wx"].shape[1]:
                st["wx"][m, j] = st["wx"][m, j] + delta
        if "q" in st and "q" in cols:
            st["q"][m] = st["q"][m] + delta
        if "q_raw" in st and "q" in cols:
            st["q_raw"][m] = st["q_raw"][m] + delta
    return f

base = build()
print(f"derived {len(base)} normalization constants from {len(ids)} stations "
      f"(train_end {TRAIN_END.date()})\n")

CASES = [
    ("TRAIN  2010-01-01..2012-12-31", "2010-01-01", "2012-12-31", False),
    ("VAL    2015-01-01..2017-12-31", "2015-01-01", "2017-12-31", True),
    ("TEST   2019-01-01..2024-12-31", "2019-01-01", "2024-12-31", True),
]
print(f"{'perturbed window':<34}{'constants moved':>17}{'max |delta|':>14}   status")
fired = []
for label, lo, hi, must_be_zero in CASES:
    pert = build(edit_at(lo, hi, {"precipitation_sum", "temperature_2m_mean", "q"}, 25.0))
    moved, mx = 0, 0.0
    for k, v in base.items():
        w = pert.get(k)
        if w is None: continue
        dv = float(np.nanmax(np.abs(np.asarray(w) - np.asarray(v))))
        if dv > 1e-9: moved += 1
        mx = max(mx, dv)
    if must_be_zero:
        st = "⛔ LEAK" if moved else "✅ invariant (correct)"
        if moved: fired.append(label)
    else:
        st = "✅ responds (stats are real)" if moved else "⚠️ NO response -- vacuous test"
        if not moved: fired.append(label + " [vacuous]")
    print(f"  {label:<32}{moved:>17}{mx:>14.6f}   {st}")

print("\n" + "=" * 78)
if fired:
    print(f"⛔ VECTOR I: {fired}")
else:
    print("✅ VECTOR I CLEAN: every normalization constant is invariant to val- and test-period")
    print("   values, and does respond to train-period values, so the statistics are computed on")
    print("   the training window only and the test is not passing vacuously.")
json.dump({"n_constants": len(base), "fired": fired}, 
          open("benchmarks/m1_leakaudit_vectorI_normalization.json", "w"), indent=2)
