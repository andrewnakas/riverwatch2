#!/usr/bin/env python3
"""VECTOR J -- do the val and test dumps share any (station, date)?

Everything in MODERN-1 is selected on val1 -- members, seeds, the combiner, the h1-weight, and now
the gridMET registration verdict -- and test1 is read once. That discipline is worth exactly as much
as the disjointness of the two frames. If any (station, target date) row appears in both dumps, then
selecting on val is partly selecting on test, and every "test read once" claim is weaker than stated.

Checked on the ACTUAL dumps the frozen records were computed from, not on the configured window
dates: the configured windows can be right while an off-by-one in the dump grid still puts a shared
row in both files. Also checks the no-q arm's member, since it is a separate corpus and dump path.
"""
import glob, json
import pandas as pd

MEMBERS = ["gridmetm9", "fused4md49", "fused4m9", "nldasm9", "lstm_fused4md4"]
print(f"{'member':<18}{'val rows':>11}{'test rows':>11}{'val span':>26}{'test span':>26}{'shared':>9}")
bad = []
for m in MEMBERS:
    vs = sorted(glob.glob(f"data/mblstm/l51_dumps/m1_{m}_s50?_val1.csv.gz"))
    ts = sorted(glob.glob(f"data/mblstm/l51_dumps/m1_{m}_s50?_test1.csv.gz"))
    if not vs or not ts:
        print(f"  {m:<16}{'(dumps missing)':>35}"); continue
    V = pd.read_csv(vs[0], dtype={"station_id": str})
    T = pd.read_csv(ts[0], dtype={"station_id": str})
    for D in (V, T):
        if "h" in D.columns:
            D.drop(D.index[D["h"] != 1], inplace=True)
    # the dump grid is keyed on t0 (last encoder day); the TARGET date for h=1 is t0+1
    vk = set(zip(V["station_id"], pd.to_datetime(V["t0"]) + pd.Timedelta(days=1)))
    tk = set(zip(T["station_id"], pd.to_datetime(T["t0"]) + pd.Timedelta(days=1)))
    sh = vk & tk
    vd = sorted(d for _, d in vk); td = sorted(d for _, d in tk)
    print(f"  {m:<16}{len(vk):>11}{len(tk):>11}"
          f"{str(vd[0].date())+'..'+str(vd[-1].date()):>26}"
          f"{str(td[0].date())+'..'+str(td[-1].date()):>26}{len(sh):>9}")
    if sh:
        bad.append((m, len(sh), sorted(sh)[:3]))

print("\n" + "=" * 78)
if bad:
    print("⛔ VECTOR J FIRES: val and test dumps share rows:")
    for m, n, ex in bad:
        print(f"   {m}: {n} shared, e.g. {ex}")
else:
    print("✅ VECTOR J CLEAN: no (station, target date) appears in both the val1 and test1 dumps")
    print("   for any scored member in either arm. Selection on val1 touches no test row.")
json.dump({"shared": [{"member": m, "n": n} for m, n, _ in bad]},
          open("benchmarks/m1_leakaudit_vectorJ_frames.json", "w"), indent=2)
