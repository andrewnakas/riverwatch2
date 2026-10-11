"""Index-only probe: can the 14-lead areal GFS schema be derived at all?

Zero GRIB bytes, zero writes. Reads only the .idx text sidecars and parses the
accumulation/averaging WINDOW of every record we would want, for every forecast
step out to f336.

This is the cheapest possible test of the hardest part of the f336 extension.
build_gfs_areal_lead1.py's own comment records how it goes wrong: the
"0-1 day acc" 24-hour APCP record exists ONLY at f024, and GFS ships BOTH a
6-hour bucket and a run-total at each step whose index strings are the SAME
LENGTH -- so the old "shortest window" tiebreak was arbitrary and could
double-count. It produced a 300 mm/day basin maximum.

So: parse the window explicitly as (a, b) forecast hours and require, for each
lead day L, exactly four DISJOINT 6-hour buckets tiling 24(L-1)..24L. A
run-total has a == 0 and b > 6, which that test rejects by construction.
"""
import re, sys
import s3fs

B = "noaa-gfs-bdp-pds"
DAY = sys.argv[1] if len(sys.argv) > 1 else "20261001"
WANT = {("APCP", "surface"), ("DSWRF", "surface"), ("PRES", "surface"),
        ("TMP", "2 m above ground"), ("TMAX", "2 m above ground"),
        ("TMIN", "2 m above ground"), ("SPFH", "2 m above ground")}

fs = s3fs.S3FileSystem(anon=True)

def idx_lines(h):
    p = f"{B}/gfs.{DAY}/00/atmos/gfs.t00z.pgrb2.0p25.f{h:03d}.idx"
    try:
        return fs.cat_file(p).decode().splitlines()
    except Exception:
        return None

WIN = re.compile(r"^(\d+)-(\d+) (hour|day) (acc|ave|max|min)")
def window(field):
    m = WIN.match(field)
    if not m:
        return None
    a, b, unit = int(m.group(1)), int(m.group(2)), m.group(3)
    if unit == "day":
        a, b = a * 24, b * 24
    return (a, b, m.group(4))

# 1. which steps exist?
steps = []
for h in list(range(6, 121, 6)) + list(range(123, 337, 3)):
    if h > 336: break
    if idx_lines(h) is not None:
        steps.append(h)
print(f"init gfs.{DAY}/00z -- steps present from the 6h/3h ladder: {len(steps)}")
print(f"  first {steps[:4]} ... last {steps[-4:]}")
six = [h for h in steps if h % 6 == 0]
print(f"  multiples of 6 available: {len(six)} (need 56 for f006..f336)")

# 2. for each lead day, can we tile it with 4 disjoint 6-hour buckets?
by_step = {}
for h in six:
    ls = idx_lines(h)
    recs = []
    for ln in ls:
        parts = ln.split(":")
        if len(parts) < 6: continue
        var, lev, fld = parts[3], parts[4], parts[5]
        if (var, lev) in WANT:
            recs.append((var, lev, fld, window(fld)))
    by_step[h] = recs

print()
print("lead | APCP 6h buckets found (a-b)                        | tiles 24h?")
ok_all = True
for L in range(1, 15):
    lo = 24 * (L - 1)
    need = [(lo, lo+6), (lo+6, lo+12), (lo+12, lo+18), (lo+18, lo+24)]
    found = {}
    for h in six:
        if not (lo < h <= lo + 24): continue
        for var, lev, fld, w in by_step.get(h, []):
            if var != "APCP" or w is None: continue
            a, b, kind = w
            if kind == "acc" and b - a == 6:
                found[(a, b)] = h
    tiles = sorted(found) == need
    ok_all &= tiles
    shown = ",".join(f"{a}-{b}" for a, b in sorted(found)) or "(none)"
    print(f"{L:4d} | {shown:50s} | {'YES' if tiles else 'NO'}")

print()
print("per-lead availability of the other channels (count of 4 expected):")
for var, lev in sorted(WANT - {("APCP", "surface")}):
    counts = []
    for L in range(1, 15):
        lo = 24 * (L - 1)
        n = sum(1 for h in six if lo < h <= lo + 24
                for v, lv, f, w in by_step.get(h, []) if (v, lv) == (var, lev))
        counts.append(n)
    print(f"  {var:5s} {lev:18s} {counts}")
print()
print("VERDICT:", "f336 schema is derivable as designed" if ok_all
      else "the 4x6h tiling FAILS at some lead -- schema undecided, do not build")
