#!/usr/bin/env python3
"""Feasibility probe for a FRESH, AREAL GFS lead-1 forcing sample.

Why rebuild rather than wait for the unmounted drive:
  - the archived data/mblstm/gfs_fcst is POINT-SAMPLED at gauge coordinates, which runs ~53% dry
    versus the areal mean in mountain terrain, while the whole modern corpus is areal. A fresh
    build can be areal and therefore removes a stated caveat instead of inheriting it.
  - it needs no user action.

The cost question is the only open one: a GFS f024 pgrb2 0p25 file is ~400 MB, and a few hundred
dates of those is ~100 GB. GRIB2 files ship a .idx sidecar giving each record's byte offset, so the
right move is a RANGE request for the one precipitation record. This measures whether that works and
what it costs per date.
"""
import s3fs

B = "noaa-gfs-bdp-pds"
fs = s3fs.S3FileSystem(anon=True)
SIZE = "Size"

print("=== earliest coverage ===")
for day in ("20201001", "20210201", "20210301", "20210501"):
    p = f"{B}/gfs.{day}/00/atmos"
    print(f"  gfs.{day}: {'present' if fs.exists(p) else 'ABSENT'}")

day = "20230615"
p = f"{B}/gfs.{day}/00/atmos"
# ⚠️ Select by EXACT basename. A substring match on "pgrb2.0p25.f024" also catches
# gfs.t00z.goessimpgrb2.0p25.f024 -- a 4 MB simulated-satellite product whose .idx has 4 records
# and none of the surface fields, which made the first run of this probe report 0.00 MB per date.
BASE = "gfs.t00z.pgrb2.0p25.f024"
fl = fs.ls(p)
grb = [x for x in fl if x.split("/")[-1] == BASE]
idx = [x for x in fl if x.split("/")[-1] == BASE + ".idx"]
assert grb, f"no {BASE} under {p}"
print(f"\n=== gfs.{day}/00/atmos f024 ===")
for x in grb:
    print(f"  grib: {x.split('/')[-1]}  {fs.info(x)[SIZE]/1e6:.0f} MB")
for x in idx:
    print(f"  idx : {x.split('/')[-1]}  {fs.info(x)[SIZE]/1e3:.0f} KB")

if not idx:
    print("no .idx sidecar -> range subsetting not available this way"); raise SystemExit(0)

raw = fs.cat(idx[0]).decode()
lines = raw.strip().split("\n")
print(f"\nindex has {len(lines)} records; looking for surface precip / 2m temp")
want = []
for i, ln in enumerate(lines):
    f = ln.split(":")
    if len(f) < 5: continue
    var, lev = f[3], f[4]
    if (var == "APCP" and "surface" in lev) or (var in ("TMP", "TMAX", "TMIN") and "2 m above" in lev) \
       or (var == "DSWRF" and "surface" in lev) or (var == "SPFH" and "2 m above" in lev):
        start = int(f[1])
        nxt = lines[i + 1].split(":")[1] if i + 1 < len(lines) else None
        end = int(nxt) - 1 if nxt else None
        want.append((var, lev, start, end, (end - start + 1) if end else None))
for v, lev, s, e, n in want:
    print(f"  {v:<6} {lev:<28} bytes {s}-{e}  ({n/1e6:.2f} MB)" if n else f"  {v} {lev} bytes {s}-EOF")

tot = sum(n for *_, n in want if n)
print(f"\nrange-request total per date: {tot/1e6:.2f} MB  (vs {fs.info(grb[0])[SIZE]/1e6:.0f} MB whole file)")
print(f"for 300 sampled dates: {tot*300/1e9:.2f} GB  -- vs {fs.info(grb[0])[SIZE]*300/1e9:.0f} GB unsubsetted")

# prove a range read actually returns a decodable GRIB message
v, lev, s, e, n = want[0]
blob = fs.read_block(grb[0], s, (e - s + 1))
print(f"\nrange read of the {v} record: {len(blob)} bytes, "
      f"magic={blob[:4]!r} (expect b'GRIB')")
print("✅ byte-range subsetting works" if blob[:4] == b"GRIB" else "⚠️ not a GRIB message")
