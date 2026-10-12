"""Are the differences multiples of the GRIB packing increment?

If so they are quantisation, not selection error: summing four independently
packed buckets accumulates up to 4x the packing step, while the single 24-hour
record carries one. The areal reducer is a weighted mean, so a per-cell step
appears attenuated and smeared in basin space -- what should still hold is that
the GRID-level differences are near-exact multiples of the step.
"""
import sys, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/home/nakas/riverwatch2")
def load(n, f):
    sp = importlib.util.spec_from_file_location(n, ROOT / "scripts_modern" / f)
    m = importlib.util.module_from_spec(sp); sys.modules[n] = m; sp.loader.exec_module(m); return m
g1 = load("_g1", "build_gfs_areal_lead1.py")
g14 = load("_g14", "build_gfs_areal_14lead.py")

import s3fs
fs = s3fs.S3FileSystem(anon=True)
day = "20261001"
recs = {}
for h in (6, 12, 18, 24):
    grb, rngs = g1.idx_ranges(fs, day, h)
    recs[h] = g1.decode(fs, grb, rngs)

buckets = g14.pick_bucket(recs, 1, "APCP", "acc")
cands = recs[24].get("APCP", [])
single = [a for ln, a in cands if "0-1 day acc" in ln or "0-24 hour acc" in ln][0]

# GRID level, before any areal averaging.
four_g = np.sum(buckets, axis=0)
d = (four_g - single).ravel()
ad = np.abs(d)
nz = ad[ad > 0]
print("GRID cells:", d.size)
print(f"  cells differing      {len(nz)} ({len(nz)/d.size*100:.2f}%)")
print(f"  mean  diff           {d.mean():+.3e} mm  <- bias")
print(f"  max  |diff|          {ad.max():.6f} mm")
if len(nz):
    # The packing step shows up as the smallest non-zero magnitude.
    step = np.min(nz)
    print(f"  smallest non-zero    {step:.8f} mm   <- candidate packing step")
    ratios = nz / step
    near_int = np.abs(ratios - np.round(ratios)) < 0.02
    print(f"  of differing cells, fraction at an INTEGER multiple of that step: "
          f"{near_int.mean()*100:.2f}%")
    print(f"  distinct |diff| values: {len(np.unique(np.round(nz, 8)))}")
    print(f"  max multiple observed: {ratios.max():.2f}  (4 buckets => <= ~4 expected)")
print()
ok = abs(d.mean()) < 1e-3 and (len(nz) == 0 or near_int.mean() > 0.9)
print("VERDICT:", "QUANTISATION -- differences are integer multiples of the packing step, "
      "no bias. The bucket selection is correct."
      if ok else "*** NOT quantisation -- the selection is suspect")
