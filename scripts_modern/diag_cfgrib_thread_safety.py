"""Is cfgrib safe to decode from multiple threads?

The builder serialises decode behind a lock because libhdf5 needed exactly that
in the sibling NLDAS builder (measured: segfault). But the lock makes one init
date ~11 min, so a 115-date build is ~21 h. If eccodes is in fact thread-safe
here, dropping the lock is a large, free speedup -- and if it is not, finding
out by measurement beats finding out by a corrupted corpus.

Measures BOTH: does it crash, and does it return identical arrays.
"""
import sys, importlib.util, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path("/home/nakas/riverwatch2")
sp = importlib.util.spec_from_file_location("_g1", ROOT / "scripts_modern" / "build_gfs_areal_lead1.py")
g1 = importlib.util.module_from_spec(sp); sys.modules["_g1"] = g1; sp.loader.exec_module(g1)

import s3fs
fs = s3fs.S3FileSystem(anon=True)
grb, rngs = g1.idx_ranges(fs, "20261001", 24)
rngs = rngs[:4]
print(f"{len(rngs)} records from f024")

# reference, strictly serial
t0 = time.time()
ref = g1.decode(fs, grb, rngs)
t_serial = time.time() - t0
ref_arrays = {v: [a for _, a in lst] for v, lst in ref.items()}
print(f"serial decode of {len(rngs)} records: {t_serial:.2f}s")

def one(_):
    return g1.decode(fs, grb, rngs)

N = 12
for label, workers in (("threads=6", 6),):
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(one, range(N)))
    except Exception as e:
        print(f"{label}: RAISED {type(e).__name__}: {e}")
        sys.exit(0)
    el = time.time() - t0
    mism = 0
    for o in outs:
        for v, lst in ref_arrays.items():
            got = [a for _, a in o.get(v, [])]
            if len(got) != len(lst) or any(not np.array_equal(x, y) for x, y in zip(got, lst)):
                mism += 1
    print(f"{label}: {N} concurrent decodes in {el:.2f}s "
          f"({N*len(rngs)/el:.1f} rec/s vs serial {len(rngs)/t_serial:.1f} rec/s)")
    print(f"   output mismatches: {mism} (0 => byte-identical to the serial reference)")
    print("   VERDICT:", "thread-safe here -- the lock can go" if mism == 0
          else "*** NOT safe -- keep the lock")
