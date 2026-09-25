#!/usr/bin/env python
"""MODERN-1: is the Daymet DATA route actually usable right now?

Exercises the real path -- the builder's own `fetch()`, with a real basin bbox
and a real DAP4 constraint -- rather than a metadata endpoint.

⚠️ Why this exists. The first OPeNDAP recovery watcher probed the collection's
`.dmr` metadata. That came back (200) while actual data requests were still
failing with curl rc=35 (SSL connect error), so the watcher declared victory,
relaunched the build, and the next basin failed immediately. Metadata recovering
is not the same as data recovering. Probe the thing you actually need.

exit 0 = route usable, exit 1 = not.
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def main() -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dm", "scripts_modern/build_daymet_areal_modern.py")
    dm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dm)
    import geopandas as gpd

    gid = sys.argv[1] if len(sys.argv) > 1 else "02055100"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 1982
    g = gpd.read_file(dm.SHP)
    g["gid"] = g["GAGE_ID"].astype(str).str.zfill(8)
    row = g[g.gid == gid]
    if row.empty:
        print(f"PROBE FAIL: basin {gid} not in the shapefile")
        return 1
    i0, i1, j0, j1 = dm.basin_idx(row.to_crs(dm.LCC).geometry.iloc[0])
    wd = Path(tempfile.mkdtemp(prefix="dmprobe_", dir="/tmp"))
    out = wd / "probe.nc4"
    try:
        dm.fetch("prcp", year, i0, i1, j0, j1, out, str(wd / ".ck"))
        sz = out.stat().st_size if out.exists() else 0
        with open(out, "rb") as fh:
            magic = fh.read(4)
        if magic != b"\x89HDF":
            print(f"PROBE FAIL: {sz} bytes but not HDF5 (magic={magic!r})")
            return 1
        print(f"PROBE OK: {sz} bytes of real HDF5 from the data route")
        return 0
    except Exception as e:
        print(f"PROBE FAIL: {type(e).__name__}: {str(e)[:160]}")
        return 1
    finally:
        out.unlink(missing_ok=True)
        for p in wd.glob("*"):
            p.unlink(missing_ok=True)
        wd.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
