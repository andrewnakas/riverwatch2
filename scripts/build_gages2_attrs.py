#!/usr/bin/env python3
"""v14.5a: pull GAGES-II static basin attributes for our 1893 stations.

GAGES-II (Falcone 2011) has 9067 CONUS gauges with curated static basin
attributes — land cover %, hydrologic soil groups, basin slope/elev/area,
baseflow index, climate normals. ~68% of our active list joins by 8-digit
USGS site number. Misses (newer gauges, post-2014 actives, AK/HI/PR) get
no GAGES-II row and feature cells stay NaN — LightGBM handles natively.

Run once locally; commits the trimmed `data/gages2_attrs.json` so CI never
needs to fetch the 53 MB zip:

    python scripts/build_gages2_attrs.py

Output schema per station:
    {
      "FORESTNLCD06": float,        # % forest (NLCD 2006)
      "DEVNLCD06": float,           # % developed (impervious proxy)
      "WOODYWETNLCD06": float,      # % woody wetland
      "EMERGWETNLCD06": float,      # % emergent wetland
      "HGA_PCT": float,             # hydrologic soil group A (well-drained sand)
      "HGB_PCT": float,             # B (moderate)
      "HGC_PCT": float,             # C (slow infiltration)
      "HGD_PCT": float,             # D (clay/bedrock; high runoff)
      "AWCAVE": float,              # avg available water capacity
      "PERMAVE": float,             # avg permeability (in/hr)
      "ELEV_MEAN_M_BASIN": float,
      "SLOPE_PCT": float,           # mean basin slope %
      "BFI_AVE": float,             # baseflow index 0-100 (Wolock 2003)
      "TOPWET": float,              # topographic wetness index
      "PPTAVG_BASIN": float,        # 30yr mean annual precip (cm/yr)
      "T_AVG_BASIN": float,         # 30yr mean annual temp (C)
      "SNOW_PCT_PRECIP": float,     # % of precip falling as snow (basin)
      "RUNAVE7100": float           # 30yr mean annual runoff (mm/yr)
    }
"""
from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

GAGES2_URL = "https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip"
ROOT = Path(__file__).resolve().parents[1]
STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
OUT_PATH = ROOT / "data" / "gages2_attrs.json"

# (table, columns we want from it). Every column we don't list gets dropped
# at write time so the JSON stays compact.
TABLES = {
    "conterm_lc06_basin.txt": (
        "FORESTNLCD06", "DEVNLCD06", "WOODYWETNLCD06", "EMERGWETNLCD06",
    ),
    "conterm_soils.txt": (
        "HGA", "HGB", "HGC", "HGD", "AWCAVE", "PERMAVE",
    ),
    "conterm_topo.txt": (
        "ELEV_MEAN_M_BASIN", "SLOPE_PCT",
    ),
    "conterm_hydro.txt": (
        "BFI_AVE", "TOPWET", "RUNAVE7100",
    ),
    "conterm_climate.txt": (
        "PPTAVG_BASIN", "T_AVG_BASIN", "SNOW_PCT_PRECIP",
    ),
}

# Renames so output keys are self-documenting (HGA → HGA_PCT etc.).
RENAMES = {
    "HGA": "HGA_PCT", "HGB": "HGB_PCT", "HGC": "HGC_PCT", "HGD": "HGD_PCT",
}


def _maybe_float(v: str) -> float | None:
    if v is None or str(v).strip() == "" or str(v).strip().upper() in {"NA", "NULL", "-9999"}:
        return None
    try:
        f = float(v)
        if f == -9999.0 or f == -999.0:
            return None
        return f
    except Exception:
        return None


def main() -> int:
    print(f"loading station list from {STATIONS_PATH}")
    ours = {s["id"] for s in json.loads(STATIONS_PATH.read_text())["stations"]}
    print(f"  {len(ours)} target stations")

    # Pull the outer zip into memory; the inner zip
    # `spreadsheets-in-csv-format.zip` is what we actually want.
    print(f"fetching {GAGES2_URL}…")
    with urllib.request.urlopen(GAGES2_URL, timeout=180) as r:
        outer_bytes = r.read()
    print(f"  {len(outer_bytes)/1e6:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer_z:
        names = outer_z.namelist()
        inner = next((n for n in names if n.endswith("spreadsheets-in-csv-format.zip")), None)
        if inner is None:
            print(f"ERROR: no inner csv zip found; saw {names}")
            return 1
        inner_bytes = outer_z.read(inner)
    print(f"inner zip: {len(inner_bytes)/1e6:.1f} MB")

    out: dict[str, dict] = {sid: {} for sid in ours}
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_z:
        for table, cols in TABLES.items():
            try:
                raw = inner_z.read(table).decode("utf-8", errors="replace")
            except KeyError:
                print(f"  WARN: {table} missing in inner zip; skipping")
                continue
            r = csv.DictReader(io.StringIO(raw))
            staid_field = "STAID"
            n_hits = 0
            for row in r:
                sid = str(row.get(staid_field) or "").strip()
                if sid not in ours:
                    continue
                rec = out[sid]
                for c in cols:
                    v = _maybe_float(row.get(c))
                    if v is None:
                        continue
                    out_key = RENAMES.get(c, c)
                    rec[out_key] = v
                n_hits += 1
            print(f"  {table}: {n_hits} matches")

    # Drop stations that got no rows at all so the JSON only has hits.
    out_clean = {sid: rec for sid, rec in out.items() if rec}
    print(f"final coverage: {len(out_clean)}/{len(ours)} ({100*len(out_clean)/len(ours):.1f}%)")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out_clean, indent=0, sort_keys=True))
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
