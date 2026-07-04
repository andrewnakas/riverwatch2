#!/usr/bin/env python3
"""v17: minimal registry metadata for the CAMELS basins.

The mb-LSTM trainer reads lat/lon/alt_ft/drain_area_sqmi per station from
data/stations_40_enriched.json — but only 60 of the 671 CAMELS basins are in
that production registry, so a CAMELS benchmark train median-fills the
lat/lon/alt/area static features for the rest. Rather than appending inert
entries to the production registry (which app/server.py and the forcing
fetchers iterate), this writes a supplemental file that
scripts/train_mblstm.py merges in as a fallback:

    data/camels_station_meta.json
      {"stations": [{"id", "lat", "lon", "alt_ft", "drain_area_sqmi"}, ...]}

Sources (CAMELS basin_dataset_public_v1p2 on the SD card):
  - basin_metadata/gauge_information.txt: LAT, LONG, area km2 (-> sqmi)
  - basin_mean_forcing/daymet/<huc>/<id>_lump_cida_forcing_leap.txt header
    line 2: gauge elevation in meters (-> alt_ft)

Run once locally; commit the ~60 KB output:

    python scripts/build_camels_station_meta.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMELS_IDS_PATH = ROOT / "data" / "camels_gauge_ids.json"
OUT_PATH = ROOT / "data" / "camels_station_meta.json"
CAMELS_RAW = Path("/Volumes/STORAGE_SD/riverwatch2_data/camels_raw/basin_dataset_public_v1p2")
GAUGE_INFO = CAMELS_RAW / "basin_metadata" / "gauge_information.txt"
DAYMET_DIR = CAMELS_RAW / "basin_mean_forcing" / "daymet"

KM2_TO_SQMI = 0.386102
M_TO_FT = 3.28084


def daymet_elev_m(sid: str) -> float | None:
    """Gauge elevation (m) from the Daymet forcing header: line 1 latitude,
    line 2 elevation m, line 3 basin area m^2."""
    hits = sorted(DAYMET_DIR.glob(f"*/{sid}_lump_cida_forcing_leap.txt"))
    if not hits:
        return None
    with hits[0].open() as f:
        f.readline()
        try:
            return float(f.readline().strip())
        except ValueError:
            return None


def main() -> int:
    cam = json.loads(CAMELS_IDS_PATH.read_text())
    wanted = {sid for k, v in cam.items() if not k.startswith("_") for sid in v}
    print(f"{len(wanted)} CAMELS ids")

    # Tab-delimited: HUC_02, GAGE_ID, GAGE_NAME, LAT, LONG, area km^2.
    stations: list[dict] = []
    for line in GAUGE_INFO.read_text().splitlines()[1:]:
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 6 or parts[1] not in wanted:
            continue
        sid = parts[1]
        elev_m = daymet_elev_m(sid)
        stations.append({
            "id": sid,
            "lat": float(parts[3]),
            "lon": float(parts[4]),
            "alt_ft": round(elev_m * M_TO_FT, 2) if elev_m is not None else None,
            "drain_area_sqmi": round(float(parts[5]) * KM2_TO_SQMI, 3),
        })

    stations.sort(key=lambda s: s["id"])
    found = {s["id"] for s in stations}
    missing = sorted(wanted - found)
    no_elev = [s["id"] for s in stations if s["alt_ft"] is None]
    print(f"metadata rows: {len(stations)}; missing from gauge_information: "
          f"{missing or 'none'}; no daymet elevation: {no_elev or 'none'}")
    OUT_PATH.write_text(json.dumps({
        "description": "Supplemental registry metadata for CAMELS basins not in "
                       "stations_40_enriched.json; merged by scripts/train_mblstm.py. "
                       "Built by scripts/build_camels_station_meta.py from "
                       "basin_dataset_public_v1p2 (gauge_information.txt + Daymet headers).",
        "stations": stations,
    }, indent=0, sort_keys=True))
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
