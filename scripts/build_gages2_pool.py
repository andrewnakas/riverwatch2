#!/usr/bin/env python3
"""Ledger 41, Probe A: census the non-CAMELS GAGES-II pretraining pool.

The ledger-41 design pretrains on non-CAMELS US basins forced with the same
gauge-calibrated products CAMELS uses. This script answers the cheapest
question that can kill that design before any data is built: are there enough
qualifying basins?

A basin qualifies when it

  1. has a GAGES-II boundary polygon (forcings are extracted by coverage-
     weighted mean over that polygon; no polygon, no basin),
  2. is NOT a CAMELS basin under EITHER test -- 8-digit USGS id match OR gauge
     coordinates within LEAK_KM of a CAMELS gauge. The id test alone is not
     sufficient: the ledger-40 audit found 456 CAMELS gauges hiding under
     hysets_* ids in Caravan, which would have leaked 77% of the eval set.
     Here ids come from the same USGS namespace so the geo test is expected to
     add little, but "expected" is not "measured" -- it is run and reported.
  3. has continuous daily discharge over the pretraining window, measured from
     the FlowRec sheet's per-water-year flags rather than a bulk NWIS query,
  4. is not so large that its routing dominates its rainfall-runoff response.

Reference basins are reported separately because CAMELS was itself drawn from
the GAGES-II reference set: matching the finetune DOMAIN, not merely its
forcing product, is the correction ledger 40 was missing.
"""
import argparse
import csv
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEETS = ROOT / "data" / "cache" / "gages2" / "sheets"
SHP_DIR = ROOT / "data" / "gages2" / "boundaries-shapefiles-by-aggeco"
CAMELS_META = ROOT / "data" / "camels_station_meta.json"
CAMELS_IDS = ROOT / "data" / "camels_gauge_ids.json"
OUT = ROOT / "data" / "ledger41_pool.json"

LEAK_KM = 1.0          # entity dedup radius (protocol lock 2)
WY_FIRST, WY_LAST = 1981, 1995   # water years spanned by 1980-10-01..1995-09-30
MIN_WY = 12            # of the 15
MAX_AREA_KM2 = 25000.0


def read_csv(name):
    with open(SHEETS / f"{name}.csv", newline="") as f:
        return {r["STAID"].strip().zfill(8): r for r in csv.DictReader(f)}


def dbf_ids(path):
    """GAGE_IDs present in a shapefile's attribute table.

    Minimal dBase III reader: the DBFs here carry only AREA, PERIMETER and
    GAGE_ID, so pulling in a GIS stack to read three fields is not warranted.
    """
    out = []
    with open(path, "rb") as f:
        nrec, hlen, rlen = struct.unpack("<IHH", f.read(32)[4:12])
        fields, pos = [], 0
        while True:
            fd = f.read(32)
            if fd[:1] in (b"\r", b"", b"\x1a"):
                break
            name = fd[0:11].split(b"\x00")[0].decode("latin-1")
            fields.append((name, fd[16], pos))
            pos += fd[16]
        f.seek(hlen)
        for _ in range(nrec):
            rec = f.read(rlen)
            if not rec or rec[:1] == b"*":      # deleted record
                continue
            for name, ln, off in fields:
                if name.upper() in ("GAGE_ID", "STAID", "GAGEID"):
                    out.append(rec[1 + off:1 + off + ln].decode("latin-1").strip())
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def camels_reference():
    ids = set()
    raw = json.load(open(CAMELS_IDS))
    for v in (raw.values() if isinstance(raw, dict) else [raw]):
        if isinstance(v, list):
            ids.update(str(x).zfill(8) for x in v)
    coords = []
    meta = json.load(open(CAMELS_META))
    for s in meta["stations"]:
        ids.add(str(s["id"]).zfill(8))
        if s.get("lat") is not None and s.get("lon") is not None:
            coords.append((str(s["id"]).zfill(8), float(s["lat"]), float(s["lon"])))
    return ids, coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-wy", type=int, default=MIN_WY)
    ap.add_argument("--max-area", type=float, default=MAX_AREA_KM2)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    basinid, classif, flowrec = read_csv("BasinID"), read_csv("Bas_Classif"), read_csv("FlowRec")

    # polygons: reference set is one file, non-reference is split by ecoregion
    poly = {}
    for shp in sorted(SHP_DIR.glob("*.dbf")):
        tag = "ref" if "bas_ref" in shp.name else "nonref"
        for gid in dbf_ids(shp):
            poly.setdefault(gid.zfill(8), []).append((tag, shp.stem))
    print(f"polygons: {len(poly)} unique gauge ids across {len(list(SHP_DIR.glob('*.dbf')))} shapefiles")

    camels_ids, camels_coords = camels_reference()
    print(f"CAMELS reference set: {len(camels_ids)} ids, {len(camels_coords)} with coordinates")

    wy_cols = [f"wy{y}" for y in range(WY_FIRST, WY_LAST + 1)]
    drop = {k: 0 for k in ("no_polygon", "no_attrs", "camels_id", "camels_geo",
                           "short_record", "too_big", "no_coords", "akhipr")}
    kept, geo_hits = [], []

    for gid, b in basinid.items():
        if gid not in poly:
            drop["no_polygon"] += 1
            continue
        if gid in camels_ids:
            drop["camels_id"] += 1
            continue
        try:
            lat, lon = float(b["LAT_GAGE"]), float(b["LNG_GAGE"])
        except (ValueError, KeyError):
            drop["no_coords"] += 1
            continue
        near = min(((haversine_km(lat, lon, cl, cn), cid)
                    for cid, cl, cn in camels_coords), default=(9e9, None))
        if near[0] <= LEAK_KM:
            drop["camels_geo"] += 1
            geo_hits.append({"gauge": gid, "camels": near[1], "km": round(near[0], 3)})
            continue
        fr = flowrec.get(gid)
        if fr is None:
            drop["no_attrs"] += 1
            continue
        n_wy = sum(1 for c in wy_cols if str(fr.get(c, "")).strip() in ("1", "1.0"))
        if n_wy < args.min_wy:
            drop["short_record"] += 1
            continue
        try:
            area = float(b["DRAIN_SQKM"])
        except (ValueError, KeyError):
            area = float("nan")
        if not (area == area) or area > args.max_area:
            drop["too_big"] += 1
            continue
        cl = classif.get(gid, {})
        try:
            hdi = float(cl.get("HYDRO_DISTURB_INDX", "") or "nan")
        except ValueError:
            hdi = float("nan")
        kept.append({
            "gauge_id": gid, "lat": lat, "lon": lon, "area_km2": area,
            "n_wy": n_wy, "poly": poly[gid][0][0], "poly_file": poly[gid][0][1],
            "klass": (cl.get("CLASS") or "").strip(),
            "ecoregion": (cl.get("AGGECOREGION") or "").strip(),
            "hydro_disturb_indx": None if hdi != hdi else hdi,
            "huc02": (b.get("HUC02") or "").strip(),
            "state": (b.get("STATE") or "").strip(),
            "km_to_nearest_camels": round(near[0], 3),
        })

    ref = [k for k in kept if k["klass"].lower().startswith("ref")]
    nonref = [k for k in kept if not k["klass"].lower().startswith("ref")]

    print("\ndropped:")
    for k, v in sorted(drop.items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v}")
    print(f"\nQUALIFYING POOL: {len(kept)}  (reference {len(ref)}, non-reference {len(nonref)})")
    print(f"  geo-only leakage catches beyond the id rule: {len(geo_hits)}")
    for g in geo_hits[:10]:
        print(f"    {g['gauge']} ~ CAMELS {g['camels']} at {g['km']} km")

    from collections import Counter
    eco = Counter(k["ecoregion"] for k in kept)
    print("\n  by ecoregion:")
    for e, c in eco.most_common():
        r = sum(1 for k in ref if k["ecoregion"] == e)
        print(f"    {e:16s} {c:5d}  (ref {r})")

    areas = sorted(k["area_km2"] for k in kept)
    q = lambda p: areas[int(p * (len(areas) - 1))]
    print(f"\n  area km2: min {areas[0]:.1f}  q25 {q(.25):.0f}  median {q(.5):.0f} "
          f" q75 {q(.75):.0f}  max {areas[-1]:.0f}")
    full = sum(1 for k in kept if k["n_wy"] == WY_LAST - WY_FIRST + 1)
    print(f"  full {WY_LAST - WY_FIRST + 1}-water-year records: {full}/{len(kept)}")

    payload = {
        "_meta": {
            "purpose": "ledger-41 domain-aligned pretraining pool (Probe A census)",
            "window_water_years": [WY_FIRST, WY_LAST],
            "min_water_years": args.min_wy,
            "max_area_km2": args.max_area,
            "leak_dedup_km": LEAK_KM,
            "camels_ids_excluded": len(camels_ids),
            "dropped": drop,
            "geo_only_leak_catches": geo_hits,
            "n_total": len(kept), "n_ref": len(ref), "n_nonref": len(nonref),
        },
        "basins": sorted(kept, key=lambda k: k["gauge_id"]),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
