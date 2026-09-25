#!/usr/bin/env python3
"""Ledger 41: select the final pretraining pool.

Intersects the Probe-A GAGES-II census with HYSETS, which supplies
pre-aggregated per-catchment forcings from two independent gauge-based
sources. Applies the leakage dedup a SECOND time in HYSETS' own coordinate
space -- the Probe-A pass tested GAGES-II gauge coordinates, and a basin can
carry a different Official_ID and different coordinates in a different
aggregator. 644 of the 671 CAMELS basins are present in HYSETS, so this is not
a hypothetical risk: it is the ledger-40 hysets trap in a new coordinate
system.

Admission of non-reference basins is set by CAMELS itself rather than by
taste: every CAMELS basin is GAGES-II Ref class, and their HYDRO_DISTURB_INDX
runs 1..29 (median 9, q90 15). Basins at or below the CAMELS q90 are inside
the disturbance range the finetune domain already spans; beyond it the prior
would be learning regulated-river behaviour the finetune set does not contain.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "ledger41_pool.json"
PROPS = ROOT / "data" / "hysets" / "HYSETS_watershed_properties.txt"
CAMELS_META = ROOT / "data" / "camels_station_meta.json"
SHEETS = ROOT / "data" / "cache" / "gages2" / "sheets"
OUT = ROOT / "data" / "ledger41_basins.json"
LEAK_KM = 1.0


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdi-quantile", type=float, default=0.90,
                    help="admit basins at or below this quantile of the CAMELS "
                         "hydrologic-disturbance distribution")
    ap.add_argument("--area-tol", type=float, default=0.5,
                    help="max |HYSETS area / GAGES-II area - 1| before a basin is "
                         "dropped as a boundary disagreement")
    args = ap.parse_args()

    pool = {b["gauge_id"]: b for b in json.load(open(POOL))["basins"]}
    meta = json.load(open(CAMELS_META))["stations"]
    camels_ids = {str(s["id"]).zfill(8) for s in meta}
    camels_pts = [(str(s["id"]).zfill(8), float(s["lat"]), float(s["lon"]))
                  for s in meta if s.get("lat") is not None]

    classif = {r["STAID"].strip().zfill(8): r
               for r in csv.DictReader(open(SHEETS / "Bas_Classif.csv"))}
    hdi_camels = sorted(float(classif[g]["HYDRO_DISTURB_INDX"])
                        for g in camels_ids
                        if g in classif and classif[g]["HYDRO_DISTURB_INDX"])
    hdi_bar = hdi_camels[int(args.hdi_quantile * (len(hdi_camels) - 1))]
    print(f"CAMELS HYDRO_DISTURB_INDX q{args.hdi_quantile:.2f} = {hdi_bar:.0f} "
          f"(range {hdi_camels[0]:.0f}..{hdi_camels[-1]:.0f}) -> admission bar")

    hy = {}
    for p in csv.DictReader(open(PROPS)):
        if p["Source"].strip().upper() != "USGS":
            continue
        gid = p["Official_ID"].strip().zfill(8)
        hy[gid] = p

    drop = Counter()
    kept, geo_hits = [], []
    for gid, b in pool.items():
        h = hy.get(gid)
        if h is None:
            drop["not_in_hysets"] += 1
            continue
        if gid in camels_ids:            # belt and braces; Probe A already did this
            drop["camels_id"] += 1
            continue
        try:
            hlat = float(h["Hydrometric_station_latitude"])
            hlon = float(h["Hydrometric_station_longitude"])
        except ValueError:
            hlat, hlon = float(h["Centroid_Lat_deg_N"]), float(h["Centroid_Lon_deg_E"])
        near = min((haversine_km(hlat, hlon, cl, cn), cid) for cid, cl, cn in camels_pts)
        if near[0] <= LEAK_KM:
            drop["camels_geo_hysets_space"] += 1
            geo_hits.append({"gauge": gid, "camels": near[1], "km": round(near[0], 3)})
            continue
        hdi = b["hydro_disturb_indx"]
        if hdi is None or hdi > hdi_bar:
            drop["too_disturbed"] += 1
            continue
        try:
            harea = float(h["Drainage_Area_km2"])
        except ValueError:
            drop["no_hysets_area"] += 1
            continue
        if harea <= 0 or abs(harea / b["area_km2"] - 1.0) > args.area_tol:
            # the two projects delineated the same gauge differently; a forcing
            # aggregated over the wrong polygon is a silent error, so drop it
            drop["area_disagreement"] += 1
            continue
        kept.append({**b,
                     "hysets_watershed_id": int(h["Watershed_ID"]),
                     "hysets_area_km2": harea,
                     "hysets_lat": hlat, "hysets_lon": hlon,
                     "area_ratio": round(harea / b["area_km2"], 4),
                     "hysets_permeability_logk_m2": h.get("Permeability_logk_m2", ""),
                     "hysets_porosity_frac": h.get("Porosity_frac", "")})

    print("\ndropped from the Probe-A pool:")
    for k, v in drop.most_common():
        print(f"  {k:26s} {v}")
    ref = [k for k in kept if k["klass"].lower().startswith("ref")]
    print(f"\nFINAL POOL: {len(kept)} basins  (GAGES-II reference {len(ref)}, "
          f"other {len(kept) - len(ref)})")
    print(f"  full 15-water-year discharge records: "
          f"{sum(1 for k in kept if k['n_wy'] == 15)}")
    print(f"  extra leakage catches in HYSETS coordinate space: {len(geo_hits)}")
    for g in geo_hits[:8]:
        print(f"    {g['gauge']} ~ CAMELS {g['camels']} at {g['km']} km")

    eco = Counter(k["ecoregion"] for k in kept)
    ceco = Counter(classif[g]["AGGECOREGION"].strip() for g in camels_ids if g in classif)
    tp, tc = sum(eco.values()), sum(ceco.values())
    print(f"\n{'ecoregion':16s} {'POOL%':>7s} {'CAMELS%':>8s}  {'delta':>6s}")
    for e in sorted(set(eco) | set(ceco), key=lambda x: -ceco.get(x, 0)):
        a, b_ = 100 * eco.get(e, 0) / tp, 100 * ceco.get(e, 0) / tc
        print(f"{e:16s} {a:6.1f}% {b_:7.1f}%  {a - b_:+6.1f}")

    payload = {"_meta": {"purpose": "ledger-41 pretraining pool (final selection)",
                         "source": "Probe-A GAGES-II census INTERSECT HYSETS USGS basins",
                         "hdi_quantile": args.hdi_quantile, "hdi_bar": hdi_bar,
                         "leak_dedup_km": LEAK_KM, "area_tol": args.area_tol,
                         "dropped": dict(drop),
                         "geo_leak_catches_hysets_space": geo_hits,
                         "n_total": len(kept), "n_gagesii_reference": len(ref)},
               "basins": sorted(kept, key=lambda k: k["gauge_id"])}
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
