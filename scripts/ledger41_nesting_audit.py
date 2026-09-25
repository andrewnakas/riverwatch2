#!/usr/bin/env python3
"""Ledger 41: audit the pretraining pool for basins NESTED with CAMELS basins.

The entity dedup screens gauge COORDINATES within 1 km, which catches the same
site under two ids. It cannot catch nesting: a pool gauge can sit far from every
CAMELS gauge while draining a catchment that contains, or is contained by, a
CAMELS one. The pretraining set would then have seen much of the same water the
evaluation basins carry -- not the same series, but the same storms over the
same ground.

The prereg named this as false-positive mode 2 and deferred it to "after Stage 3
passes". Running it BEFORE the finetune result exists is strictly better: a
leakage number measured now cannot be argued with once there is a result to
defend.

Uses the GAGES-II polygons both projects were delineated from, so containment is
measured on real boundaries rather than inferred from gauge distance and area.
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHP_DIR = ROOT / "data" / "gages2" / "boundaries-shapefiles-by-aggeco"
POOL = ROOT / "data" / "ledger41_basins.json"
CAMELS_IDS = ROOT / "data" / "camels_station_meta.json"
OUT = ROOT / "benchmarks" / "ledger41_nesting_audit.json"


def load_polygons(ids):
    frames = []
    want = set(ids)
    for shp in sorted(SHP_DIR.glob("*.shp")):
        g = gpd.read_file(shp)
        col = next(c for c in g.columns if c.upper() in ("GAGE_ID", "STAID", "GAGEID"))
        g["gid"] = g[col].astype(str).str.zfill(8)
        g = g[g["gid"].isin(want)]
        if len(g):
            frames.append(g[["gid", "geometry"]])
    if not frames:
        raise SystemExit("no polygons matched")
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=frames[0].crs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap-frac", type=float, default=0.10,
                    help="flag a pool basin when this fraction of EITHER "
                         "catchment lies inside the other")
    args = ap.parse_args()

    pool_ids = [b["gauge_id"] for b in json.load(open(POOL))["basins"]]
    cam_ids = [s["id"].zfill(8) for s in json.load(open(CAMELS_IDS))["stations"]]

    print(f"loading polygons: {len(pool_ids)} pool, {len(cam_ids)} CAMELS ...")
    pool = load_polygons(pool_ids)
    cam = load_polygons(cam_ids).to_crs(pool.crs)
    print(f"  matched {len(pool)}/{len(pool_ids)} pool and {len(cam)}/{len(cam_ids)} CAMELS")

    # equal-area projection so the overlap fractions mean what they say
    ea = "EPSG:5070"          # NAD83 / Conus Albers
    pool = pool.to_crs(ea)
    cam = cam.to_crs(ea)
    pool["area_m2"] = pool.geometry.area
    cam["area_m2"] = cam.geometry.area

    inter = gpd.overlay(pool, cam, how="intersection", keep_geom_type=False)
    if inter.empty:
        print("no pool/CAMELS polygon intersections at all")
        OUT.write_text(json.dumps({"n_flagged": 0, "flagged": []}, indent=1))
        return
    inter["ia"] = inter.geometry.area
    inter["frac_pool"] = inter["ia"] / inter["area_m2_1"]
    inter["frac_cam"] = inter["ia"] / inter["area_m2_2"]
    inter["frac_max"] = inter[["frac_pool", "frac_cam"]].max(axis=1)

    flagged = inter[inter["frac_max"] >= args.overlap_frac].copy()
    flagged = flagged.sort_values("frac_max", ascending=False)

    n_pool_flagged = flagged["gid_1"].nunique()
    n_cam_touched = flagged["gid_2"].nunique()
    print(f"\npool basins overlapping a CAMELS basin by >= {100*args.overlap_frac:.0f}%: "
          f"**{n_pool_flagged}** of {len(pool)} ({100*n_pool_flagged/len(pool):.1f}%)")
    print(f"CAMELS basins touched: {n_cam_touched} of {len(cam)} "
          f"({100*n_cam_touched/len(cam):.1f}%)")
    for band in (0.9, 0.5, 0.25, 0.10):
        n = inter[inter["frac_max"] >= band]["gid_1"].nunique()
        print(f"  >= {100*band:3.0f}% overlap: {n:4d} pool basins")
    print("\nworst offenders:")
    for _, r in flagged.head(12).iterrows():
        print(f"  pool {r['gid_1']} vs CAMELS {r['gid_2']}: "
              f"{100*r['frac_pool']:5.1f}% of pool / {100*r['frac_cam']:5.1f}% of CAMELS")

    payload = {
        "overlap_frac_bar": args.overlap_frac,
        "n_pool": int(len(pool)), "n_camels": int(len(cam)),
        "n_pool_flagged": int(n_pool_flagged),
        "n_camels_touched": int(n_cam_touched),
        "by_band": {str(b): int(inter[inter["frac_max"] >= b]["gid_1"].nunique())
                    for b in (0.9, 0.5, 0.25, 0.10)},
        "flagged": [{"pool": r["gid_1"], "camels": r["gid_2"],
                     "frac_pool": float(r["frac_pool"]),
                     "frac_camels": float(r["frac_cam"])}
                    for _, r in flagged.iterrows()],
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\n-> {OUT}")
    ids = sorted(flagged["gid_1"].unique())
    (ROOT / "data" / "ledger41_nested_basins.json").write_text(json.dumps(ids, indent=1))
    print(f"-> data/ledger41_nested_basins.json ({len(ids)} ids, for a clean-subset arm)")


if __name__ == "__main__":
    main()
