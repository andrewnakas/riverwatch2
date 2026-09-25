#!/usr/bin/env python3
"""Ledger 45 track 1: how many of the 531 benchmark CAMELS basins are NESTED
in each other?

The GNN-routing lead (HESS 2026, LamaH-CE) needs a river network to route over:
upstream gauged basins feeding downstream ones. CAMELS-US was selected for
independent, minimally-impacted catchments, so the honest prior is that the
graph is empty. This measures it on the real GAGES-II polygons -- the same
geometry source ledger 41 used for the pool-vs-CAMELS audit
(scripts/ledger41_nesting_audit.py), but here CAMELS-internal: 531 x 531.

A pair is NESTED when the smaller basin's polygon is >= 50% inside the larger
one (counts at 0.9/0.5/0.25/0.1 reported so the answer is threshold-robust).

Sanity check: projected polygon areas must correlate with area_gages2 from
data/camels_attrs.json -- a bad CRS or a bad ID join would silently fake the
answer.

Inputs:  --ids-areas JSON {"ids": [...531...], "area_gages2": {id: km2}}
         --shp-dir   GAGES-II boundaries-shapefiles-by-aggeco
Output:  --out JSON (ledger45_camels_nesting.json)
"""
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

THRESHOLDS = (0.9, 0.5, 0.25, 0.10)
NESTED_BAR = 0.5
EA_CRS = "EPSG:5070"  # NAD83 / Conus Albers, equal-area (same as ledger 41)


def load_polygons(shp_dir, ids):
    frames = []
    want = set(ids)
    for shp in sorted(Path(shp_dir).glob("*.shp")):
        g = gpd.read_file(shp)
        col = next(c for c in g.columns if c.upper() in ("GAGE_ID", "STAID", "GAGEID"))
        g["gid"] = g[col].astype(str).str.zfill(8)
        g = g[g["gid"].isin(want)]
        if len(g):
            frames.append(g[["gid", "geometry"]].to_crs(EA_CRS))
    if not frames:
        raise SystemExit("no polygons matched")
    out = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=EA_CRS)
    dup = gdf["gid"].duplicated(keep="first")
    if dup.any():
        print(f"note: dropping {dup.sum()} duplicate polygon rows (same gid in >1 aggeco file)")
        gdf = gdf[~dup].reset_index(drop=True)
    return gdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-areas", required=True)
    ap.add_argument("--shp-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.load(open(args.ids_areas))
    ids = meta["ids"]
    ref_area = meta["area_gages2"]  # km2
    print(f"benchmark basins: {len(ids)}")

    gdf = load_polygons(args.shp_dir, ids)
    matched = set(gdf["gid"])
    missing = sorted(set(ids) - matched)
    print(f"polygons matched: {len(gdf)}/{len(ids)}; missing: {missing}")

    gdf["geometry"] = gdf.geometry.buffer(0)  # heal invalid rings
    gdf["area_km2"] = gdf.geometry.area / 1e6

    # ---- sanity: polygon area vs area_gages2 ------------------------------
    ref = np.array([ref_area[g] for g in gdf["gid"]], float)
    poly = gdf["area_km2"].to_numpy()
    r = float(np.corrcoef(np.log(ref), np.log(poly))[0, 1])
    ratio = poly / ref
    gross = [
        {"gid": g, "poly_km2": round(float(p), 2), "area_gages2_km2": round(float(a), 2),
         "ratio": round(float(p / a), 3)}
        for g, p, a in zip(gdf["gid"], poly, ref) if not 0.9 <= p / a <= 1.1
    ]
    print(f"sanity: log-area correlation r={r:.6f}; "
          f"median ratio {np.median(ratio):.4f}; gross (>10%) mismatches: {len(gross)}")
    if r < 0.99:
        print("WARNING: area correlation is weak -- do not trust the result")

    # ---- pairwise intersections via spatial index -------------------------
    sidx = gdf.sindex
    geoms = gdf.geometry.values
    gids = gdf["gid"].to_numpy()
    areas = gdf["area_km2"].to_numpy()
    pairs = []
    seen = set()
    for i in range(len(gdf)):
        for j in sidx.query(geoms[i], predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            inter = geoms[i].intersection(geoms[j]).area / 1e6
            if inter <= 0:
                continue
            fa, fb = inter / areas[i], inter / areas[j]
            small_i = i if areas[i] <= areas[j] else j
            large_i = j if small_i == i else i
            pairs.append({
                "small": gids[small_i], "large": gids[large_i],
                "area_small_km2": round(float(areas[small_i]), 2),
                "area_large_km2": round(float(areas[large_i]), 2),
                "inter_km2": round(float(inter), 4),
                "frac_small": round(float(inter / areas[small_i]), 6),
                "frac_large": round(float(inter / areas[large_i]), 6),
            })
    pairs.sort(key=lambda p: -p["frac_small"])
    print(f"\nintersecting pairs (any overlap > 0): {len(pairs)}")

    by_band = {}
    for t in THRESHOLDS:
        sel = [p for p in pairs if p["frac_small"] >= t]
        n_basins = len({x for p in sel for x in (p["small"], p["large"])})
        by_band[str(t)] = {"n_pairs": len(sel), "n_basins_involved": n_basins}
        print(f"  frac_small >= {t:4}: {len(sel):3d} pairs, {n_basins:3d} basins involved")

    # ---- containment graph at the NESTED bar ------------------------------
    nested = [p for p in pairs if p["frac_small"] >= NESTED_BAR]
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in nested:
        ra, rb = find(p["small"]), find(p["large"])
        if ra != rb:
            parent[ra] = rb
    comps = Counter(find(x) for x in parent)
    sizes = sorted(comps.values(), reverse=True)
    involved = sorted({x for p in nested for x in (p["small"], p["large"])})
    print(f"\nNESTED (frac_small >= {NESTED_BAR}): {len(nested)} pairs, "
          f"{len(involved)} of {len(ids)} basins involved "
          f"({100 * len(involved) / len(ids):.1f}%)")
    print(f"connected components (size >= 2): {len(sizes)}; sizes: {sizes}")

    payload = {
        "question": "CAMELS-internal nesting among the 531 benchmark basins",
        "n_basins": len(ids),
        "n_polygons_matched": int(len(gdf)),
        "missing_polygon_ids": missing,
        "geometry_source": str(Path(args.shp_dir).resolve()),
        "equal_area_crs": EA_CRS,
        "nested_definition": f"smaller basin >= {NESTED_BAR} contained in larger",
        "n_intersecting_pairs_any": len(pairs),
        "by_band": by_band,
        "nested_pairs": nested,
        "n_nested_pairs": len(nested),
        "n_basins_involved": len(involved),
        "basins_involved": involved,
        "n_components_ge2": len(sizes),
        "component_sizes": sizes,
        "component_size_hist": dict(Counter(sizes)),
        "all_intersecting_pairs": pairs,
        "area_sanity": {
            "log_area_pearson_r": r,
            "median_poly_over_gages2_ratio": float(np.median(ratio)),
            "n_gross_mismatch_gt10pct": len(gross),
            "gross_mismatches": gross,
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
