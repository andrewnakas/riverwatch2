import sys
p = "aorc_exact_extract.py"
s = open(p).read()
if "GAGES2_SHP" in s:
    print("already patched"); sys.exit()

old = '''def get_polygon(basin, cache=Path("data/basin_polygons")):
    """GAGES-II-equivalent watershed polygon from the USGS NLDI service."""'''
new = '''GAGES2_SHP = "data/gages2/boundaries-shapefiles-by-aggeco/bas_ref_all.shp"


def get_polygon_gages2(basin, shp=GAGES2_SHP):
    """The ACTUAL GAGES-II polygon the HydroShare product was built from.

    Falcone 2011, ScienceBase 631405bbd34e36012efa304a. This is the input we
    were missing: NLDI serves the modern NHDPlus watershed, which differs from
    GAGES-II at the ridgelines, and in orographic terrain a different ridgeline
    admits different (much wetter) cells. Using NLDI, precipitation reproduced
    the published series at ratio 0.858 sd 0.051 -- basin-varying, so not
    correctable by a constant. Temperature and pressure were near-exact, which
    is what pointed at the polygon rather than the aggregation.

    Reprojected to EPSG:4326 to match the AORC grid.
    """
    import geopandas as gpd
    g = gpd.read_file(shp)
    idcol = next(c for c in g.columns if c.upper() in ("GAGE_ID", "STAID", "GAGEID"))
    row = g[g[idcol].astype(str).str.zfill(8) == str(basin).zfill(8)]
    if row.empty:
        raise KeyError(f"{basin} not in {shp}")
    row = row.to_crs("EPSG:4326").copy()
    row["divide_id"] = str(basin).zfill(8)
    return row[["divide_id", "geometry"]]


def get_polygon(basin, cache=Path("data/basin_polygons")):
    """Basin polygon. Prefers the real GAGES-II shapefile when present."""
    if Path(GAGES2_SHP).exists():
        try:
            return get_polygon_gages2(basin)
        except KeyError:
            pass  # fall back to NLDI for basins GAGES-II lacks'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched: GAGES-II polygons preferred over NLDI")
