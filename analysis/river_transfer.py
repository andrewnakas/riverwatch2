"""Pure geometry and hydrology for the map's click-anywhere runoff estimate.

The browser does the same arithmetic in `app/static/river_transfer.js`. It lives
here as well, and is tested here, because `pages.yml`'s `test` job runs pytest
only and hard-gates the deploy -- so this is the copy the gate can actually
check, and `tests/test_river_transfer.py` pins the values the JS must reproduce.

WHAT THE CLICK PATH DOES. USGS NLDI snaps the clicked point to an NHD flowline
and returns the upstream basin polygon (with NO area property, so area is
computed here) plus the gauges reachable up- and downstream along the network.
Runoff at the clicked point is then the nearest suitable gauge's forecast scaled
by the drainage-area ratio -- the classic ungauged transfer.

WHAT IT IS NOT. Area-ratio transfer carries no new hydrology: it assumes the
donor's specific discharge (runoff per unit area) applies at the target. It
degrades as the area ratio departs from 1 and as the donor gets further away,
and it is NOT the no-q model. The UI must show the measured skill and name the
donor, which is what `analysis/measure_area_ratio_transfer.py` is for.

⚠️ One measured caveat, correctly scoped: NLDI serves the modern NHDPlus
watershed, which differs from the GAGES-II polygons the models trained on at
the ridgelines (precipitation reproduced the published series at ratio 0.858
sd 0.051, basin-varying and so not correctable by a constant). That bites a
path that FORCES a model with NLDI-delineated areal weather. It does NOT bite
area-ratio transfer, which uses only areas and network topology.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

EARTH_RADIUS_KM = 6371.0088          # IUGG mean radius
# 1 ft^3 = 0.028316846592 m^3 exactly.
CFS_TO_M3S = 0.028316846592
SECONDS_PER_DAY = 86400.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _segment_distance_km(lat: float, lon: float,
                         lat_a: float, lon_a: float,
                         lat_b: float, lon_b: float) -> float:
    """Distance from a point to a segment, on a local equirectangular plane.

    Flat-earth is appropriate here: NHD flowline segments are ~100 m to a few km
    and the answer only has to decide "is this click on a river", so sub-metre
    fidelity is irrelevant. Longitude is scaled by cos(latitude) about the
    query point so the plane is locally equal-area.
    """
    kx = math.cos(math.radians(lat)) * 111.320
    ky = 110.574
    px, py = lon * kx, lat * ky
    ax, ay = lon_a * kx, lat_a * ky
    bx, by = lon_b * kx, lat_b * ky
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den <= 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / den
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def point_to_polyline_km(lat: float, lon: float,
                         coords: Sequence[Sequence[float]]) -> Optional[float]:
    """Shortest distance from a point to a GeoJSON LineString.

    `coords` is [[lon, lat], ...] -- GeoJSON order, which is the opposite of the
    argument order here. Getting that backwards silently returns a plausible
    number, so it is spelled out. Returns None for empty input.
    """
    pts = [c for c in coords if c is not None and len(c) >= 2]
    if not pts:
        return None
    if len(pts) == 1:
        return haversine_km(lat, lon, pts[0][1], pts[0][0])
    best = float("inf")
    for (lon_a, lat_a), (lon_b, lat_b) in zip(pts, pts[1:]):
        d = _segment_distance_km(lat, lon, lat_a, lon_a, lat_b, lon_b)
        if d < best:
            best = d
    return best


def _ring_area_km2(ring: Sequence[Sequence[float]]) -> float:
    """Signed spherical area of one ring of [lon, lat] vertices."""
    if ring is None or len(ring) < 3:
        return 0.0
    total = 0.0
    n = len(ring)
    for i in range(n):
        lon1, lat1 = ring[i][0], ring[i][1]
        lon2, lat2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        total += math.radians(lon2 - lon1) * (
            2.0 + math.sin(math.radians(lat1)) + math.sin(math.radians(lat2)))
    return total * EARTH_RADIUS_KM * EARTH_RADIUS_KM / 2.0


def geojson_area_km2(geometry: dict) -> Optional[float]:
    """Area of a GeoJSON Polygon or MultiPolygon, in km^2.

    Holes (interior rings) are subtracted by the signed-area sum, since a
    correctly wound GeoJSON ring reverses orientation for holes. NLDI returns
    the basin with no `properties.area`, so this is the only source of area for
    a clicked point.
    """
    if not isinstance(geometry, dict):
        return None
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return None
    if gtype == "Polygon":
        polys = [coords]
    elif gtype == "MultiPolygon":
        polys = coords
    else:
        return None
    total = 0.0
    for poly in polys:
        for i, ring in enumerate(poly):
            a = _ring_area_km2(ring)
            total += abs(a) if i == 0 else -abs(a)
    return abs(total)


def cfs_to_mm_per_day(q_cfs: float, area_km2: float) -> Optional[float]:
    """Specific discharge (runoff depth) in mm/day.

    This is the quantity that actually transfers between catchments of different
    size, and the honest headline for an ungauged point.
    """
    if area_km2 is None or not (area_km2 > 0) or q_cfs is None:
        return None
    m3_per_day = float(q_cfs) * CFS_TO_M3S * SECONDS_PER_DAY
    return m3_per_day / (float(area_km2) * 1.0e6) * 1000.0


def mm_per_day_to_cfs(mm_day: float, area_km2: float) -> Optional[float]:
    if area_km2 is None or not (area_km2 > 0) or mm_day is None:
        return None
    m3_per_day = float(mm_day) / 1000.0 * (float(area_km2) * 1.0e6)
    return m3_per_day / SECONDS_PER_DAY / CFS_TO_M3S


SQMI_TO_KM2 = 2.589988110336


def sqmi_to_km2(a: Optional[float]) -> Optional[float]:
    return None if a is None else float(a) * SQMI_TO_KM2


def area_ratio_penalty(target_km2: float, donor_km2: float) -> Optional[float]:
    """|log(area ratio)| -- 0 for equal areas, ln(2) for a 2x mismatch.

    Symmetric in the ratio by construction, which is what we want: a donor
    twice the size is as bad as one half the size.
    """
    if not target_km2 or not donor_km2 or target_km2 <= 0 or donor_km2 <= 0:
        return None
    return abs(math.log(float(target_km2) / float(donor_km2)))


def select_donor(candidates: Iterable[dict], target_km2: float,
                 *, max_ratio: float = 10.0,
                 upstream_bonus: float = 0.0) -> Optional[dict]:
    """Pick the gauge whose drainage area best matches the clicked point.

    `candidates` are dicts with at least `area_km2`, optionally `direction`
    ("upstream"/"downstream") and `distance_km`. Scored by |log area ratio|
    alone; distance breaks ties only.

    ⛔ `upstream_bonus` defaults to 0. It was 0.15, on the reasoning that an
    upstream gauge is a subset of the clicked point's catchment so its runoff is
    physically part of what passes the click. The measurement does not support
    that: over 322 scored pairs
    (`analysis/measure_area_ratio_transfer.py`, `benchmarks/river_transfer_study.json`)
    upstream donors reached median NSE **0.6880** against downstream **0.7459**.
    That contrast is NOT paired -- the two sets differ systematically in area
    ratio -- so it does not license a downstream bonus either. The honest
    response to "my heuristic is unsupported" is to remove it, not to replace it
    with the opposite unsupported one. The parameter survives so a PAIRED
    measurement could set it later.

    Returns None when nothing is within `max_ratio`, rather than returning a
    bad donor -- a refusal is honest, a 20x extrapolation is not.
    """
    best, best_score = None, float("inf")
    for c in candidates or []:
        a = c.get("area_km2")
        pen = area_ratio_penalty(target_km2, a)
        if pen is None or pen > math.log(max_ratio):
            continue
        score = pen - (upstream_bonus if c.get("direction") == "upstream" else 0.0)
        score += 1e-4 * float(c.get("distance_km") or 0.0)
        if score < best_score:
            best, best_score = c, score
    return best


def transfer_by_area_ratio(donor_q_cfs: Sequence[float],
                           donor_km2: float, target_km2: float,
                           *, exponent: float = 1.0) -> Optional[list]:
    """Scale a donor hydrograph to the target catchment.

    Q_target = Q_donor * (A_target / A_donor) ** exponent

    `exponent` is 1.0 -- equal specific discharge, the standard drainage-area
    ratio method and the only choice that needs no fitting. It is a parameter so
    that a measured exponent can be substituted later if
    `analysis/measure_area_ratio_transfer.py` finds one, but it must NOT be
    tuned on the window it is reported on.
    """
    if not donor_km2 or not target_km2 or donor_km2 <= 0 or target_km2 <= 0:
        return None
    k = (float(target_km2) / float(donor_km2)) ** float(exponent)
    out = []
    for q in donor_q_cfs or []:
        out.append(None if q is None else float(q) * k)
    return out
