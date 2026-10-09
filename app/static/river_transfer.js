/* Click-anywhere runoff arithmetic — the browser mirror of
 * analysis/river_transfer.py.
 *
 * The Python copy is the one the CI gate tests (pages.yml's `test` job runs
 * pytest only). This file must reproduce tests/river_transfer_fixture.json to
 * within 1e-6 relative; `node tests/check_river_transfer_js.mjs` verifies that.
 * If you change one copy, change both and re-run that check.
 *
 * WHAT THIS IS FOR. USGS NLDI snaps a clicked point to an NHD flowline and
 * returns the upstream basin polygon (with NO area property) plus the gauges
 * reachable along the network. Runoff at the click is the best-matched gauge's
 * forecast scaled by the drainage-area ratio.
 *
 * WHAT IT IS NOT. Area-ratio transfer carries no new hydrology: it assumes the
 * donor's runoff per unit area applies at the target. It degrades as the area
 * ratio leaves 1, and it is not the no-q model. Always show the donor and the
 * measured skill.
 */
(function (root) {
  "use strict";

  var EARTH_RADIUS_KM = 6371.0088;      // IUGG mean radius
  var CFS_TO_M3S = 0.028316846592;      // 1 ft^3 = 0.028316846592 m^3 exactly
  var SECONDS_PER_DAY = 86400.0;
  var SQMI_TO_KM2 = 2.589988110336;

  function rad(d) { return (d * Math.PI) / 180.0; }

  function haversineKm(lat1, lon1, lat2, lon2) {
    var p1 = rad(lat1), p2 = rad(lat2);
    var dp = p2 - p1, dl = rad(lon2 - lon1);
    var a = Math.sin(dp / 2) * Math.sin(dp / 2) +
            Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) * Math.sin(dl / 2);
    return 2.0 * EARTH_RADIUS_KM * Math.asin(Math.min(1.0, Math.sqrt(a)));
  }

  /* Point-to-segment on a local equirectangular plane. Flat-earth is fine: NHD
   * segments are ~100 m to a few km and the answer only decides "is this click
   * on a river". Longitude is scaled by cos(lat) about the query point. */
  function segmentDistanceKm(lat, lon, latA, lonA, latB, lonB) {
    var kx = Math.cos(rad(lat)) * 111.320, ky = 110.574;
    var px = lon * kx, py = lat * ky;
    var ax = lonA * kx, ay = latA * ky;
    var bx = lonB * kx, by = latB * ky;
    var dx = bx - ax, dy = by - ay;
    var den = dx * dx + dy * dy;
    if (den <= 0.0) return Math.hypot(px - ax, py - ay);
    var t = ((px - ax) * dx + (py - ay) * dy) / den;
    t = Math.max(0.0, Math.min(1.0, t));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  /* `coords` is GeoJSON order [[lon, lat], ...] — the OPPOSITE of the first two
   * arguments. Swapping them silently returns a plausible number. */
  function pointToPolylineKm(lat, lon, coords) {
    if (!coords || !coords.length) return null;
    var pts = coords.filter(function (c) { return c && c.length >= 2; });
    if (!pts.length) return null;
    if (pts.length === 1) return haversineKm(lat, lon, pts[0][1], pts[0][0]);
    var best = Infinity;
    for (var i = 0; i < pts.length - 1; i++) {
      var d = segmentDistanceKm(lat, lon, pts[i][1], pts[i][0], pts[i + 1][1], pts[i + 1][0]);
      if (d < best) best = d;
    }
    return best;
  }

  function ringAreaKm2(ring) {
    if (!ring || ring.length < 3) return 0.0;
    var total = 0.0, n = ring.length;
    for (var i = 0; i < n; i++) {
      var a = ring[i], b = ring[(i + 1) % n];
      total += rad(b[0] - a[0]) * (2.0 + Math.sin(rad(a[1])) + Math.sin(rad(b[1])));
    }
    return (total * EARTH_RADIUS_KM * EARTH_RADIUS_KM) / 2.0;
  }

  /* Polygon / MultiPolygon area in km^2. NLDI returns the basin with no
   * properties.area, so this is the only source of area for a clicked point. */
  function geojsonAreaKm2(geometry) {
    if (!geometry || typeof geometry !== "object") return null;
    var coords = geometry.coordinates;
    if (!coords || !coords.length) return null;
    var polys;
    if (geometry.type === "Polygon") polys = [coords];
    else if (geometry.type === "MultiPolygon") polys = coords;
    else return null;
    var total = 0.0;
    for (var p = 0; p < polys.length; p++) {
      for (var r = 0; r < polys[p].length; r++) {
        var a = Math.abs(ringAreaKm2(polys[p][r]));
        total += (r === 0) ? a : -a;      // interior rings are holes
      }
    }
    return Math.abs(total);
  }

  /* Specific discharge (runoff depth) in mm/day — the quantity that actually
   * transfers between catchments of different size. */
  function cfsToMmPerDay(qCfs, areaKm2) {
    if (!(areaKm2 > 0) || qCfs === null || qCfs === undefined) return null;
    var m3PerDay = qCfs * CFS_TO_M3S * SECONDS_PER_DAY;
    return (m3PerDay / (areaKm2 * 1.0e6)) * 1000.0;
  }

  function mmPerDayToCfs(mmDay, areaKm2) {
    if (!(areaKm2 > 0) || mmDay === null || mmDay === undefined) return null;
    var m3PerDay = (mmDay / 1000.0) * (areaKm2 * 1.0e6);
    return m3PerDay / SECONDS_PER_DAY / CFS_TO_M3S;
  }

  function sqmiToKm2(a) {
    return (a === null || a === undefined) ? null : a * SQMI_TO_KM2;
  }

  /* |log(area ratio)| — symmetric, so a donor 2x too big is as bad as 2x too small. */
  function areaRatioPenalty(targetKm2, donorKm2) {
    if (!(targetKm2 > 0) || !(donorKm2 > 0)) return null;
    return Math.abs(Math.log(targetKm2 / donorKm2));
  }

  /* Best donor by area match, with a small discount for upstream gauges: an
   * upstream gauge's flow is physically part of what passes the click, while a
   * downstream one includes tributaries that never do. Distance breaks ties.
   * Returns null when nothing is within maxRatio — a refusal is honest. */
  function selectDonor(candidates, targetKm2, opts) {
    opts = opts || {};
    var maxRatio = opts.maxRatio === undefined ? 10.0 : opts.maxRatio;
    var upstreamBonus = opts.upstreamBonus === undefined ? 0.0 : opts.upstreamBonus;
    var best = null, bestScore = Infinity;
    (candidates || []).forEach(function (c) {
      var pen = areaRatioPenalty(targetKm2, c.area_km2);
      if (pen === null || pen > Math.log(maxRatio)) return;
      var score = pen - (c.direction === "upstream" ? upstreamBonus : 0.0);
      score += 1e-4 * (c.distance_km || 0.0);
      if (score < bestScore) { best = c; bestScore = score; }
    });
    return best;
  }

  /* Q_target = Q_donor * (A_target / A_donor) ** exponent.
   * exponent 1.0 = equal specific discharge: the standard drainage-area ratio
   * method, and the only choice that needs no fitting. */
  function transferByAreaRatio(donorQCfs, donorKm2, targetKm2, exponent) {
    if (!(donorKm2 > 0) || !(targetKm2 > 0)) return null;
    var e = (exponent === undefined || exponent === null) ? 1.0 : exponent;
    var k = Math.pow(targetKm2 / donorKm2, e);
    return (donorQCfs || []).map(function (q) {
      return (q === null || q === undefined) ? null : q * k;
    });
  }

  var api = {
    EARTH_RADIUS_KM: EARTH_RADIUS_KM,
    SQMI_TO_KM2: SQMI_TO_KM2,
    haversineKm: haversineKm,
    pointToPolylineKm: pointToPolylineKm,
    geojsonAreaKm2: geojsonAreaKm2,
    cfsToMmPerDay: cfsToMmPerDay,
    mmPerDayToCfs: mmPerDayToCfs,
    sqmiToKm2: sqmiToKm2,
    areaRatioPenalty: areaRatioPenalty,
    selectDonor: selectDonor,
    transferByAreaRatio: transferByAreaRatio,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.RiverTransfer = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
