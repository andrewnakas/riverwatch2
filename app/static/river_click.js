/* Click anywhere near a river and get the runoff.
 *
 * Clicking a marker gives that gauge's own ensemble forecast. Clicking the map
 * instead asks USGS NLDI to snap the point to an NHD flowline, delineates the
 * upstream basin, finds the gauges reachable along the network, and transfers
 * the best-matched gauge's forecast by drainage-area ratio. All of it runs in
 * the browser against public APIs, so it works on the static Pages deploy with
 * no backend — the same pattern the page already uses to pull live USGS
 * instantaneous values.
 *
 * HONESTY RULES BUILT IN, not bolted on:
 *  - NLDI's comid/position returns the CONTAINING CATCHMENT's flowline even for
 *    a point nowhere near water (a click on a desert flat still returns a
 *    reach). So we measure the distance from the click to that reach ourselves
 *    and say how far away it is instead of pretending the click was on a river.
 *  - comid/{id}/basin has NO area property, so area is computed from the
 *    polygon (RiverTransfer.geojsonAreaKm2).
 *  - Navigation results are unordered and include non-discharge sites (15-digit
 *    measurement points), so candidates are intersected with stations.json —
 *    which is also exactly the set that has a precomputed forecast and a
 *    drainage area.
 *  - The headline number is mm/day (runoff depth). That is the quantity that
 *    actually transfers between catchments of different size, and it is what
 *    "the runoff at this point" honestly means.
 *  - The donor, the area ratio and the measured transfer skill are always shown.
 *
 * Arithmetic lives in river_transfer.js, mirrored and tested in
 * analysis/river_transfer.py.
 */
(function (root) {
  "use strict";

  var NLDI = "https://api.water.usgs.gov/nldi/linked-data/comid";
  // Beyond this the click is reported as "not near a mapped channel" rather
  // than silently answered for a catchment the user did not click.
  var NEAR_RIVER_KM = 2.0;
  var NAV_DISTANCE_KM = 100;        // how far along the network to look for gauges
  var CACHE_TTL_MS = 10 * 60 * 1000;

  var RT = root.RiverTransfer;
  var _cache = new Map();           // "lat,lon" (3dp) -> {at, result}
  var _layers = null;               // the drawn reach + basin
  var _skill = null;                // data/river_transfer_skill.json, if present
  var _skillTried = false;
  var _cfg = { map: null, stationsById: null, onRender: null };

  function init(cfg) {
    _cfg.map = cfg.map;
    _cfg.stationsById = cfg.stationsById;
    _cfg.onRender = cfg.onRender;
    if (!_cfg.map) return;
    _cfg.map.on("click", function (e) {
      handleMapClick(e.latlng.lat, e.latlng.lng);
    });
  }

  async function getJSON(url) {
    var r = await fetch(url);
    if (!r.ok) {
      var err = new Error("HTTP " + r.status);
      err.status = r.status;
      throw err;
    }
    return r.json();
  }

  /* Measured skill for the transfer, if the offline study has been committed.
   * Absent is fine and must stay fine: the UI then says "unmeasured" rather
   * than quoting a number nobody computed. */
  async function loadSkill() {
    if (_skillTried) return _skill;
    _skillTried = true;
    try {
      var r = await fetch("data/river_transfer_skill.json");
      if (r.ok) _skill = await r.json();
    } catch (_) { _skill = null; }
    return _skill;
  }

  function skillLabel(ratio) {
    if (!_skill || !_skill.bins) return "transfer skill: not yet measured";
    var lr = Math.abs(Math.log(ratio));
    var best = null;
    (_skill.bins || []).forEach(function (b) {
      if (lr <= b.max_abs_log_ratio && (best === null || b.max_abs_log_ratio < best.max_abs_log_ratio)) best = b;
    });
    if (!best) return "transfer skill: outside the measured range";
    // Use the bin's own label ("1.25x") rather than re-deriving it from the log
    // cut, which rounds 1.25 to "1.3".
    var band = best.label || (Math.exp(best.max_abs_log_ratio).toFixed(1) + "x");
    // The measurement is observed-to-observed: a donor gauge's OBSERVED flow
    // rescaled onto the target. What the panel shows beside it is a FORECAST,
    // so the donor's own forecast error is additional and unmeasured. Say so
    // here -- data/river_transfer_skill.json carries the same caveat under
    // `_note`, but nothing reads `_note`, and a caveat in an unread key is not
    // a disclosure.
    return "measured median NSE " + best.median_nse.toFixed(3) +
           " for area ratios within " + band +
           " (n=" + best.n_pairs + " gauge pairs)" +
           " \u2014 gauge-to-gauge transfer only, before the donor's own forecast error";
  }

  function clearLayers() {
    if (!_layers || !_cfg.map) return;
    _layers.forEach(function (l) { _cfg.map.removeLayer(l); });
    _layers = null;
  }

  function drawReachAndBasin(reachGeom, basinGeom) {
    clearLayers();
    _layers = [];
    if (basinGeom) {
      _layers.push(L.geoJSON(basinGeom, {
        style: { color: "#4cc8ff", weight: 1, fillColor: "#4cc8ff", fillOpacity: 0.12 },
        interactive: false,
      }).addTo(_cfg.map));
    }
    if (reachGeom) {
      _layers.push(L.geoJSON(reachGeom, {
        style: { color: "#ff8a4c", weight: 4, opacity: 0.9 },
        interactive: false,
      }).addTo(_cfg.map));
    }
  }

  function gaugeCandidates(navFeatures, direction, clickLat, clickLon) {
    var out = [];
    (navFeatures || []).forEach(function (f) {
      var ident = (f.properties || {}).identifier || "";
      var id = ident.replace(/^USGS-/, "");
      var st = _cfg.stationsById ? _cfg.stationsById.get(id) : null;
      if (!st) return;                       // no forecast / no area -> unusable
      var areaKm2 = RT.sqmiToKm2(st.drain_area_sqmi);
      if (!(areaKm2 > 0)) return;
      var c = f.geometry && f.geometry.coordinates;
      var dist = (c && c.length >= 2)
        ? RT.haversineKm(clickLat, clickLon, c[1], c[0]) : null;
      out.push({
        id: id, name: st.name, area_km2: areaKm2,
        direction: direction, distance_km: dist, station: st,
      });
    });
    return out;
  }

  async function handleMapClick(lat, lon) {
    var key = lat.toFixed(3) + "," + lon.toFixed(3);
    var hit = _cache.get(key);
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) {
      render(hit.result);
      return;
    }
    render({ state: "loading", lat: lat, lon: lon });
    var result;
    try {
      result = await resolveClick(lat, lon);
    } catch (exc) {
      result = { state: "error", lat: lat, lon: lon, message: exc.message };
    }
    _cache.set(key, { at: Date.now(), result: result });
    render(result);
  }

  async function resolveClick(lat, lon) {
    await loadSkill();
    var base = { lat: lat, lon: lon };

    // 1. Snap to the containing catchment's flowline.
    var pos;
    try {
      pos = await getJSON(NLDI + "/position?coords=POINT(" + lon.toFixed(6) + "%20" + lat.toFixed(6) + ")");
    } catch (exc) {
      if (exc.status === 404) {
        return Object.assign(base, { state: "no_catchment" });
      }
      throw exc;
    }
    var feat = (pos.features || [])[0];
    if (!feat) return Object.assign(base, { state: "no_catchment" });
    var comid = (feat.properties || {}).comid;
    var reachGeom = feat.geometry;
    var distKm = RT.pointToPolylineKm(lat, lon, (reachGeom || {}).coordinates || []);

    // 2. The upstream basin, and its area (NLDI gives no area).
    var basinGeom = null, areaKm2 = null;
    try {
      var basin = await getJSON(NLDI + "/" + comid + "/basin");
      var bf = (basin.features || [])[0];
      if (bf) {
        basinGeom = bf.geometry;
        areaKm2 = RT.geojsonAreaKm2(basinGeom);
      }
    } catch (_) { /* area stays null; handled below */ }

    drawReachAndBasin(reachGeom, basinGeom);

    if (distKm !== null && distKm > NEAR_RIVER_KM) {
      return Object.assign(base, {
        state: "far_from_river", comid: comid, distance_km: distKm,
        area_km2: areaKm2,
      });
    }
    if (!(areaKm2 > 0)) {
      return Object.assign(base, { state: "no_area", comid: comid, distance_km: distKm });
    }

    // 3. Gauges reachable up- and downstream along the network.
    var cands = [];
    for (var i = 0; i < 2; i++) {
      var dir = i === 0 ? "UM" : "DM";
      try {
        var nav = await getJSON(NLDI + "/" + comid + "/navigation/" + dir +
                                "/nwissite?distance=" + NAV_DISTANCE_KM);
        cands = cands.concat(gaugeCandidates(nav.features, i === 0 ? "upstream" : "downstream", lat, lon));
      } catch (_) { /* one direction may legitimately have none */ }
    }
    var donor = RT.selectDonor(cands, areaKm2);
    if (!donor) {
      return Object.assign(base, {
        state: "no_donor", comid: comid, distance_km: distKm,
        area_km2: areaKm2, n_candidates: cands.length,
      });
    }

    // 4. The donor's precomputed forecast, transferred by area ratio.
    var fc = await getJSON("forecasts/" + donor.id + ".json");
    var blend = (fc.blend || []).slice();
    var qs = blend.map(function (r) { return r.q_cfs; });
    var transferred = RT.transferByAreaRatio(qs, donor.area_km2, areaKm2);
    var rows = blend.map(function (r, i) {
      return {
        date: r.date,
        q_cfs: transferred[i],
        mm_day: RT.cfsToMmPerDay(transferred[i], areaKm2),
        donor_q_cfs: r.q_cfs,
      };
    });
    return Object.assign(base, {
      state: "ok", comid: comid, distance_km: distKm, area_km2: areaKm2,
      donor: donor, ratio: areaKm2 / donor.area_km2, rows: rows,
      issued_at: fc.issued_at, n_candidates: cands.length,
    });
  }

  function fmt(x, d) {
    if (x === null || x === undefined || !isFinite(x)) return "—";
    return Number(x).toFixed(d === undefined ? 1 : d);
  }

  function render(res) {
    if (_cfg.onRender) { _cfg.onRender(res); return; }
    console.log("[river_click]", res);
  }

  var api = {
    init: init,
    handleMapClick: handleMapClick,
    clearLayers: clearLayers,
    NEAR_RIVER_KM: NEAR_RIVER_KM,
    skillLabel: skillLabel,
    fmt: fmt,
    _gaugeCandidates: gaugeCandidates,
    // Exposed for tests/check_river_click_render.mjs: without it the harness
    // cannot populate _skill, skillLabel() only ever returns "not yet
    // measured", and the measured branch -- the one that quotes a number at a
    // visitor -- goes unexercised.
    _loadSkill: loadSkill,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.RiverClick = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
