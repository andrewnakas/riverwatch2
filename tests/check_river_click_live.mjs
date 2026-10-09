/* LIVE check: drives app/static/river_click.js against the real USGS NLDI API
 * with stubbed Leaflet and local forecast files. Exercises the decision logic
 * the arithmetic tests cannot reach: snapping, the distance gate, delineation,
 * donor intersection with stations.json, and the area-ratio transfer.
 *
 * Run: node tests/check_river_click_live.mjs <dist-dir>
 * NEEDS NETWORK -- deliberately not in the pytest gate.
 */
/* Drive river_click.js under node against the LIVE NLDI API, with minimal
 * stubs for Leaflet and the forecast fetch. Verifies the decision logic that
 * the arithmetic tests cannot reach: snapping, the distance gate, delineation,
 * donor intersection with stations.json, and the transfer. */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";

// Derive the repo root from this file, never hardcode it: an absolute path
// makes the check silently validate ANOTHER checkout (it read the research
// branch while verifying a port worktree, and passed).
const root = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const dist = process.argv[2];
const require = createRequire(import.meta.url);

// --- stubs ------------------------------------------------------------
globalThis.L = { geoJSON: () => ({ addTo: () => ({}) }) };
const drawn = [];
globalThis.L.geoJSON = (g) => { drawn.push(g && g.type); return { addTo: () => ({}) }; };

const realFetch = globalThis.fetch;
globalThis.fetch = async (url) => {
  if (String(url).startsWith("http")) return realFetch(url);        // NLDI: live
  const p = join(dist, String(url));                                 // local files
  try { return { ok: true, json: async () => JSON.parse(readFileSync(p, "utf8")) }; }
  catch { return { ok: false, status: 404, json: async () => ({}) }; }
};

globalThis.RiverTransfer = require(join(root, "app/static/river_transfer.js"));
const RC = require(join(root, "app/static/river_click.js"));

const stations = JSON.parse(readFileSync(join(dist, "stations.json"), "utf8")).stations;
const stationsById = new Map(stations.map(s => [s.id, s]));

let captured = null;
RC.init({ map: null, stationsById, onRender: (r) => { if (r.state !== "loading") captured = r; } });

async function probe(label, lat, lon) {
  captured = null;
  drawn.length = 0;
  await RC.handleMapClick(lat, lon);
  const r = captured || {};
  const o = { label, state: r.state };
  if (r.distance_km != null) o.dist_km = +r.distance_km.toFixed(3);
  if (r.area_km2 != null) o.basin_km2 = +r.area_km2.toFixed(1);
  if (r.comid) o.comid = r.comid;
  if (r.n_candidates != null) o.candidates = r.n_candidates;
  if (r.donor) {
    o.donor = r.donor.id; o.donor_dir = r.donor.direction;
    o.donor_km2 = +r.donor.area_km2.toFixed(1); o.ratio = +r.ratio.toFixed(3);
    o.day1_cfs = +r.rows[0].q_cfs.toFixed(1);
    o.day1_mm = +r.rows[0].mm_day.toFixed(3);
    o.donor_day1_cfs = +r.rows[0].donor_q_cfs.toFixed(1);
    o.leads = r.rows.length;
    // The defining invariant: specific discharge is preserved by the transfer.
    const mmDonor = RiverTransfer.cfsToMmPerDay(r.rows[0].donor_q_cfs, r.donor.area_km2);
    o.specific_discharge_preserved = Math.abs(mmDonor - r.rows[0].mm_day) < 1e-9;
  }
  o.drew = drawn.filter(Boolean);
  return o;
}

const results = [];
results.push(await probe("on the Narraguagus River, ME", 44.608, -67.935));
results.push(await probe("Sandy River, OR",               45.400, -122.140));
results.push(await probe("Everglades, 25 km off-channel", 25.500, -80.600));
results.push(await probe("Lake Michigan",                 43.500, -87.300));
results.push(await probe("open Pacific",                  40.000, -130.000));
console.log(JSON.stringify(results, null, 1));
