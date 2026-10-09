/* Offline check: renderRiverClick() for every result state -- no throw, no
 * "undefined"/"NaN" leaking into the page, and the "ok" panel must always name
 * the donor, the area ratio, the measured skill and the limitation.
 *
 * Run: node tests/check_river_click_render.mjs
 * Not part of the pytest gate (pages.yml runs pytest only).
 */
/* Verify renderRiverClick() produces sane HTML for every result state and
 * never throws or leaks "undefined"/"NaN" into the page. */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
const root = "/Users/nakas/Documents/RiverWatch2/riverwatch2";
const require = createRequire(import.meta.url);
globalThis.RiverTransfer = require(root + "/app/static/river_transfer.js");
globalThis.L = { geoJSON: () => ({ addTo: () => ({}) }) };
globalThis.RiverClick = require(root + "/app/static/river_click.js");

const els = {};
globalThis.document = {
  getElementById: (id) => (els[id] = els[id] || { id, style: {}, innerHTML: "", innerText: "" }),
};
globalThis.stationsById = new Map();
globalThis.map = { setView: () => {} };
globalThis.selectStation = () => {};

// Pull just the two functions out of static_app.js.
const src = readFileSync(root + "/app/static/static_app.js", "utf8");
const start = src.indexOf("function renderRiverClick(res)");
const end = src.indexOf("loadStations().then");
if (start < 0 || end < 0) { console.log("FAIL: could not locate renderRiverClick"); process.exit(1); }
// Indirect eval runs in global sloppy scope, so the declarations become globals.
(0, eval)(src.slice(start, end) + "\n;globalThis.renderRiverClick = renderRiverClick;");

const cases = [
  ["loading",        { state: "loading", lat: 44.6, lon: -67.9 }],
  ["no_catchment",   { state: "no_catchment", lat: 40, lon: -130 }],
  ["far_from_river", { state: "far_from_river", lat: 25.5, lon: -80.6, distance_km: 25.394, area_km2: 364.2 }],
  ["no_area",        { state: "no_area", lat: 44.6, lon: -67.9, comid: 123 }],
  ["no_donor",       { state: "no_donor", lat: 45.3, lon: -69.4, comid: 1721767, area_km2: 117.5, n_candidates: 0 }],
  ["error",          { state: "error", lat: 44.6, lon: -67.9, message: "HTTP 503" }],
  ["ok",             { state: "ok", lat: 44.608, lon: -67.935, comid: 2677104, distance_km: 0.014,
                       area_km2: 838.8, ratio: 1.427, issued_at: "2026-10-09T06:00:00Z", n_candidates: 2,
                       donor: { id: "01022500", name: "Narraguagus River at Cherryfield, Maine",
                                direction: "upstream", area_km2: 587.9 },
                       rows: Array.from({ length: 14 }, (_, i) => ({
                         date: `2026-10-${String(10 + i).padStart(2, "0")}`,
                         q_cfs: 485.8 - i * 5, mm_day: 1.417 - i * 0.01, donor_q_cfs: 340.5 - i * 3 })) }],
];

let fails = 0;
for (const [name, res] of cases) {
  els["river-click"] = { style: {}, innerHTML: "" };
  try { globalThis.renderRiverClick(res); } catch (e) { console.log(`FAIL ${name}: threw ${e.message}`); fails++; continue; }
  const html = els["river-click"].innerHTML;
  const bad = [];
  if (/undefined/.test(html)) bad.push("contains 'undefined'");
  if (/NaN/.test(html)) bad.push("contains 'NaN'");
  if (!/<h2>/.test(html)) bad.push("no heading");
  if (html.length < 60) bad.push("suspiciously short");
  if (name === "ok") {
    if (!/mm\/day/.test(html)) bad.push("no mm/day headline");
    if (!/01022500/.test(html)) bad.push("donor not named");
    if (!/1\.43×|1\.43x/.test(html)) bad.push("area ratio not shown");
    if (!/not yet measured|median NSE/.test(html)) bad.push("no skill statement");
    if (!/adds no\s+new hydrology|no\s+new hydrology/.test(html)) bad.push("no limitation caveat");
    if ((html.match(/<tr>/g) || []).length < 14) bad.push("hydrograph table incomplete");
  }
  if (name === "far_from_river" && !/25\.4 km/.test(html)) bad.push("distance not reported");
  const heading = (html.match(/<h2>(.*?)<\/h2>/) || [, "?"])[1];
  if (bad.length) { console.log(`FAIL ${name}: ${bad.join("; ")}`); fails++; }
  else console.log(`ok   ${name.padEnd(15)} h2="${heading}" (${html.length} chars)`);
}
console.log(fails ? `${fails} FAILED` : "all render states clean: no throw, no undefined/NaN");
process.exit(fails ? 1 : 0);
