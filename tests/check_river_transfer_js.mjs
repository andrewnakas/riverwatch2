/* Cross-check app/static/river_transfer.js against the fixture emitted by
 * tests/test_river_transfer.py. Run: node tests/check_river_transfer_js.mjs
 *
 * This is NOT part of the pytest gate (pages.yml runs pytest only). It is the
 * stated mechanism by which the JS mirror is kept honest.
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const require = createRequire(import.meta.url);
const RT = require(join(root, "app", "static", "river_transfer.js"));
const fx = JSON.parse(readFileSync(join(here, "river_transfer_fixture.json"), "utf8"));

let fails = 0, checks = 0;
const REL = 1e-6;

function close(got, want) {
  if (want === null || want === undefined) return got === null || got === undefined;
  if (got === null || got === undefined) return false;
  if (want === 0) return Math.abs(got) < 1e-12;
  return Math.abs(got - want) / Math.abs(want) <= REL;
}
function check(name, got, want) {
  checks++;
  if (!close(got, want)) {
    fails++;
    console.log(`FAIL ${name}: got ${got}, want ${want}`);
  }
}

for (const c of fx.haversine_km) check("haversineKm", RT.haversineKm(...c.args), c.want);
for (const c of fx.point_to_polyline_km)
  check("pointToPolylineKm", RT.pointToPolylineKm(c.lat, c.lon, c.coords), c.want);
for (const c of fx.geojson_area_km2) check("geojsonAreaKm2", RT.geojsonAreaKm2(c.geometry), c.want);
for (const c of fx.cfs_to_mm_per_day) check("cfsToMmPerDay", RT.cfsToMmPerDay(...c.args), c.want);
for (const c of fx.sqmi_to_km2) check("sqmiToKm2", RT.sqmiToKm2(...c.args), c.want);
for (const c of fx.transfer_by_area_ratio) {
  const got = RT.transferByAreaRatio(c.series, c.donor_km2, c.target_km2);
  c.want.forEach((w, i) => check(`transferByAreaRatio[${i}]`, got[i], w));
}
for (const c of fx.select_donor) {
  const got = RT.selectDonor(c.candidates, c.target_km2);
  checks++;
  const gotId = got ? got.id : null;
  if (gotId !== c.want) { fails++; console.log(`FAIL selectDonor: got ${gotId}, want ${c.want}`); }
}

console.log(`${checks - fails}/${checks} JS cross-checks passed (rel tol ${REL})`);
process.exit(fails ? 1 : 0);
