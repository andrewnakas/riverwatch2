/* Node check for app/static/member_notes.js.
 *
 * NOT part of the pytest gate (pages.yml runs pytest only), the same standing
 * as tests/check_river_transfer_js.mjs. It exists because the measured-branch
 * string -- the one that quotes a number at a visitor -- is public copy about
 * a model's skill, and it must not be able to drift silently.
 */
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const MN = require(path.join(here, "..", "app", "static", "member_notes.js"));
const label = JSON.parse(
  fs.readFileSync(path.join(here, "..", "data", "m1_served_skill.json"), "utf8"));

let fails = 0;
function check(name, cond, extra) {
  if (cond) { console.log(`ok   ${name}`); }
  else { console.log(`FAIL ${name}${extra ? " -- " + extra : ""}`); fails++; }
}

// 1. No sidecar loaded: nothing numeric may be quoted.
const cold = MN.memberLabel("mblstm_modern", { served: true });
check("cold branch says something", cold.length > 0);
check("cold branch quotes NO number", !/\d\.\d/.test(cold), cold);
check("cold branch still discloses substitution", /substituted forcing/.test(cold), cold);

// 2. Other members get no label at all.
check("other members unlabelled", MN.memberLabel("chronos_bolt", { served: true }) === "");

// 3. Measured branch, with the real shipped sidecar.
MN._setLabel(label);
const warm = MN.memberLabel("mblstm_modern", { served: true });
check("names what it trained on", /NLDAS-2/.test(warm), warm);
check("names what it is served on", /Open-Meteo/.test(warm), warm);
check("quotes the measured substitution cost", warm.includes("0.0595"), warm);
check("states the frame's basin count", /150-basin/.test(warm), warm);
check("states the frame is validation", /val1/.test(warm), warm);
check("states how many basins degrade", /91% of basins degrade/.test(warm), warm);
check("says served skill not yet measured", /not yet measured/.test(warm), warm);
check("says it carries no blend weight", /no blend weight/.test(warm), warm);

// 4. The forbidden claims must be unreachable from this copy.
check("never quotes the perfect-forcing record", !warm.includes("0.9058"), warm);
check("never quotes a research level as served skill", !warm.includes("0.8086"), warm);
check("never quotes the summed discount", !warm.includes("0.1076"), warm);

// 5. Not-served branch.
const nr = MN.memberLabel("mblstm_modern", { served: false });
check("not-run branch explains why", /cohort/.test(nr), nr);
check("not-run branch quotes no skill number", !/\d\.\d{3}/.test(nr), nr);

console.log(fails ? `\n${fails} check(s) FAILED` : "\nall member_notes checks passed");
process.exit(fails ? 1 : 0);
