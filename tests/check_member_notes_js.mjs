/* Node check for app/static/member_notes.js -- the public copy that quotes a
 * measured number at a visitor.
 *
 * NOT part of the pytest gate by origin, but the pages.yml `test` job now runs
 * every tests/check_*.mjs, so this does gate the deploy.
 *
 * It exercises BOTH forcing configurations the label builder can emit, because
 * which one is correct depends on what is actually published:
 *   GATE 1 — the member served Open-Meteo point forcing throughout (-0.059534)
 *   GATE 2 — the areal provider landed, so most of the encoder window is the
 *            trained-on product and only the newest days are substituted
 *            (-0.004018 paired, -0.036142 on the level)
 * Quoting GATE 1 once GATE 2 applies overstates the discount, which is still a
 * false impression -- so both branches are pinned.
 */
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const MN = require(path.join(here, "..", "app", "static", "member_notes.js"));
const root = path.join(here, "..");

let fails = 0;
function check(name, cond, extra) {
  if (cond) console.log(`ok   ${name}`);
  else { console.log(`FAIL ${name}${extra ? " -- " + extra : ""}`); fails++; }
}
const g1 = JSON.parse(fs.readFileSync(path.join(root, "benchmarks/m1_gate1_verdict.json"), "utf8"));
const g2 = JSON.parse(fs.readFileSync(path.join(root, "benchmarks/m1_gate2_verdict.json"), "utf8"));

// --- 1. no sidecar: nothing numeric may be quoted.
MN._setLabel(null);
const cold = MN.memberLabel("mblstm_modern", { served: true });
check("cold branch says something", cold.length > 0);
check("cold branch quotes NO number", !/\d\.\d/.test(cold), cold);
check("cold branch still discloses substitution", /substituted forcing/.test(cold), cold);

// --- 2. other members get no label at all.
check("other members unlabelled", MN.memberLabel("chronos_bolt", { served: true }) === "");

// --- 3. GATE 1 configuration (point forcing throughout).
MN._setLabel({
  schema_version: "m1-label-2", trained_on: "NLDAS-2 basin-average (areal) forcing",
  served_on: "Open-Meteo point forcing at the gauge coordinate",
  governing_gate: "GATE 1",
  encoder_substitution_delta: g1.paired_delta_point_minus_areal,
  encoder_substitution_ci95: g1.ci95, encoder_substitution_breadth: g1.breadth,
  encoder_substitution_level: null,
  encoder_substitution_frame: { name: g1.frame, n_basins: g1.basins },
  live_measured: null,
});
const p = MN.memberLabel("mblstm_modern", { served: true });
check("GATE 1: names both forcings", /NLDAS-2/.test(p) && /Open-Meteo/.test(p), p);
check("GATE 1: quotes 0.0595", p.includes("0.0595"), p);
check("GATE 1: states the frame", /150-basin/.test(p) && /val1/.test(p), p);
check("GATE 1: no level clause (none measured)", !/cohort level/.test(p), p);

// --- 4. GATE 2 configuration (areal provider landed).
MN._setLabel({
  schema_version: "m1-label-2", trained_on: "NLDAS-2 basin-average (areal) forcing",
  served_on: "NLDAS-2 basin-average (areal) forcing -- the product it was trained on -- for 98% of the encoder window, with the newest days Open-Meteo point-filled because NLDAS-2 publishes at today-5",
  governing_gate: "GATE 2",
  encoder_substitution_delta: g2.paired_delta_spliced_minus_areal,
  encoder_substitution_ci95: g2.ci95, encoder_substitution_breadth: g2.breadth,
  encoder_substitution_level: g2.difference_of_medians,
  encoder_substitution_frame: { name: g2.frame, n_basins: g2.basins },
  live_measured: null,
});
const a = MN.memberLabel("mblstm_modern", { served: true });
check("GATE 2: quotes the paired 0.0040", a.includes("0.0040"), a);
check("GATE 2: ALSO quotes the level 0.0361", a.includes("0.0361"), a);
check("GATE 2: says it is served on the trained-on product", /trained on/.test(a), a);
check("GATE 2: does NOT quote GATE 1's 0.0595", !a.includes("0.0595"), a);

// --- 4b. the extrapolation warning must reach the rendered copy.
MN._setLabel({
  schema_version: "m1-label-2", trained_on: "NLDAS-2 basin-average (areal) forcing",
  served_on: "mostly the trained-on product",
  governing_gate: "GATE 2",
  encoder_substitution_delta: g2.paired_delta_spliced_minus_areal,
  encoder_substitution_level: g2.difference_of_medians,
  encoder_substitution_frame: { name: g2.frame, n_basins: g2.basins },
  extrapolation_warning: "The newest basin-average day is 11 days old, but the gate that priced this configuration substituted only 4. Its delta is therefore an extrapolation here, and the true cost is likely larger.",
  live_measured: null,
});
const warn = MN.memberLabel("mblstm_modern", { served: true });
check("extrapolation warning is RENDERED, not just stored",
  /extrapolation/.test(warn), warn);
check("warning appears beside the number it corrects",
  warn.indexOf("0.0040") < warn.indexOf("extrapolation"), warn);

// --- 5. the forbidden figures, unreachable from either configuration.
for (const [tag, s] of [["GATE 1", p], ["GATE 2", a]]) {
  check(`${tag}: never the perfect-forcing record`, !s.includes("0.9058"), s);
  check(`${tag}: never a research level as served skill`, !s.includes("0.8086") && !s.includes("0.8936"), s);
  check(`${tag}: never the summed discount`, !s.includes("0.1076") && !s.includes("0.1035"), s);
  check(`${tag}: says served skill not yet measured`, /not yet measured/.test(s), s);
  check(`${tag}: says it carries no blend weight`, /no blend weight/.test(s), s);
}

// --- 6. not-served branch.
const nr = MN.memberLabel("mblstm_modern", { served: false });
// Matches "served domain", not "cohort": GATE 3 widened the domain to two
// tiers, so the public string names 620 and no longer says "490-basin cohort".
// The NUMBER is asserted too, because that is the part that silently drifts.
check("not-run branch explains why", /served domain/.test(nr), nr);
check("not-run branch names the current domain size", /620/.test(nr), nr);
check("not-run branch does not name the stale 490", !/490-basin/.test(nr), nr);

// --- 7. the tier a gauge is in must reach the label.
// GATE 3: "The UI must mark these basins as a distinct tier." Before this, a
// visitor on one of the 130 basins the model was never trained on saw exactly
// the label a trained-on basin shows.
const t2 = MN.memberLabel("mblstm_modern",
                          { served: true, tier: "out_of_cohort_gate3" });
check("tier 2 is marked as outside the training basins",
      /OUTSIDE the basins this model was trained on/.test(t2), t2);
check("tier 2 quotes its OWN median", /0\.854214/.test(t2), t2);
check("tier 2 states the cohort figure it must not be confused with",
      /0\.876335/.test(t2), t2);
check("tier 2 says which figure applies",
      /the lower figure is the one that applies here/.test(t2), t2);

const t1 = MN.memberLabel("mblstm_modern", { served: true, tier: "in_cohort" });
check("tier 1 carries no out-of-cohort marker",
      !/OUTSIDE the basins/.test(t1), t1);
check("tier 1 does not quote the out-of-cohort median",
      !/0\.854214/.test(t1), t1);

const tNone = MN.memberLabel("mblstm_modern", { served: true });
check("an absent tier behaves exactly like tier 1",
      tNone === t1, tNone + " VS " + t1);
check("not-run branch quotes no skill number", !/\d\.\d{3}/.test(nr), nr);

// --- 7. the SHIPPED sidecar must render through the same path.
const shipped = path.join(root, "data/m1_served_skill.json");
if (fs.existsSync(shipped)) {
  MN._setLabel(JSON.parse(fs.readFileSync(shipped, "utf8")));
  const live = MN.memberLabel("mblstm_modern", { served: true });
  check("shipped sidecar renders a number", /\d\.\d{4}/.test(live), live);
  check("shipped sidecar quotes no forbidden figure",
    !live.includes("0.9058") && !live.includes("0.1076"), live);
}

console.log(fails ? `\n${fails} check(s) FAILED` : "\nall member_notes checks passed");
process.exit(fails ? 1 : 0);
