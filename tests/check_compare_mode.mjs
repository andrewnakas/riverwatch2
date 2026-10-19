/* The three-way comparison a visitor actually asks for: physics baseline,
 * the ensemble that is serving, and the new research model.
 *
 * Every member already shipped in the payload and NONE was visible without a
 * click -- the chart drew history + blend only -- and when they were shown,
 * mblstm_modern had no entry in memColors and drew as an anonymous grey line
 * among ten. The model this whole track is about was the least identifiable
 * thing on the page.
 *
 * What must hold, because this renders numbers at a visitor:
 *  - exactly the three named series, each with its own colour;
 *  - MODERN-1's interval drawn, since it is the only member with a quantile
 *    head and the only one that can show one;
 *  - and the legend must state, for each curve, whether its error is MEASURED
 *    on this river -- MODERN-1's is not, and it carries zero blend weight, so
 *    a drawn line would otherwise imply a standing it has not earned.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url)).replace(/\/$/, "");
const src = readFileSync(root + "/app/static/static_app.js", "utf8");
const html = readFileSync(root + "/app/templates/index.html", "utf8");

let failed = 0;
function check(name, cond, got) {
  if (cond) { console.log("ok   " + name); return; }
  failed++;
  console.log("FAIL " + name + (got === undefined ? "" : " -- " + got));
}

// --- the three series, named and coloured.
check("compare mode defines exactly three series",
      (src.match(/key: "nwm"|key: "__blend__"|key: "mblstm_modern"/g) || []).length === 3);
check("MODERN-1 is named, not anonymous", /MODERN-1 \(research\)/.test(src));
check("NWM is named as the physics baseline", /NWM \(physics\)/.test(src));
check("the serving ensemble is named", /Serving ensemble/.test(src));

// --- mblstm_modern must no longer fall through to grey in the 10-member view.
const mc = src.slice(src.indexOf("const memColors"), src.indexOf("const memColors") + 600);
check("mblstm_modern has its own colour in the member view",
      /mblstm_modern: "#/.test(mc), mc.slice(0, 120));
check("nwm_residual has its own colour too", /nwm_residual: "#/.test(mc));

// --- only MODERN-1 gets a band, because only it has a quantile head.
check("MODERN-1's interval is attached in compare mode",
      /cs\.key === "mblstm_modern"[\s\S]{0,400}band_lo: r\.q_lo/.test(src));

// --- the two modes are mutually exclusive, so a visitor never sees 13 lines.
check("compare mode clears the members view", /if \(_fcst\.compare\) _fcst\.showMembers = false;/.test(src));
check("members view clears compare mode", /if \(_fcst\.showMembers\) _fcst\.compare = false;/.test(src));

// --- THE HONESTY REQUIREMENT.
const leg = src.slice(src.indexOf("function renderCompareLegend"),
                      src.indexOf("function renderCompareLegend") + 2200);
check("legend reports a measured MAE when there is one", /rolling MAE/.test(leg));
check("legend says plainly when there is NOT one",
      /no measured error on this river yet/.test(leg), leg.slice(0, 80));
check("legend states MODERN-1 carries zero weight", /zero weight/.test(leg));
check("legend says its benchmark numbers are not from real-time forcing",
      /forcing it does not get in real time/.test(leg));
// Matched within a single source literal: the sentence is built by
// concatenation, so a regex spanning the "+" would fail on formatting rather
// than on meaning.
check("legend states the other two share one basis",
      /basis for both/.test(leg) && /re-running them on past windows/.test(leg));

// --- the control exists and is reachable.
check("the compare button is in the DOM", /id="toggle-compare"/.test(html));
check("the legend slot is in the DOM", /id="compare-legend"/.test(html));
check("the button is wired", /getElementById\("toggle-compare"\)/.test(src));
check("the chart receives quantiles and mae",
      /quantiles: payload\.member_quantiles, mae: payload\.rolling_mae/.test(src));

if (failed) { console.log("\n" + failed + " check(s) FAILED"); process.exit(1); }
console.log("\nall compare-mode checks passed");
