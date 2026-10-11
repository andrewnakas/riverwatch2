/* Per-member labels for the forecast panel.
 *
 * WHY THIS FILE EXISTS. The MODERN-1 member (`mblstm_modern`) is served on
 * forcing it was not trained on, and that substitution has a MEASURED cost:
 * GATE 1 priced it at a paired -0.059534 median NSE. A member whose output a
 * visitor can see has to carry that, in the panel, in the same sentence as any
 * number -- not only in a JSON key nothing reads. (That mistake was already
 * made once here: the river-transfer caveat lived in `_note` and `_note` was
 * read by nothing, which is not a disclosure.)
 *
 * Modelled on `river_click.js:skillLabel()`, including the module.exports
 * guard at the bottom -- that is what lets tests/check_member_notes_js.mjs
 * require this file and exercise the branch that quotes a number at a visitor.
 * `static_app.js` is not require-able, so logic that must be tested lives here.
 */
(function (root) {
  "use strict";

  var _label = null;

  /* Absent is fine and must stay fine: the UI then says "not yet measured"
   * rather than quoting a number nobody computed. */
  function loadLabel(fetchImpl) {
    var f = fetchImpl || (typeof fetch !== "undefined" ? fetch : null);
    if (!f) return Promise.resolve(null);
    return f("data/m1_served_skill.json")
      .then(function (r) { return r && r.ok ? r.json() : null; })
      .then(function (j) { _label = j; return j; })
      .catch(function () { _label = null; return null; });
  }

  function _setLabel(j) { _label = j; }   // tests

  function fmtDelta(d) {
    // A loss is shown as a loss. Math.abs() with the word "costs" reads
    // clearly and cannot be mistaken for a gain the way a bare "-0.0595" can.
    return Math.abs(d).toFixed(4);
  }

  /* The one-line label for a member, or "" when there is nothing to say. */
  function memberLabel(name, opts) {
    if (name !== "mblstm_modern") return "";
    var served = !!(opts && opts.served);
    if (!_label) {
      return served
        ? "research model, served on substituted forcing — discount not yet published"
        : "";
    }
    if (!served) {
      return "not run at this gauge — outside its " +
             "490-basin cohort, or its forcing was unavailable";
    }
    var fr = _label.substitution_frame || {};
    var degrade = (1 - (_label.substitution_breadth || 0)) * 100;
    var live = _label.live_measured;
    var s = "trained on " + (_label.trained_on || "a different product") +
            ", served here on " + (_label.served_on || "substituted forcing") +
            ". That substitution was measured to cost " +
            fmtDelta(_label.substitution_delta_nse) + " NSE (" +
            (fr.n_basins || "?") + "-basin " + (fr.name || "validation") +
            " sample, " + degrade.toFixed(0) + "% of basins degrade).";
    // Never quote a research level as the served skill, and never imply the
    // substitution cost and the forecast-decay cost add.
    s += live == null
      ? " Its own served skill is not yet measured."
      : " Measured on issued forecasts: " + live.headline + ".";
    s += " Display only — it carries no blend weight.";
    return s;
  }

  var api = {
    loadLabel: loadLabel,
    memberLabel: memberLabel,
    _setLabel: _setLabel,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.MemberNotes = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
