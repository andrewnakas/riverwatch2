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
      // 620, not 490: GATE 3 widened the served domain to two tiers. A public
      // string naming the old number tells a visitor the wrong reason their
      // gauge has no member.
      return "not run at this gauge — outside its " +
             "620-basin served domain, or its forcing was unavailable";
    }
    // ⭐ WHICH TIER THIS GAUGE IS IN. GATE 3's decision table requires that
    // out-of-cohort basins be marked as a distinct tier and quoted their OWN
    // median, "never the cohort one". Until this was passed through, a visitor
    // on one of the 130 basins the model has never been trained on saw exactly
    // the label a trained-on basin shows -- the caveat was published, archived
    // and rendered on /benchmarks, and absent from the one place that visitor
    // was actually looking.
    var tier = (opts && opts.tier) || "";
    var tierNote = "";
    if (tier === "out_of_cohort_gate3") {
      tierNote = "⚠️ This gauge is OUTSIDE the basins this model was trained " +
                 "on. Those 130 measure a median NSE of 0.854214 " +
                 "(CI [0.838676, 0.895274]) against 0.876335 for the 490 it " +
                 "was trained on, on the same window — the lower figure is " +
                 "the one that applies here. ";
    }
    // Schema v2 renamed these and added the cohort LEVEL; v1 keys are still
    // read so an older sidecar cannot blank the label.
    var delta = _label.encoder_substitution_delta;
    if (delta == null) delta = _label.substitution_delta_nse;
    var fr = _label.encoder_substitution_frame || _label.substitution_frame || {};
    var level = _label.encoder_substitution_level;
    var live = _label.live_measured;
    if (delta == null) {
      return "research model, served on substituted forcing \u2014 discount not yet published";
    }
    var s = "trained on " + (_label.trained_on || "a different product") +
            ", served here on " + (_label.served_on || "substituted forcing") +
            ". The substituted part was measured to cost " + fmtDelta(delta) +
            " NSE (" + (fr.n_basins || "?") + "-basin " +
            (fr.name || "validation") + " sample";
    // A level shift must travel WITH the paired figure: areal forcing recovers
    // 93.3% of the point-forcing loss paired but only 57.5% on the level, so
    // the paired number alone overstates the fix.
    if (level != null) s += "; " + fmtDelta(level) + " on the cohort level";
    s += ").";
    // Never quote a research level as the served skill, and never imply the
    // substitution cost and the forecast-decay cost add.
    s += live == null
      ? " Its own served skill is not yet measured."
      : " Measured on issued forecasts: " + live.headline + ".";
    // An extrapolation warning is a correction to the number just quoted, so
    // it must appear beside it. A caveat living only in an unread JSON key is
    // not a disclosure -- that mistake was already made once here.
    if (_label.extrapolation_warning) s += " \u26a0 " + _label.extrapolation_warning;
    if (tierNote) s = tierNote + s;
    s += " Display only \u2014 it carries no blend weight.";
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
