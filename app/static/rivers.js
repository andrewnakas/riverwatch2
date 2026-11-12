"use strict";
/* Everything a river-watcher needs around the forecast itself:
 *
 *   Conditions  current flow vs the gauge's own record (dist/current.json,
 *               written by scripts/current_conditions.py at merge time)
 *   NWS         flood stages, rating curve, gage height and the official RFC
 *               forecast, straight from NOAA's National Water Prediction
 *               Service (CORS-open, looked up by USGS site number)
 *   Favs        starred gauges, per browser (localStorage)
 *   Search      by river name, gauge id, state or town
 *   DeepLink    #site=<id> opens a gauge; selecting one updates the hash
 *
 * Loaded before app.js, which calls into these.
 */

const Conditions = (() => {
  // WaterWatch-style palette, ordered by class index (0 = unknown).
  const COLORS = ["#6b7390", "#8b1a1a", "#ff9f43", "#2ecc71", "#4cc8ff", "#3557ff"];
  const NAMES = ["no data", "record low", "below normal", "normal", "above normal", "record high"];
  let sites = {}, fields = [], generatedAt = null;

  async function load() {
    try {
      const r = await fetch("current.json", { cache: "no-cache" });
      if (!r.ok) return false;
      const j = await r.json();
      sites = j.sites || {};
      fields = j.fields || [];
      generatedAt = j.generated_at || null;
      return true;
    } catch (_) {
      return false;
    }
  }

  function get(id) {
    const row = sites[id];
    if (!row) return null;
    const o = {};
    fields.forEach((f, i) => { o[f] = row[i]; });
    return o;
  }

  //: Forecast to climb well past normal: the peak lands above p75 for its day
  //: AND is at least 1.5x today's flow. Both, so a gauge that merely sits
  //: high, or doubles from a trickle, is not flagged. Judged on the NEAR peak
  //: (next 5 days, current_conditions.NEAR_DAYS) when the build provides it:
  //: days 6-14 of the blend run wet, and flagged ~1 in 9 gauges in Nov 2026.
  function nearPeak(c) {
    if (!c) return null;
    return "near_cls" in c
      ? { q: c.near_q, date: c.near_date, cls: c.near_cls, ratio: c.near_ratio }
      : { q: c.peak_q, date: c.peak_date, cls: c.peak_cls, ratio: c.ratio };
  }
  function rising(c) {
    const p = nearPeak(c);
    return !!p && p.cls >= 4 && Number.isFinite(p.ratio) && p.ratio >= 1.5;
  }

  return { load, get, rising, nearPeak, COLORS, NAMES, get generatedAt() { return generatedAt; } };
})();

const NWS = (() => {
  const BASE = "https://api.water.noaa.gov/nwps/v1/gauges/";
  const cache = new Map();
  const CATS = ["action", "minor", "moderate", "major"];
  const CAT_COLORS = { action: "#ffd23f", minor: "#ff9f43", moderate: "#ff4d4d", major: "#d14bff" };

  async function _json(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  function _interp(xs, ys, x) {
    if (!Number.isFinite(x) || xs.length < 2 || x < xs[0] || x > xs[xs.length - 1]) return null;
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
    const f = xs[hi] === xs[lo] ? 0 : (x - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] + f * (ys[hi] - ys[lo]);
  }

  /** Everything NWS knows about one USGS gauge, or null if it has no NWS gauge.
   *  Resolves once per site per page load; individual sub-requests that fail
   *  just leave their field empty. */
  function lookup(usgsId) {
    if (cache.has(usgsId)) return cache.get(usgsId);
    const p = (async () => {
      let g;
      try { g = await _json(BASE + encodeURIComponent(usgsId)); } catch (_) { return null; }
      if (!g || !g.lid || String(g.usgsId || "") !== String(usgsId)) return null;
      const out = {
        lid: g.lid, name: g.name, wfo: g.wfo?.abbreviation,
        stageUnits: g.flood?.stageUnits || "ft",
        thresholds: [], rating: null, obs: null, fcst: [], fcstIssued: null,
        obsCategory: g.status?.observed?.floodCategory,
        fcstCategory: g.status?.forecast?.floodCategory,
        url: `https://water.noaa.gov/gauges/${g.lid.toLowerCase()}`,
      };
      const so = g.status?.observed;
      if (so && so.primary > -900 && so.primaryUnit === "ft") {
        out.obs = { stage: so.primary, time: so.validTime,
                    flow: so.secondary > -900 ? so.secondary * (so.secondaryUnit === "kcfs" ? 1000 : 1) : null };
      }
      const [rat, sf] = await Promise.allSettled([
        _json(BASE + g.lid + "/ratings"), _json(BASE + g.lid + "/stageflow/forecast"),
      ]);
      if (rat.status === "fulfilled" && (rat.value?.data || []).length > 1) {
        const d = rat.value.data.filter(r => Number.isFinite(r.stage) && Number.isFinite(r.flow))
          .sort((a, b) => a.stage - b.stage);
        out.rating = { stage: d.map(r => r.stage), flow: d.map(r => r.flow) };
      }
      if (sf.status === "fulfilled") {
        const f = sf.value?.forecast || sf.value || {};
        const mult = f.secondaryUnits === "kcfs" ? 1000 : 1;
        out.fcstIssued = f.issuedTime || null;
        out.fcst = (f.data || [])
          .filter(p => Number.isFinite(p.secondary) && p.secondary > -900)
          .map(p => ({ date: p.validTime, q_cfs: p.secondary * mult, stage: p.primary }));
      }
      for (const k of CATS) {
        const c = g.flood?.categories?.[k];
        if (!c || !(c.stage > -900 || c.flow > 0)) continue;
        const stage = c.stage > -900 ? c.stage : null;
        const flow = c.flow > 0 ? c.flow : stageToFlow(out, stage);
        out.thresholds.push({ cat: k, stage, flow, color: CAT_COLORS[k] });
      }
      return out;
    })();
    cache.set(usgsId, p);
    return p;
  }

  function stageToFlow(n, stage) {
    return n?.rating && stage != null ? _interp(n.rating.stage, n.rating.flow, stage) : null;
  }
  function flowToStage(n, flow) {
    return n?.rating ? _interp(n.rating.flow, n.rating.stage, flow) : null;
  }
  /** Highest threshold a flow reaches, or null. */
  function categoryFor(n, flow) {
    let hit = null;
    for (const t of n?.thresholds || []) if (t.flow != null && flow >= t.flow) hit = t;
    return hit;
  }
  return { lookup, stageToFlow, flowToStage, categoryFor, CAT_COLORS };
})();

/** Gage height from USGS (parameter 00065), for gauges NWS does not carry. */
async function usgsGageHeight(siteId) {
  try {
    const r = await fetch(`https://waterservices.usgs.gov/nwis/iv/?format=json&sites=${siteId}&parameterCd=00065&period=PT12H&siteStatus=all`);
    if (!r.ok) return null;
    const v = (await r.json())?.value?.timeSeries?.[0]?.values?.[0]?.value || [];
    const last = v[v.length - 1];
    const ft = last ? parseFloat(last.value) : NaN;
    return Number.isFinite(ft) && ft > -900 ? { stage: ft, time: last.dateTime } : null;
  } catch (_) {
    return null;
  }
}

const Favs = (() => {
  const KEY = "rw2.favorites";
  function all() {
    try { return JSON.parse(localStorage.getItem(KEY) || "[]").filter(x => typeof x === "string"); }
    catch (_) { return []; }
  }
  function has(id) { return all().includes(id); }
  function toggle(id) {
    const cur = all();
    const next = cur.includes(id) ? cur.filter(x => x !== id) : [id, ...cur];
    try { localStorage.setItem(KEY, JSON.stringify(next.slice(0, 200))); } catch (_) {}
    return next.includes(id);
  }
  return { all, has, toggle };
})();

const Search = (() => {
  let index = [];
  function build(stations) {
    index = stations.map(s => ({
      s, id: s.id,
      hay: `${s.id} ${s.name || ""} ${s.state || ""}`.toLowerCase().replace(/[.,]/g, " "),
    }));
  }
  function query(q, limit = 12) {
    const toks = q.toLowerCase().replace(/[.,]/g, " ").split(/\s+/).filter(Boolean);
    if (!toks.length) return [];
    const out = [];
    for (const e of index) {
      if (!toks.every(t => e.hay.includes(t))) continue;
      // Exact id first, then names that START with the first token (the river
      // name), then everything else, each alphabetically.
      const score = e.id === toks[0] ? 0 : e.hay.split(" ").slice(1).some(w => w.startsWith(toks[0])) ? 1 : 2;
      out.push([score, e.s]);
    }
    out.sort((a, b) => a[0] - b[0] || String(a[1].name).localeCompare(String(b[1].name)));
    return out.slice(0, limit).map(x => x[1]);
  }
  return { build, query };
})();

const DeepLink = {
  read() {
    const m = /(?:^|[#&])site=([0-9A-Za-z]+)/.exec(location.hash);
    return m ? m[1] : null;
  },
  write(id) {
    const h = id ? `#site=${id}` : "";
    if (location.hash !== h) history.replaceState(null, "", h || location.pathname + location.search);
  },
};
