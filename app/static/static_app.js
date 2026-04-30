"use strict";

const MOUNTAIN_WEST = new Set(["MT", "WY", "ID", "CO", "UT", "NV", "NM", "AZ"]);

const map = L.map("map", { worldCopyJump: true }).setView([45, -110], 4);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18,
  attribution: "&copy; OpenStreetMap",
}).addTo(map);

const cluster = L.markerClusterGroup({ maxClusterRadius: 40 });
map.addLayer(cluster);

const markersById = new Map();

function colorForState(state) {
  if (state === "AK") return "#4cc8ff";
  if (MOUNTAIN_WEST.has(state)) return "#ff8a4c";
  return "#b8c1d6";
}

function makeIcon(state) {
  const color = colorForState(state);
  return L.divIcon({
    className: "rw-marker",
    html: `<div style="background:${color}; width:14px; height:14px; border-radius:50%; border:2px solid #0b1020; box-shadow: 0 0 6px rgba(0,0,0,0.6);"></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });
}

async function loadStations() {
  const r = await fetch("stations.json");
  const j = await r.json();
  const bounds = [];
  for (const s of j.stations) {
    if (s.lat == null || s.lon == null) continue;
    const m = L.marker([s.lat, s.lon], { icon: makeIcon(s.state) });
    m.bindTooltip(`${s.id} — ${s.name}`);
    m.on("click", () => selectStation(s));
    cluster.addLayer(m);
    markersById.set(s.id, m);
    bounds.push([s.lat, s.lon]);
  }
  if (bounds.length) map.fitBounds(bounds, { padding: [40, 40] });

  // Show build freshness in the header
  try {
    const sr = await fetch("index_summary.json");
    if (sr.ok) {
      const sj = await sr.json();
      const note = document.createElement("p");
      note.style.fontSize = "12px";
      note.style.color = "#aab7d4";
      note.style.margin = "4px 0 0";
      note.innerHTML = `Built ${sj.generated_at} · ${sj.stations_succeeded}/${sj.stations_total} stations · build took ${sj.build_seconds}s`;
      document.querySelector(".topbar").appendChild(note);
    }
  } catch (_) {}
}

function fmtNumber(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function renderMeta(station) {
  const el = document.getElementById("station-meta");
  el.innerHTML = `
    <b>${station.id}</b> · ${station.state}<br/>
    ${station.name}<br/>
    <span style="opacity:0.7">drainage ${fmtNumber(station.drain_area_sqmi)} mi² · alt ${fmtNumber(station.alt_ft)} ft</span>
  `;
}

const MS_PER_DAY = 86400000;

function pickDateTicks(xmin, xmax, targetCount = 12) {
  const spanDays = Math.max(1, (xmax - xmin) / MS_PER_DAY);
  const candidateSteps = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365];
  let step = candidateSteps[0];
  for (const s of candidateSteps) {
    if (spanDays / s <= targetCount) { step = s; break; }
    step = s;
  }
  const ticks = [];
  if (step >= 30) {
    // Snap to month boundaries so labels read "May 1, Jun 1, …" rather than
    // arbitrary 30-day offsets.
    const startMonth = new Date(xmin);
    startMonth.setUTCDate(1);
    let cursor = startMonth.getTime();
    while (cursor <= xmax) {
      if (cursor >= xmin) ticks.push(cursor);
      const dt = new Date(cursor);
      dt.setUTCMonth(dt.getUTCMonth() + Math.max(1, Math.round(step / 30)));
      cursor = dt.getTime();
    }
  } else {
    const startDay = Math.ceil(xmin / MS_PER_DAY);
    for (let d = startDay; d * MS_PER_DAY <= xmax; d += step) {
      ticks.push(d * MS_PER_DAY);
    }
  }
  return { ticks, step };
}

function fmtTick(ts, step, prevYear) {
  const d = new Date(ts);
  const mo = d.toLocaleString(undefined, { month: "short", timeZone: "UTC" });
  const day = d.getUTCDate();
  const yr = d.getUTCFullYear();
  // Label crosses a year boundary -> stamp the year. Otherwise short label.
  if (step >= 60) {
    return prevYear !== yr ? `${mo} '${yr.toString().slice(2)}` : mo;
  }
  if (step >= 30) {
    return prevYear !== yr ? `${mo} '${yr.toString().slice(2)}` : mo;
  }
  return prevYear !== yr ? `${mo} ${day} '${yr.toString().slice(2)}` : `${mo} ${day}`;
}

// State for forecast chart's zoom slider. range = [frac0, frac1] in [0, 1] of the
// full series x-extent. _fcst.full = {xmin, xmax}; the renderer uses range to
// derive the visible window.
let _fcst = { canvas: null, history: [], members: {}, blend: [], full: null, range: [0, 1] };

function drawChart(canvas, history, members, blend) {
  _fcst.canvas = canvas;
  _fcst.history = history;
  _fcst.members = members;
  _fcst.blend = blend;
  const allPts = [
    ...history,
    ...blend,
    ...Object.values(members).flat(),
  ];
  if (!allPts.length) return;
  const xs0 = allPts.map(p => Date.parse(p.date));
  _fcst.full = { xmin: Math.min(...xs0), xmax: Math.max(...xs0) };
  if (_fcst.range[0] >= _fcst.range[1]) _fcst.range = [0, 1];
  _renderForecast();
}

function _renderForecast() {
  const canvas = _fcst.canvas;
  if (!canvas || !_fcst.full) return;
  const history = _fcst.history, members = _fcst.members, blend = _fcst.blend;
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#131a2d"; ctx.fillRect(0, 0, w, h);

  const series = [];
  series.push({ name: "history", color: "#e7ecf3", points: history });
  series.push({ name: "blend", color: "#ffd166", points: blend, dashed: true, width: 3 });
  const memColors = { persistence_lag1: "#5fa0ff", runoff_ridge: "#7be07b", chronos_bolt: "#ff8a4c" };
  for (const [name, pts] of Object.entries(members)) {
    series.push({ name, color: memColors[name] || "#cccccc", points: pts, dashed: true });
  }

  const span = _fcst.full.xmax - _fcst.full.xmin || 1;
  const xmin = _fcst.full.xmin + _fcst.range[0] * span;
  const xmax = _fcst.full.xmin + _fcst.range[1] * span;

  // y-extent over points visible in the current zoom window only
  const ys = [];
  for (const s of series) {
    for (const p of s.points) {
      const t = Date.parse(p.date);
      if (t < xmin || t > xmax) continue;
      if (p.q_cfs != null && Number.isFinite(p.q_cfs)) ys.push(p.q_cfs);
    }
  }
  const ymin = 0, ymax = (ys.length ? Math.max(...ys) : 1) * 1.15 || 1;

  const pad = { l: 64, r: 64, t: 18, b: 34 };
  const xToPx = x => pad.l + (x - xmin) / (xmax - xmin) * (w - pad.l - pad.r);
  const yToPx = y => h - pad.b - (y - ymin) / (ymax - ymin) * (h - pad.t - pad.b);

  // Horizontal gridlines + dual CFS labels (left + right mirror)
  ctx.strokeStyle = "#1f2942"; ctx.lineWidth = 1;
  ctx.font = "10px Inter, sans-serif";
  const Y_DIVS = 8;
  for (let i = 0; i <= Y_DIVS; i++) {
    const y = pad.t + i * (h - pad.t - pad.b) / Y_DIVS;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    const yval = ymax - i * (ymax - ymin) / Y_DIVS;
    ctx.fillStyle = "#7c87a8";
    ctx.textAlign = "right";
    ctx.fillText(fmtNumber(yval), pad.l - 4, y + 3);
    ctx.textAlign = "left";
    ctx.fillText(fmtNumber(yval), w - pad.r + 4, y + 3);
  }
  // Y-axis "cfs" unit labels
  ctx.fillStyle = "#aab7d4";
  ctx.textAlign = "right";
  ctx.fillText("cfs", pad.l - 4, pad.t - 4);
  ctx.textAlign = "left";
  ctx.fillText("cfs", w - pad.r + 4, pad.t - 4);

  // Faint forecast-zone shading: distinguishes the prediction window visually.
  if (history.length && blend.length) {
    const txNow = xToPx(Date.parse(history[history.length - 1].date));
    const xRight = w - pad.r;
    if (xRight > txNow) {
      ctx.fillStyle = "rgba(255, 209, 102, 0.05)";
      ctx.fillRect(txNow, pad.t, xRight - txNow, h - pad.t - pad.b);
    }
  }

  // X-axis: span-aware date ticks (denser so users can read exact dates)
  const { ticks, step } = pickDateTicks(xmin, xmax, 12);
  ctx.fillStyle = "#7c87a8";
  ctx.textAlign = "center";
  ctx.strokeStyle = "#1f2942";
  let prevYear = null;
  for (const t of ticks) {
    if (t < xmin || t > xmax) continue;
    const px = xToPx(t);
    ctx.beginPath(); ctx.moveTo(px, h - pad.b); ctx.lineTo(px, h - pad.b + 3); ctx.stroke();
    const label = fmtTick(t, step, prevYear);
    ctx.fillText(label, px, h - 14);
    prevYear = new Date(t).getUTCFullYear();
  }
  // Always show the start-year + end-year on the bottom strip so the user has
  // an absolute anchor regardless of how the ticks fall.
  const y0 = new Date(xmin).getUTCFullYear();
  const y1 = new Date(xmax).getUTCFullYear();
  ctx.font = "9px Inter, sans-serif";
  ctx.fillStyle = "#5c6685";
  ctx.textAlign = "left";
  ctx.fillText(String(y0), pad.l, h - 2);
  ctx.textAlign = "right";
  ctx.fillText(String(y1), w - pad.r, h - 2);
  ctx.font = "10px Inter, sans-serif";

  if (history.length) {
    const tx = xToPx(Date.parse(history[history.length - 1].date));
    ctx.strokeStyle = "#33446f"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(tx, pad.t); ctx.lineTo(tx, h - pad.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#7c87a8";
    ctx.textAlign = "center";
    ctx.font = "9px Inter, sans-serif";
    ctx.fillText("now", tx, pad.t - 2);
    ctx.fillText("forecast →", tx + 32, pad.t - 2);
    ctx.font = "10px Inter, sans-serif";
  }

  for (const s of series) {
    if (!s.points.length) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 2;
    ctx.setLineDash(s.dashed ? [5, 4] : []);
    ctx.beginPath();
    let started = false;
    for (const p of s.points) {
      if (p.q_cfs == null || !Number.isFinite(p.q_cfs)) continue;
      const x = xToPx(Date.parse(p.date)), y = yToPx(p.q_cfs);
      if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.font = "11px Inter, sans-serif"; ctx.textAlign = "left";
  let lx = pad.l + 4, ly = pad.t + 12;
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, ly - 8, 10, 3);
    ctx.fillStyle = "#aab7d4";
    ctx.fillText(s.name, lx + 14, ly);
    lx += ctx.measureText(s.name).width + 36;
  }
}

// ---- Reusable zoom-slider widget ----------------------------------------
// Two fat handles (left + right) sized to represent the visible window. The
// middle band is grabbable to slide the whole window. Reports range as
// fractions in [0, 1] so callers can map to whatever axis they like.
function makeZoomSlider(container, onChange, opts = {}) {
  const minSpan = opts.minSpan ?? 0.02;
  container.style.display = "block";
  container.innerHTML = `
    <div class="zs-track">
      <div class="zs-window" data-zs="window"></div>
      <div class="zs-handle" data-zs="left"></div>
      <div class="zs-handle" data-zs="right"></div>
      <div class="zs-label" data-zs="label"></div>
    </div>
  `;
  const track = container.querySelector(".zs-track");
  const winEl = container.querySelector('[data-zs="window"]');
  const leftEl = container.querySelector('[data-zs="left"]');
  const rightEl = container.querySelector('[data-zs="right"]');
  const labelEl = container.querySelector('[data-zs="label"]');
  let state = { lo: 0, hi: 1, fmt: opts.fmt || ((lo, hi) => `${(lo*100|0)}–${(hi*100|0)}%`) };
  const layout = () => {
    const w = track.clientWidth;
    const x0 = state.lo * w, x1 = state.hi * w;
    winEl.style.left = `${x0}px`;
    winEl.style.width = `${Math.max(0, x1 - x0)}px`;
    leftEl.style.left = `${x0 - 7}px`;
    rightEl.style.left = `${x1 - 7}px`;
    labelEl.textContent = state.fmt(state.lo, state.hi);
  };
  const setRange = (lo, hi) => {
    lo = Math.max(0, Math.min(1, lo));
    hi = Math.max(0, Math.min(1, hi));
    if (hi - lo < minSpan) {
      const c = (lo + hi) / 2;
      lo = Math.max(0, c - minSpan / 2);
      hi = Math.min(1, c + minSpan / 2);
    }
    state.lo = lo; state.hi = hi;
    layout();
    onChange([lo, hi]);
  };
  const fracFromEvent = (e) => {
    const r = track.getBoundingClientRect();
    return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  };
  let drag = null;
  const onDown = (kind) => (e) => {
    e.preventDefault();
    drag = { kind, startFrac: fracFromEvent(e), lo0: state.lo, hi0: state.hi };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp, { once: true });
  };
  const onMove = (e) => {
    if (!drag) return;
    const f = fracFromEvent(e);
    if (drag.kind === "left") {
      setRange(Math.min(f, drag.hi0 - minSpan), drag.hi0);
    } else if (drag.kind === "right") {
      setRange(drag.lo0, Math.max(f, drag.lo0 + minSpan));
    } else if (drag.kind === "window") {
      const span = drag.hi0 - drag.lo0;
      let lo = drag.lo0 + (f - drag.startFrac);
      lo = Math.max(0, Math.min(1 - span, lo));
      setRange(lo, lo + span);
    }
  };
  const onUp = () => {
    drag = null;
    document.removeEventListener("pointermove", onMove);
  };
  leftEl.addEventListener("pointerdown", onDown("left"));
  rightEl.addEventListener("pointerdown", onDown("right"));
  winEl.addEventListener("pointerdown", onDown("window"));
  // Click on bare track recenters around click point (keep current span)
  track.addEventListener("pointerdown", (e) => {
    if (e.target !== track) return;
    const f = fracFromEvent(e);
    const span = state.hi - state.lo;
    let lo = Math.max(0, Math.min(1 - span, f - span / 2));
    setRange(lo, lo + span);
  });
  // Initial layout — onresize will keep things tidy if the panel resizes.
  window.addEventListener("resize", layout);
  state.set = setRange;
  state.get = () => [state.lo, state.hi];
  state.setFmt = (fn) => { state.fmt = fn; layout(); };
  setRange(opts.initial?.[0] ?? 0, opts.initial?.[1] ?? 1);
  return state;
}

// ---- Per-station discharge-history lazy fetch ----------------------------
const _historyCache = new Map(); // siteId -> {rows: {date: q_cfs}, ...}
async function loadStationHistory(siteId) {
  if (_historyCache.has(siteId)) return _historyCache.get(siteId);
  try {
    const r = await fetch(`history/${siteId}.json`);
    if (!r.ok) return null;
    const j = await r.json();
    _historyCache.set(siteId, j);
    return j;
  } catch (_) {
    return null;
  }
}

function renderForecast(payload) {
  document.getElementById("panel-empty").style.display = "none";
  document.getElementById("panel-content").style.display = "block";

  renderMeta(payload.station);
  document.getElementById("station-title").textContent = `${payload.station.id} — ${payload.station.name}`;
  const status = document.getElementById("forecast-status");
  status.textContent = `Built ${payload.issued_at?.slice(0, 19)?.replace("T", " ")} UTC (static)`;

  const sumEl = document.getElementById("forecast-summary");
  const blend = payload.blend || [];
  const t1 = blend[0]?.q_cfs;
  const t7 = blend[blend.length - 1]?.q_cfs;
  const strat = payload.weights_strategy || "soft_blend";
  const stratLabel = strat.startsWith("snap_to:")
    ? `auto-snap to <b>${strat.split(":")[1]}</b>`
    : "soft blend (1/MAE²)";
  sumEl.innerHTML = `
    <table>
      <tr><th>Chosen on rolling MAE</th><td>${payload.chosen}</td></tr>
      <tr><th>Blend strategy</th><td>${stratLabel}</td></tr>
      <tr><th>Day +1 blend</th><td>${fmtNumber(t1)} cfs</td></tr>
      <tr><th>Day +7 blend</th><td>${fmtNumber(t7)} cfs</td></tr>
    </table>
  `;

  drawChart(document.getElementById("chart"), payload.history || [], payload.members || {}, blend);
  setupForecastZoom();

  renderRecordStartAndStats(payload);
  renderSnotel(payload);

  const mae7 = payload.rolling_mae_h7 || {};
  const mae14 = payload.rolling_mae_h14 || {};
  const mape = payload.rolling_mape || {};
  const mape7 = payload.rolling_mape_h7 || {};
  const mape14 = payload.rolling_mape_h14 || {};
  const fmtPct = v => v == null || !isFinite(v) ? "—" : `${(v * 100).toFixed(1)}%`;
  const rows = Object.entries(payload.weights || {})
    .sort((a, b) => b[1] - a[1])
    .map(([name, w]) => `
      <tr>
        <td>${name}</td>
        <td>${(w * 100).toFixed(1)}%</td>
        <td>${fmtNumber(payload.rolling_mae?.[name])}</td>
        <td>${fmtNumber(mae7[name])}</td>
        <td>${fmtNumber(mae14[name])}</td>
        <td>${fmtPct(mape[name])}</td>
        <td>${fmtPct(mape7[name])}</td>
        <td>${fmtPct(mape14[name])}</td>
      </tr>`).join("");
  const ensembleRow = (mae7.ensemble_blend != null || mae14.ensemble_blend != null) ? `
      <tr class="ensemble-row">
        <td>ensemble_blend</td>
        <td>—</td>
        <td>—</td>
        <td>${fmtNumber(mae7.ensemble_blend)}</td>
        <td>${fmtNumber(mae14.ensemble_blend)}</td>
        <td>—</td>
        <td>${fmtPct(mape7.ensemble_blend)}</td>
        <td>${fmtPct(mape14.ensemble_blend)}</td>
      </tr>` : "";
  document.getElementById("member-table").innerHTML = `
    <table>
      <thead><tr>
        <th>Member</th><th>Weight</th>
        <th>MAE (cfs)</th><th>MAE @ 7d</th><th>MAE @ 14d</th>
        <th>MAPE</th><th>MAPE @ 7d</th><th>MAPE @ 14d</th>
      </tr></thead>
      <tbody>${rows}${ensembleRow}</tbody>
    </table>
    <details class="mae-explainer">
      <summary>What is rolling MAE?</summary>
      <p>
        <b>MAE = Mean Absolute Error</b>, in cubic-feet-per-second (cfs). It's the
        average gap between what each forecaster predicted and what actually flowed,
        across past days where we already know the answer.
      </p>
      <p>
        <b>Rolling</b> means we backtest each forecaster against the most recent
        history at this gauge — not a fixed test set. Persistence and Chronos use
        a single horizon-length holdout; the ridge model holds out the trailing
        30 days for every horizon-day. Lower is better.
      </p>
      <p>
        <b>How it drives the blend:</b> each member's weight is proportional to
        <code>1 / rolling_mae²</code>, so the forecaster with the smallest recent
        error gets a much bigger say than under linear weighting. If one model is
        decisively best at a site (≥30% lower MAE than the runner-up), the system
        <i>snaps</i> 90% of the weight onto it instead — the "Blend strategy" row
        above tells you when this happens.
      </p>
      <p style="opacity:0.7">
        Watch the weights swing across stations — Chronos dominates on smooth
        rivers it has seen pattern-likes of, ridge wins where weather forcing
        matters, and persistence quietly wins on steady baseflow.
      </p>
    </details>
    ${(payload.notes || []).length ? `<div class="notes">notes: ${payload.notes.join("; ")}</div>` : ""}
  `;
}

function renderSnotel(payload) {
  const el = document.getElementById("snotel-block");
  if (!el) return;
  const site = payload.snotel_site;
  const sum = payload.snotel_summary;
  if (!site || !sum) {
    el.innerHTML = "";
    return;
  }
  const fmtDelta = v =>
    v == null || !isFinite(v) ? "—" :
      `${v >= 0 ? "+" : ""}${Number(v).toFixed(1)} in`;
  el.innerHTML = `
    <div class="snotel">
      <div class="snotel-head">
        <b>SNOTEL</b> ${site.name || site.stationTriplet}
        <span class="snotel-sub">${site.distance_km} km · ${fmtNumber(site.elevation_ft)} ft</span>
      </div>
      <table class="snotel-table">
        <tr>
          <td><span class="snotel-lbl">SWE now</span><br/><b>${fmtNumber(sum.swe_in)} in</b></td>
          <td><span class="snotel-lbl">7-day Δ</span><br/>${fmtDelta(sum.swe_change_7d)}</td>
          <td><span class="snotel-lbl">30-day Δ</span><br/>${fmtDelta(sum.swe_change_30d)}</td>
        </tr>
      </table>
      <div class="snotel-asof">as of ${sum.as_of}</div>
    </div>
  `;
}

function todayStats(stats) {
  if (!stats || !stats.rows) return null;
  const t = new Date();
  const md = `${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")}`;
  return stats.rows.find(r => r.month_day === md) || null;
}

function renderRecordStartAndStats(payload) {
  const recEl = document.getElementById("record-start");
  const statsEl = document.getElementById("daily-stats");
  const canvas = document.getElementById("climatology-chart");

  const start = payload.record_start;
  const end = payload.record_end;
  const stats = payload.daily_stats;
  const today = todayStats(stats);

  recEl.innerHTML = start
    ? `<p class="record-start"><b>Record begins:</b> ${start}${end ? ` &middot; latest finalized: ${end}` : ""}</p>`
    : "";

  if (today) {
    const mostRecent = (payload.history && payload.history.length)
      ? payload.history[payload.history.length - 1].q_cfs : null;
    const todayDate = new Date();
    const monthName = todayDate.toLocaleString(undefined, { month: "short" });
    statsEl.innerHTML = `
      <details class="daily-stats" open>
        <summary>Daily discharge statistics for ${monthName} ${todayDate.getDate()} (${stats.rows.length} day-of-year records)</summary>
        <table class="climatology">
          <thead><tr>
            <th>Min${today.min_yr ? ` (${today.min_yr})` : ""}</th>
            <th>25th</th><th>Median</th>
            <th>Most Recent</th>
            <th>Mean</th>
            <th>75th</th>
            <th>Max${today.max_yr ? ` (${today.max_yr})` : ""}</th>
          </tr></thead>
          <tbody><tr>
            <td>${fmtNumber(today.min_va)}</td>
            <td>${fmtNumber(today.p25_va)}</td>
            <td>${fmtNumber(today.p50_va)}</td>
            <td>${fmtNumber(mostRecent)}</td>
            <td>${fmtNumber(today.mean_va)}</td>
            <td>${fmtNumber(today.p75_va)}</td>
            <td>${fmtNumber(today.max_va)}</td>
          </tr></tbody>
        </table>
      </details>
    `;
  } else {
    statsEl.innerHTML = "";
  }

  if (stats && stats.rows && stats.rows.length > 30) {
    canvas.style.display = "block";
    setupClimatologyControls(canvas, stats.rows, payload);
    drawClimatology(canvas, stats.rows);
  } else {
    canvas.style.display = "none";
    const ctrl = document.getElementById("climatology-controls");
    if (ctrl) ctrl.innerHTML = "";
    const zoomEl = document.getElementById("climatology-zoom");
    if (zoomEl) { zoomEl.style.display = "none"; zoomEl.innerHTML = ""; }
    const yrEl = document.getElementById("year-overlay-controls");
    if (yrEl) { yrEl.style.display = "none"; yrEl.innerHTML = ""; }
  }
}

// Predefined day-of-year ranges (start, end inclusive, in 0..365).
// Spring is the default whitewater season for snow-fed western rivers.
const SEASON_PRESETS = {
  full:   { label: "All year", range: [0, 365] },
  winter: { label: "Winter (Dec–Feb)", range: [335, 59] },  // wraps year boundary
  spring: { label: "Spring (Mar–May)", range: [60, 151] },
  summer: { label: "Summer (Jun–Aug)", range: [152, 243] },
  fall:   { label: "Fall (Sep–Nov)", range: [244, 334] },
  runoff: { label: "Runoff (Apr–Jul)", range: [91, 212] },
};
// _climState carries everything draw uses: rows = climatology table, range = DOY
// window, overlayYears = water years user has toggled on, history = cached daily
// rows {date: q_cfs} for this station, siteId for fetches.
let _climState = {
  rows: [], range: [0, 365], canvas: null,
  siteId: null, history: null, overlayYears: new Set(), payload: null,
  zoom: null,
};

function _doyRangeContains(range, doy) {
  const [a, b] = range;
  if (a <= b) return doy >= a && doy <= b;
  return doy >= a || doy <= b; // wrap
}

const YEAR_OVERLAY_COLORS = ["#ff8a4c", "#7be07b", "#ff6db6", "#9aa6ff", "#4cc8ff", "#ffd166", "#c084ff"];
function _colorForYear(yr, idxOf) {
  return YEAR_OVERLAY_COLORS[idxOf % YEAR_OVERLAY_COLORS.length];
}

let _fcstZoomMounted = false;
function setupForecastZoom() {
  const container = document.getElementById("chart-zoom");
  if (!container) return;
  if (_fcstZoomMounted) {
    // re-mount cleanly between station selections
    container.innerHTML = "";
    _fcstZoomMounted = false;
  }
  const fmt = ([lo, hi]) => {
    if (!_fcst.full) return "";
    const span = _fcst.full.xmax - _fcst.full.xmin;
    const a = new Date(_fcst.full.xmin + lo * span);
    const b = new Date(_fcst.full.xmin + hi * span);
    const f = d => `${d.getUTCMonth() + 1}/${d.getUTCDate()}/${d.getUTCFullYear().toString().slice(2)}`;
    return `${f(a)} – ${f(b)}`;
  };
  makeZoomSlider(container, ([lo, hi]) => {
    _fcst.range = [lo, hi];
    _renderForecast();
  }, { initial: [0, 1], minSpan: 0.04, fmt: (lo, hi) => fmt([lo, hi]) });
  _fcstZoomMounted = true;
}

function _doySpanFromRange(range) {
  const [a, b] = range;
  return a <= b ? (b - a) : (366 - a + b);
}

function setupClimatologyControls(canvas, rows, payload) {
  const ctrl = document.getElementById("climatology-controls");
  const zoomEl = document.getElementById("climatology-zoom");
  const yrEl = document.getElementById("year-overlay-controls");
  if (!ctrl || !zoomEl || !yrEl) return;
  _climState = {
    rows, range: [0, 365], canvas,
    siteId: payload.station?.id || null,
    history: null, overlayYears: new Set(), payload, zoom: null,
  };
  ctrl.innerHTML = `
    <div class="clim-controls">
      ${Object.entries(SEASON_PRESETS).map(([k, v]) =>
        `<button data-season="${k}" class="clim-btn${k === "full" ? " active" : ""}">${v.label}</button>`).join("")}
      <button id="clim-reset" class="clim-btn clim-reset">Reset</button>
      <span class="clim-hint">drag the slider below to zoom</span>
    </div>
  `;
  const applyRange = (range) => {
    _climState.range = range.slice();
    // Sync slider position to match preset (express as fractions of the year)
    if (_climState.zoom) {
      const [a, b] = range;
      if (a <= b) _climState.zoom.set(a / 365, b / 365);
      else _climState.zoom.set(0, 1); // wrap → show full; user can drag
    }
    drawClimatology(canvas, _climState.rows);
  };
  ctrl.querySelectorAll("button[data-season]").forEach(b => {
    b.addEventListener("click", () => {
      ctrl.querySelectorAll("button[data-season]").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      applyRange(SEASON_PRESETS[b.dataset.season].range);
    });
  });
  document.getElementById("clim-reset")?.addEventListener("click", () => {
    ctrl.querySelectorAll("button[data-season]").forEach(x => x.classList.remove("active"));
    ctrl.querySelector('[data-season="full"]')?.classList.add("active");
    applyRange([0, 365]);
    _climState.overlayYears.clear();
    renderYearOverlayUI();
    drawClimatology(canvas, _climState.rows);
  });

  // Zoom slider: maps fractions [0,1] of the year to DOY 0..365. Doesn't try
  // to express wrap-around — when a wrap preset is active, slider shows full.
  const fmt = ([lo, hi]) => {
    const labelFor = (frac) => {
      const day = Math.round(frac * 365);
      const d = new Date(Date.UTC(2024, 0, day + 1));
      return `${d.toLocaleString(undefined, { month: "short", timeZone: "UTC" })} ${d.getUTCDate()}`;
    };
    return `${labelFor(lo)} – ${labelFor(hi)}`;
  };
  zoomEl.innerHTML = "";
  _climState.zoom = makeZoomSlider(zoomEl, ([lo, hi]) => {
    _climState.range = [Math.round(lo * 365), Math.round(hi * 365)];
    ctrl.querySelectorAll("button[data-season]").forEach(x => x.classList.remove("active"));
    drawClimatology(canvas, _climState.rows);
  }, { initial: [0, 1], minSpan: 0.02, fmt: (lo, hi) => fmt([lo, hi]) });

  // Build year overlay UI now (before history fetch). The first click triggers fetch.
  renderYearOverlayUI();
}

function _availableYearsFor(payload) {
  // Prefer record_start/record_end; fall back to history dates.
  const out = [];
  const start = payload?.record_start;
  const end = payload?.record_end;
  if (start && end) {
    const y0 = parseInt(start.slice(0, 4), 10);
    const y1 = parseInt(end.slice(0, 4), 10);
    if (Number.isFinite(y0) && Number.isFinite(y1)) {
      for (let y = y1; y >= y0; y--) out.push(y);
    }
  }
  return out;
}

async function ensureHistoryLoaded() {
  if (_climState.history) return _climState.history;
  if (!_climState.siteId) return null;
  if (_climState.payload && _climState.payload.has_history_file === false) return null;
  const yrEl = document.getElementById("year-overlay-controls");
  const status = yrEl?.querySelector(".yr-status");
  if (status) status.textContent = "loading record…";
  const j = await loadStationHistory(_climState.siteId);
  _climState.history = j || { rows: {} };
  if (status) status.textContent = j ? `record loaded (${Object.keys(j.rows || {}).length.toLocaleString()} days)` : "no record file available";
  return _climState.history;
}

function renderYearOverlayUI() {
  const yrEl = document.getElementById("year-overlay-controls");
  if (!yrEl) return;
  const years = _availableYearsFor(_climState.payload);
  if (!years.length) { yrEl.innerHTML = ""; yrEl.style.display = "none"; return; }
  yrEl.style.display = "block";
  const thisYear = new Date().getUTCFullYear();
  const presets = [
    { key: "this", label: "This year", year: thisYear },
    { key: "last", label: "Last year", year: thisYear - 1 },
  ].filter(p => years.includes(p.year));
  const active = (y) => _climState.overlayYears.has(y) ? " active" : "";
  const presetChips = presets.map(p => {
    const cls = `yr-chip preset${active(p.year)}`;
    const color = _climState.overlayYears.has(p.year) ? `style="background:${_colorForYear(p.year, [...years].indexOf(p.year))}; color:#0b1020"` : "";
    return `<span class="${cls}" data-year="${p.year}" ${color}>${p.label} (${p.year})</span>`;
  }).join("");
  const yearChips = years.map(y => {
    const isActive = _climState.overlayYears.has(y);
    const idx = years.indexOf(y);
    const c = _colorForYear(y, idx);
    const style = isActive ? `style="background:${c}; color:#0b1020"` : "";
    return `<span class="yr-chip${active(y)}" data-year="${y}" ${style}>${y}</span>`;
  }).join("");
  yrEl.innerHTML = `
    <div class="yr-overlay-head">
      <b>Compare years</b>
      <span style="opacity:0.7">click any year to overlay it on the climatology</span>
      <span class="yr-status"></span>
    </div>
    ${presetChips ? `<div class="yr-overlay-grid" style="max-height:none">${presetChips}</div>` : ""}
    <div class="yr-overlay-grid">${yearChips}</div>
  `;
  yrEl.querySelectorAll(".yr-chip").forEach(el => {
    el.addEventListener("click", async () => {
      const y = parseInt(el.dataset.year, 10);
      if (!Number.isFinite(y)) return;
      if (_climState.overlayYears.has(y)) _climState.overlayYears.delete(y);
      else _climState.overlayYears.add(y);
      renderYearOverlayUI(); // re-render chip styles
      await ensureHistoryLoaded();
      drawClimatology(_climState.canvas, _climState.rows);
    });
  });
}

function drawClimatology(canvas, rows) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0d1224";
  ctx.fillRect(0, 0, W, H);

  // Convert MM-DD to day-of-year for x-axis. Use 2024 (leap) as reference.
  const xs = rows.map(r => {
    const [m, d] = r.month_day.split("-").map(n => parseInt(n, 10));
    return Math.floor((Date.UTC(2024, m - 1, d) - Date.UTC(2024, 0, 1)) / 86400000);
  });
  const series = ["min_va", "p25_va", "p50_va", "mean_va", "p75_va", "max_va"];

  // Filter rows to the active season/zoom range. y-axis recomputed on the
  // visible subset so a tighter window auto-scales.
  const range = (_climState && _climState.range) || [0, 365];
  const visibleIdx = [];
  for (let i = 0; i < xs.length; i++) {
    if (_doyRangeContains(range, xs[i])) visibleIdx.push(i);
  }
  if (visibleIdx.length === 0) return;
  let yMax = 0;
  for (const i of visibleIdx) for (const s of series) {
    const v = rows[i][s];
    if (v != null && isFinite(v) && v > yMax) yMax = v;
  }
  if (yMax <= 0) return;

  const pad = { l: 50, r: 12, t: 12, b: 22 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;
  const useLog = yMax > 200;
  const yMin = useLog
    ? Math.max(0.1, visibleIdx.reduce((a, i) => Math.min(a, rows[i].min_va || a), yMax))
    : 0;

  // Map a day-of-year back into the [0, 1] visible fraction. Handles wraparound
  // by treating a wrapping range as continuous via modular shifting.
  const [r0, r1] = range;
  const wraps = r0 > r1;
  const totalSpan = wraps ? (366 - r0 + r1) : (r1 - r0);
  const safeSpan = Math.max(1, totalSpan);
  const doyToFrac = (doy) => {
    if (!wraps) return (doy - r0) / safeSpan;
    return doy >= r0 ? (doy - r0) / safeSpan : (366 - r0 + doy) / safeSpan;
  };

  const x2px = doy => pad.l + Math.max(0, Math.min(1, doyToFrac(doy))) * innerW;
  const y2px = v => {
    if (useLog) {
      const lv = Math.log10(Math.max(v, yMin));
      const lmax = Math.log10(yMax);
      const lmin = Math.log10(yMin);
      return pad.t + innerH - ((lv - lmin) / (lmax - lmin)) * innerH;
    }
    return pad.t + innerH - (v / yMax) * innerH;
  };

  // Axes & gridlines: dense vertical lines at every 1st/15th of each visible month
  // when the zoom is wide; tighter ticks (every 5 days) when zoomed in. We label
  // months at the 1st only so the axis doesn't look noisy.
  ctx.strokeStyle = "#26304b";
  ctx.fillStyle = "#7a86a6";
  ctx.font = "11px sans-serif";
  const months = [[0, "Jan"], [1, "Feb"], [2, "Mar"], [3, "Apr"], [4, "May"], [5, "Jun"],
                  [6, "Jul"], [7, "Aug"], [8, "Sep"], [9, "Oct"], [10, "Nov"], [11, "Dec"]];
  // Pick a tick step in days based on the visible span so users can read exact
  // dates: ≤14d every day, ≤45d every 2d, ≤90d every 5d, ≤180d every 15d, else month-1st.
  let tickStep = 30;
  if (totalSpan <= 14) tickStep = 1;
  else if (totalSpan <= 45) tickStep = 2;
  else if (totalSpan <= 90) tickStep = 5;
  else if (totalSpan <= 180) tickStep = 15;
  let lastLabelX = -Infinity;
  // Walk DOYs; respect wrap by iterating along the visible window directly.
  const visDoys = [];
  if (!wraps) {
    for (let d = r0; d <= r1; d += 1) visDoys.push(d);
  } else {
    for (let d = r0; d <= 365; d += 1) visDoys.push(d);
    for (let d = 0; d <= r1; d += 1) visDoys.push(d);
  }
  for (let i = 0; i < visDoys.length; i++) {
    const doy = visDoys[i];
    const refDate = new Date(Date.UTC(2024, 0, doy + 1));
    const dom = refDate.getUTCDate();
    const monthIdx = refDate.getUTCMonth();
    const isFirst = dom === 1;
    const isMid = dom === 15;
    const isStep = (i % tickStep) === 0;
    if (!isFirst && !isStep && !isMid) continue;
    const x = x2px(doy);
    // Minor gridline
    if (isStep || isMid) {
      ctx.strokeStyle = isFirst ? "#33446f" : "#1f2942";
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + innerH); ctx.stroke();
    }
    if (isFirst && (x - lastLabelX) > 26) {
      lastLabelX = x;
      ctx.strokeStyle = "#33446f";
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + innerH); ctx.stroke();
      ctx.fillStyle = "#aab7d4";
      ctx.fillText(months[monthIdx][1], x + 2, H - 6);
    } else if ((isStep || isMid) && tickStep <= 5 && (x - lastLabelX) > 22) {
      lastLabelX = x;
      ctx.fillStyle = "#7a86a6";
      ctx.fillText(`${months[monthIdx][1]} ${dom}`, x + 2, H - 6);
    }
  }
  // y-axis gridlines + labels (8 divisions)
  const Y_DIVS = 8;
  for (let i = 0; i <= Y_DIVS; i++) {
    const yv = useLog
      ? Math.pow(10, Math.log10(yMin) + (Math.log10(yMax) - Math.log10(yMin)) * (1 - i / Y_DIVS))
      : yMax * (1 - i / Y_DIVS);
    const y = pad.t + (i / Y_DIVS) * innerH;
    ctx.strokeStyle = "#1f2942";
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = "#7a86a6";
    ctx.textAlign = "right";
    ctx.fillText(fmtNumber(yv, yv < 100 ? 1 : 0), pad.l - 4, y + 3);
    ctx.textAlign = "left";
  }
  ctx.textAlign = "left";
  ctx.fillStyle = "#aab7d4";
  ctx.fillText(useLog ? "cfs (log)" : "cfs", 4, pad.t + 10);

  const colors = {
    min_va: "#3a4d8a",   // dark blue
    p25_va: "#4a6db5",
    p50_va: "#6da3ff",   // median emphasized
    mean_va: "#ffd16a",
    p75_va: "#4a6db5",
    max_va: "#3a4d8a",
  };
  const widths = {
    min_va: 1, p25_va: 1, p50_va: 2.5, mean_va: 2, p75_va: 1, max_va: 1,
  };

  // For wraparound ranges, sort visible idxs by their fractional position so
  // the polyline traverses left-to-right across the rendered chart.
  const orderedIdx = visibleIdx.slice().sort((a, b) => doyToFrac(xs[a]) - doyToFrac(xs[b]));

  // Light fill between p25 and p75 to highlight the typical band
  ctx.fillStyle = "rgba(74, 109, 181, 0.18)";
  ctx.beginPath();
  let started = false;
  for (const i of orderedIdx) {
    const v = rows[i].p75_va;
    if (v == null || !isFinite(v)) continue;
    const x = x2px(xs[i]); const y = y2px(v);
    started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), started = true);
  }
  for (let k = orderedIdx.length - 1; k >= 0; k--) {
    const i = orderedIdx[k];
    const v = rows[i].p25_va;
    if (v == null || !isFinite(v)) continue;
    ctx.lineTo(x2px(xs[i]), y2px(v));
  }
  ctx.closePath(); ctx.fill();

  for (const s of series) {
    ctx.strokeStyle = colors[s];
    ctx.lineWidth = widths[s];
    ctx.beginPath();
    let firstPoint = true;
    for (const i of orderedIdx) {
      const v = rows[i][s];
      if (v == null || !isFinite(v)) continue;
      const x = x2px(xs[i]);
      const y = y2px(v);
      firstPoint ? (ctx.moveTo(x, y), firstPoint = false) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Year overlays: draw selected water years as colored polylines on top of bands.
  // Each year's daily flow is plotted as a function of DOY. Skips silently
  // if history hasn't loaded yet — clicking a chip kicks off the fetch.
  const histRows = _climState.history?.rows;
  const overlayYears = [..._climState.overlayYears].sort();
  if (histRows && overlayYears.length) {
    const allYears = _availableYearsFor(_climState.payload);
    for (const yr of overlayYears) {
      const colorIdx = allYears.indexOf(yr);
      const color = _colorForYear(yr, colorIdx >= 0 ? colorIdx : yr % YEAR_OVERLAY_COLORS.length);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      let started = false;
      // Walk in DOY order respecting wrap so the line doesn't stitch across the chart.
      const orderDoys = wraps
        ? [...Array(366 - r0)].map((_, i) => r0 + i).concat([...Array(r1 + 1)].map((_, i) => i))
        : [...Array(r1 - r0 + 1)].map((_, i) => r0 + i);
      for (const doy of orderDoys) {
        // Map (year, DOY) → 'YYYY-MM-DD'. Use Jan 1 + doy days (UTC).
        const d = new Date(Date.UTC(yr, 0, 1));
        d.setUTCDate(d.getUTCDate() + doy);
        const iso = d.toISOString().slice(0, 10);
        const v = histRows[iso];
        if (v == null || !isFinite(v)) {
          started = false; // gap in record → break the line
          continue;
        }
        const x = x2px(doy);
        const y = y2px(v);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else { ctx.lineTo(x, y); }
      }
      ctx.stroke();
    }
  }

  // Legend (climatology bands + active year overlays)
  ctx.font = "11px sans-serif";
  let lx = pad.l + 8;
  const ly = pad.t + 4;
  for (const [s, label] of [["min_va", "min"], ["p25_va", "25th"], ["p50_va", "median"], ["mean_va", "mean"], ["p75_va", "75th"], ["max_va", "max"]]) {
    ctx.fillStyle = colors[s];
    ctx.fillRect(lx, ly + 6, 10, 3);
    ctx.fillStyle = "#aab7d4";
    ctx.fillText(label, lx + 14, ly + 11);
    lx += ctx.measureText(label).width + 28;
  }
  if (overlayYears.length) {
    const allYears = _availableYearsFor(_climState.payload);
    for (const yr of overlayYears) {
      const colorIdx = allYears.indexOf(yr);
      const color = _colorForYear(yr, colorIdx >= 0 ? colorIdx : yr % YEAR_OVERLAY_COLORS.length);
      ctx.fillStyle = color;
      ctx.fillRect(lx, ly + 6, 10, 3);
      ctx.fillStyle = "#aab7d4";
      const label = String(yr);
      ctx.fillText(label, lx + 14, ly + 11);
      lx += ctx.measureText(label).width + 28;
    }
  }
}

async function selectStation(station) {
  document.getElementById("panel-empty").style.display = "none";
  document.getElementById("panel-content").style.display = "block";
  document.getElementById("station-title").textContent = `${station.id} — ${station.name}`;
  renderMeta(station);
  document.getElementById("forecast-status").textContent = "Loading forecast…";
  document.getElementById("forecast-summary").innerHTML = "";
  document.getElementById("member-table").innerHTML = "";
  document.getElementById("record-start").innerHTML = "";
  document.getElementById("daily-stats").innerHTML = "";
  const snEl = document.getElementById("snotel-block");
  if (snEl) snEl.innerHTML = "";
  const climCanvas = document.getElementById("climatology-chart");
  if (climCanvas) climCanvas.style.display = "none";
  const climCtrl = document.getElementById("climatology-controls");
  if (climCtrl) climCtrl.innerHTML = "";
  const climZoom = document.getElementById("climatology-zoom");
  if (climZoom) { climZoom.style.display = "none"; climZoom.innerHTML = ""; }
  const yrCtrl = document.getElementById("year-overlay-controls");
  if (yrCtrl) { yrCtrl.style.display = "none"; yrCtrl.innerHTML = ""; }
  const fcstZoom = document.getElementById("chart-zoom");
  if (fcstZoom) { fcstZoom.style.display = "none"; fcstZoom.innerHTML = ""; }
  _fcstZoomMounted = false;
  _fcst.range = [0, 1];

  try {
    const r = await fetch(`forecasts/${station.id}.json`);
    if (!r.ok) {
      document.getElementById("forecast-status").textContent =
        `no forecast available for this station in the latest build (HTTP ${r.status})`;
      return;
    }
    const j = await r.json();
    renderForecast(j);
  } catch (exc) {
    document.getElementById("forecast-status").textContent = `error: ${exc.message}`;
  }
}

const refreshBtn = document.getElementById("refresh-btn");
if (refreshBtn) {
  refreshBtn.style.display = "none";  // not meaningful for static deploy
}

loadStations();
