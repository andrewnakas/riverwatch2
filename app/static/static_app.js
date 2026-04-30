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

function pickDateTicks(xmin, xmax, targetCount = 7) {
  const spanDays = Math.max(1, (xmax - xmin) / MS_PER_DAY);
  const candidateSteps = [1, 2, 3, 7, 14, 30, 60, 90, 180, 365];
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

function drawChart(canvas, history, members, blend) {
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

  const allPts = series.flatMap(s => s.points);
  if (!allPts.length) return;
  const xs = allPts.map(p => Date.parse(p.date));
  const ys = allPts.map(p => p.q_cfs).filter(v => v != null && Number.isFinite(v));
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = 0, ymax = Math.max(...ys) * 1.15 || 1;

  const pad = { l: 64, r: 64, t: 18, b: 34 };
  const xToPx = x => pad.l + (x - xmin) / (xmax - xmin) * (w - pad.l - pad.r);
  const yToPx = y => h - pad.b - (y - ymin) / (ymax - ymin) * (h - pad.t - pad.b);

  // Horizontal gridlines + dual CFS labels (left + right mirror)
  ctx.strokeStyle = "#1f2942"; ctx.lineWidth = 1;
  ctx.font = "10px Inter, sans-serif";
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + i * (h - pad.t - pad.b) / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    const yval = ymax - i * (ymax - ymin) / 4;
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

  // X-axis: span-aware date ticks
  const { ticks, step } = pickDateTicks(xmin, xmax, 7);
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
    setupClimatologyControls(canvas, stats.rows);
    drawClimatology(canvas, stats.rows);
  } else {
    canvas.style.display = "none";
    const ctrl = document.getElementById("climatology-controls");
    if (ctrl) ctrl.innerHTML = "";
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
let _climState = { rows: [], range: [0, 365], dragStartDoy: null, dragEndDoy: null };

function _doyRangeContains(range, doy) {
  const [a, b] = range;
  if (a <= b) return doy >= a && doy <= b;
  return doy >= a || doy <= b; // wrap
}

function setupClimatologyControls(canvas, rows) {
  const ctrl = document.getElementById("climatology-controls");
  if (!ctrl) return;
  _climState = { rows, range: [0, 365], dragStartDoy: null, dragEndDoy: null };
  ctrl.innerHTML = `
    <div class="clim-controls">
      ${Object.entries(SEASON_PRESETS).map(([k, v]) =>
        `<button data-season="${k}" class="clim-btn${k === "full" ? " active" : ""}">${v.label}</button>`).join("")}
      <button id="clim-reset" class="clim-btn clim-reset">Reset zoom</button>
      <span class="clim-hint">drag on chart to zoom</span>
    </div>
  `;
  ctrl.querySelectorAll("button[data-season]").forEach(b => {
    b.addEventListener("click", () => {
      ctrl.querySelectorAll("button[data-season]").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      _climState.range = SEASON_PRESETS[b.dataset.season].range.slice();
      drawClimatology(canvas, _climState.rows);
    });
  });
  document.getElementById("clim-reset")?.addEventListener("click", () => {
    ctrl.querySelectorAll("button[data-season]").forEach(x => x.classList.remove("active"));
    ctrl.querySelector('[data-season="full"]')?.classList.add("active");
    _climState.range = [0, 365];
    drawClimatology(canvas, _climState.rows);
  });

  // Drag-to-zoom: capture range on canvas; release to apply.
  const _toDoy = (e) => {
    const rect = canvas.getBoundingClientRect();
    const fracX = (e.clientX - rect.left) / rect.width;
    // Match the drawing pad so the doy mapping stays in sync with the rendered axis.
    const padL = 50, padR = 12;
    const innerFrac = (fracX * canvas.width - padL) / (canvas.width - padL - padR);
    return Math.max(0, Math.min(365, Math.round(innerFrac * 365)));
  };
  canvas.onmousedown = (e) => {
    _climState.dragStartDoy = _toDoy(e);
    _climState.dragEndDoy = _climState.dragStartDoy;
    drawClimatology(canvas, _climState.rows);
  };
  canvas.onmousemove = (e) => {
    if (_climState.dragStartDoy == null) return;
    _climState.dragEndDoy = _toDoy(e);
    drawClimatology(canvas, _climState.rows);
  };
  canvas.onmouseup = (e) => {
    if (_climState.dragStartDoy == null) return;
    const a = _climState.dragStartDoy;
    const b = _toDoy(e);
    _climState.dragStartDoy = null;
    _climState.dragEndDoy = null;
    if (Math.abs(b - a) >= 4) {
      _climState.range = [Math.min(a, b), Math.max(a, b)];
      ctrl.querySelectorAll("button[data-season]").forEach(x => x.classList.remove("active"));
    }
    drawClimatology(canvas, _climState.rows);
  };
  canvas.onmouseleave = () => {
    if (_climState.dragStartDoy != null) {
      _climState.dragStartDoy = null;
      _climState.dragEndDoy = null;
      drawClimatology(canvas, _climState.rows);
    }
  };
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

  // Axes & gridlines: month markers that fall inside the visible range.
  ctx.strokeStyle = "#26304b";
  ctx.fillStyle = "#7a86a6";
  ctx.font = "11px sans-serif";
  const months = [[0, "Jan"], [1, "Feb"], [2, "Mar"], [3, "Apr"], [4, "May"], [5, "Jun"],
                  [6, "Jul"], [7, "Aug"], [8, "Sep"], [9, "Oct"], [10, "Nov"], [11, "Dec"]];
  let lastLabelX = -Infinity;
  for (const [m, label] of months) {
    const doy = Math.floor((Date.UTC(2024, m, 1) - Date.UTC(2024, 0, 1)) / 86400000);
    if (!_doyRangeContains(range, doy)) continue;
    const x = x2px(doy);
    if (x - lastLabelX < 28) continue; // skip if labels would overlap on a tight zoom
    lastLabelX = x;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + innerH); ctx.stroke();
    ctx.fillText(label, x + 2, H - 6);
  }
  ctx.fillText(useLog ? `${yMax.toFixed(0)} cfs (log)` : `${yMax.toFixed(0)} cfs`, 4, pad.t + 10);
  ctx.fillText(useLog ? `${yMin.toFixed(0)} cfs` : "0 cfs", 4, pad.t + innerH);

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

  // Drag-zoom selection overlay
  if (_climState.dragStartDoy != null && _climState.dragEndDoy != null
      && _climState.dragStartDoy !== _climState.dragEndDoy) {
    const a = Math.min(_climState.dragStartDoy, _climState.dragEndDoy);
    const b = Math.max(_climState.dragStartDoy, _climState.dragEndDoy);
    const xa = x2px(a), xb = x2px(b);
    ctx.fillStyle = "rgba(255, 209, 102, 0.15)";
    ctx.fillRect(xa, pad.t, xb - xa, innerH);
    ctx.strokeStyle = "#ffd166";
    ctx.lineWidth = 1;
    ctx.strokeRect(xa, pad.t, xb - xa, innerH);
  }

  // Legend
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
