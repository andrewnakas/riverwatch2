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
  const r = await fetch("/api/stations");
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

  const pad = { l: 50, r: 14, t: 14, b: 28 };
  const xToPx = x => pad.l + (x - xmin) / (xmax - xmin) * (w - pad.l - pad.r);
  const yToPx = y => h - pad.b - (y - ymin) / (ymax - ymin) * (h - pad.t - pad.b);

  // grid
  ctx.strokeStyle = "#1f2942"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + i * (h - pad.t - pad.b) / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    const yval = ymax - i * (ymax - ymin) / 4;
    ctx.fillStyle = "#7c87a8"; ctx.font = "10px Inter, sans-serif"; ctx.textAlign = "right";
    ctx.fillText(fmtNumber(yval), pad.l - 4, y + 3);
  }
  // x ticks: first/mid/last
  ctx.textAlign = "center";
  for (const t of [xmin, (xmin + xmax) / 2, xmax]) {
    const d = new Date(t).toISOString().slice(5, 10);
    ctx.fillText(d, xToPx(t), h - 10);
  }

  // a vertical "today" marker at end of history
  if (history.length) {
    const tx = xToPx(Date.parse(history[history.length - 1].date));
    ctx.strokeStyle = "#33446f"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(tx, pad.t); ctx.lineTo(tx, h - pad.b); ctx.stroke();
    ctx.setLineDash([]);
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

  // legend
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
  const cachedTag = payload.cached ? " (cached)" : " (fresh run)";
  status.textContent = `Issued ${payload.issued_at?.slice(0, 19)?.replace("T", " ")} UTC${cachedTag}`;

  const sumEl = document.getElementById("forecast-summary");
  const blend = payload.blend || [];
  const t1 = blend[0]?.q_cfs;
  const t7 = blend[blend.length - 1]?.q_cfs;
  sumEl.innerHTML = `
    <table>
      <tr><th>Chosen on rolling MAE</th><td>${payload.chosen}</td></tr>
      <tr><th>Day +1 blend</th><td>${fmtNumber(t1)} cfs</td></tr>
      <tr><th>Day +7 blend</th><td>${fmtNumber(t7)} cfs</td></tr>
    </table>
  `;

  drawChart(document.getElementById("chart"), payload.history || [], payload.members || {}, blend);

  const rows = Object.entries(payload.weights || {})
    .sort((a, b) => b[1] - a[1])
    .map(([name, w]) => `
      <tr>
        <td>${name}</td>
        <td>${(w * 100).toFixed(1)}%</td>
        <td>${fmtNumber(payload.rolling_mae?.[name])}</td>
      </tr>`).join("");
  document.getElementById("member-table").innerHTML = `
    <table>
      <thead><tr><th>Member</th><th>Weight</th><th>Rolling MAE (cfs)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${(payload.notes || []).length ? `<div class="notes">notes: ${payload.notes.join("; ")}</div>` : ""}
  `;
}

let inflightStation = null;

async function selectStation(station, { refresh = false } = {}) {
  document.getElementById("panel-empty").style.display = "none";
  document.getElementById("panel-content").style.display = "block";
  document.getElementById("station-title").textContent = `${station.id} — ${station.name}`;
  renderMeta(station);
  document.getElementById("forecast-status").textContent = "Running forecast…";
  document.getElementById("forecast-summary").innerHTML = "";
  document.getElementById("member-table").innerHTML = "";

  inflightStation = station.id;
  const url = `/api/forecast/${station.id}` + (refresh ? "?refresh=1" : "");
  try {
    const r = await fetch(url);
    const j = await r.json();
    if (inflightStation !== station.id) return;
    if (!r.ok) {
      document.getElementById("forecast-status").textContent = `error: ${j.error || r.status}`;
      return;
    }
    renderForecast(j);
  } catch (exc) {
    document.getElementById("forecast-status").textContent = `error: ${exc.message}`;
  }
}

document.getElementById("refresh-btn").addEventListener("click", () => {
  const title = document.getElementById("station-title").textContent;
  const id = title.split(" — ")[0];
  const m = Array.from(markersById.entries()).find(([sid]) => sid === id);
  if (m) {
    const station = { id, name: title.split(" — ").slice(1).join(" — "), lat: m[1].getLatLng().lat, lon: m[1].getLatLng().lng, state: "" };
    selectStation(station, { refresh: true });
  }
});

loadStations();
