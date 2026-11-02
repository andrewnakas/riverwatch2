/* RiverWatch2 service worker: network-first, cache as a fallback.
 *
 * Forecasts change every build, so nothing is ever served from cache while the
 * network answers. The cache only exists so a gauge you already opened still
 * shows its last forecast with no signal -- at a put-in, say. Cross-origin
 * requests (USGS, NOAA, map tiles) are left entirely to the browser. */
const CACHE = "rw2-v1";
const SHELL = ["./", "index.html", "static/app.js", "static/rivers.js", "static/styles.css",
               "static/river_click.js", "static/river_transfer.js", "static/member_notes.js",
               "stations.json"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).catch(() => {}).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;
  e.respondWith(
    fetch(req).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
      return res;
    }).catch(() => caches.match(req).then(r => r || caches.match("index.html")))
  );
});
