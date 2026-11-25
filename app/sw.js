// Offline cache: the core app is precached on first visit. The optional AI-note model and its runtime
// (~390 MB) are cached on first use by the fetch handler below, after the user opts in.
const CACHE = "chexlite-v4";  // bump whenever cached files change (old caches are deleted on activate)
const ASSETS = [
  "./", "index.html", "style.css", "app.js", "note.js", "llm.js", "icon.svg", "manifest.webmanifest",
  "model/meta.json", "model/chexlite.onnx",
  "vendor/ort/ort.wasm.min.js", "vendor/ort/ort-wasm-simd-threaded.wasm", "vendor/ort/ort-wasm-simd-threaded.mjs",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// Cross-origin isolation: GitHub Pages can't send COOP/COEP headers, so add them here. That turns on
// SharedArrayBuffer, which multithreaded WASM needs for the AI note. Everything the app loads is same-origin.
function isolate(res) {
  if (!res || res.type === "opaque" || res.status === 0) return res;
  const h = new Headers(res.headers);
  h.set("Cross-Origin-Opener-Policy", "same-origin");
  h.set("Cross-Origin-Embedder-Policy", "require-corp");
  h.set("Cross-Origin-Resource-Policy", "same-origin");
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET" || new URL(e.request.url).origin !== location.origin) return;
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request).then((res) => {
    if (res.ok) {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(e.request, copy));
    }
    return res;
  })).then(isolate));
});
