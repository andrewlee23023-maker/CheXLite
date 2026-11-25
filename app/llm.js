// On-device clinician note: fine-tuned Gemma 3 270M (4-bit ONNX) via transformers.js, WASM backend.
// The model sees ONLY the structured classifier output (never the image) and its note is shown only if
// note.js verify() passes; otherwise the app keeps the template. Everything is served from this origin,
// and the service worker caches it after the first load, so it works offline from then on.
import { pipeline, env } from "./vendor/tjs/transformers.js";

const BASE = new URL("./", import.meta.url).href;
// Serve the model from this origin through the "remote" code path: in transformers.js 4.3.2 the
// file-existence check skips local models whose path is an http(s) URL, so the tokenizer would never
// load. Nothing leaves this origin; files are fetched as BASE + "llm-model/" + filename.
env.allowLocalModels = false;
env.allowRemoteModels = true;
// The service worker already caches the model for offline use; a second copy in transformers.js's own
// cache would double storage and keep stale weights after an update.
env.useBrowserCache = false;
env.remoteHost = BASE;
env.remotePathTemplate = "{model}/";
env.backends.onnx.wasm.wasmPaths = {
  mjs: BASE + "vendor/tjs/ort-wasm-simd-threaded.asyncify.mjs",
  wasm: BASE + "vendor/tjs/ort-wasm-simd-threaded.asyncify.wasm",
};

const f2 = (x) => x.toFixed(2);

// Mirror of llm/note_rules.py facts_to_prompt(): the exact format the model was trained on.
export function factsToPrompt(f) {
  const lines = [`refer: ${f.refer ? "yes" : "no"} (score ${f2(f.any_score)}, threshold ${f2(f.refer_threshold)})`];
  for (const x of f.findings) {
    const state = x.flagged ? "FLAGGED" : "not flagged";
    const reg = x.flagged && x.region ? `, ${x.region}` : "";
    lines.push(`${x.name}: ${f2(x.p)} / ${f2(x.thr)} ${state}${reg}`);
  }
  return "Model output:\n" + lines.join("\n") + "\nWrite the clinician note.";
}

// Speed: multithreaded WASM needs cross-origin isolation (sw.js adds the COOP/COEP headers, since
// GitHub Pages can't). WebGPU is used when the browser offers an adapter, with WASM as the fallback.
const cores = navigator.hardwareConcurrency || 2;
export function defaultThreads() {
  return self.crossOriginIsolated ? Math.max(1, Math.min(4, cores - 1)) : 1;
}
async function hasWebGPU() {
  try { return !!(navigator.gpu && (await navigator.gpu.requestAdapter())); } catch { return false; }
}

let gen = null;
export let backend = null;  // "webgpu" | "wasm (N threads)", for the status line and tests
// opts (tests only): { device: "webgpu" | "wasm", threads: N }
export function loadModel(onProgress, opts = {}) {
  gen ??= (async () => {
    env.backends.onnx.wasm.numThreads = opts.threads ?? defaultThreads();
    const make = (device) => pipeline("text-generation", "llm-model", { dtype: "q4", device, progress_callback: onProgress });
    const wantGPU = opts.device ? opts.device === "webgpu" : await hasWebGPU();
    if (wantGPU) {
      try { const g = await make("webgpu"); backend = "webgpu"; return g; }
      catch (e) { console.warn("WebGPU unavailable, using WASM:", e); }
    }
    const g = await make("wasm");
    backend = `wasm (${env.backends.onnx.wasm.numThreads} threads)`;
    return g;
  })().catch((e) => { gen = null; throw e; });  // allow a retry after a failed load
  return gen;
}

export async function generateNote(facts) {
  const g = await loadModel();
  const out = await g([{ role: "user", content: factsToPrompt(facts) }], { max_new_tokens: 160, do_sample: false });
  const msgs = out[0].generated_text;
  return (Array.isArray(msgs) ? msgs.at(-1).content : String(msgs)).trim();
}
