// On-device clinician note: fine-tuned Gemma 3 270M (4-bit ONNX) via transformers.js, WASM backend.
// The model sees ONLY the structured classifier output (never the image) and its note is shown only if
// note.js verify() passes; otherwise the app keeps the template. Everything is served from this origin,
// and the service worker caches it after the first load, so it works offline from then on.
import { pipeline, env } from "./vendor/tjs/transformers.min.js";

const BASE = new URL("./", import.meta.url).href;
// Serve the model from this origin through the "remote" code path: in transformers.js 4.3.2 the
// file-existence check skips local models whose path is an http(s) URL, so the tokenizer would never
// load. Nothing leaves this origin; files are fetched as BASE + "llm-model/" + filename.
env.allowLocalModels = false;
env.allowRemoteModels = true;
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

let gen = null;
export function loadModel(onProgress) {
  gen ??= pipeline("text-generation", "llm-model", { dtype: "q4", device: "wasm", progress_callback: onProgress });
  return gen;
}

export async function generateNote(facts) {
  const g = await loadModel();
  const out = await g([{ role: "user", content: factsToPrompt(facts) }], { max_new_tokens: 160, do_sample: false });
  const msgs = out[0].generated_text;
  return (Array.isArray(msgs) ? msgs.at(-1).content : String(msgs)).trim();
}
