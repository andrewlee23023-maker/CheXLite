import { buildFacts, templateNote, verify } from "./note.js";

const $ = (id) => document.getElementById(id);
const views = ["empty", "busy", "error", "result"];
const show = (v) => views.forEach((n) => ($(`view-${n}`).hidden = n !== v));

let session, meta, current;

function setStatus(text, cls) {
  $("status-text").textContent = text;
  $("status").className = `status ${cls || ""}`;
}

async function init() {
  try {
    meta = await (await fetch("model/meta.json")).json();
    ort.env.wasm.wasmPaths = new URL("vendor/ort/", location.href).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    session = await ort.InferenceSession.create("model/chexlite.onnx", { executionProviders: ["wasm"] });
    setStatus("Ready · works offline", "ready");
  } catch (e) {
    console.error(e);
    setStatus("Model failed to load", "fail");
    $("pick-label").setAttribute("aria-disabled", "true");
    return;
  }
  $("about-metrics").textContent = meta.metrics_summary || `Thresholds from ${meta.threshold_source}.`;
  loadSamples();
  const img = new URLSearchParams(location.search).get("img");  // e.g. ?img=samples/x.jpg (demo links)
  if (img && new URL(img, location.href).origin === location.origin) {
    analyze(await (await fetch(img)).blob());
  }
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
}

async function loadSamples() {
  try {
    const list = await (await fetch("samples/samples.json")).json();
    if (!list.length) return;
    for (const s of list) {
      const b = document.createElement("button");
      b.className = "chip"; b.type = "button"; b.textContent = s.name;
      b.onclick = async () => analyze(await (await fetch(`samples/${s.file}`)).blob());
      $("sample-list").append(b);
    }
    $("samples").hidden = false;
  } catch { /* no samples bundled */ }
}

// Downscale in halving steps to approximate PIL's antialiased resize used in training.
function resizeTo(src, w, h) {
  let cur = src, cw = src.width, ch = src.height;
  while (cw / 2 >= w && ch / 2 >= h) {
    const c = new OffscreenCanvas(Math.round(cw / 2), Math.round(ch / 2));
    const g = c.getContext("2d");
    g.imageSmoothingQuality = "high";
    g.drawImage(cur, 0, 0, c.width, c.height);
    cur = c; cw = c.width; ch = c.height;
  }
  const out = new OffscreenCanvas(w, h);
  const g = out.getContext("2d");
  g.imageSmoothingQuality = "high";
  g.drawImage(cur, 0, 0, w, h);
  return out;
}

// Same preprocessing as training: grayscale -> 3 channels, stretch to SxS, ImageNet normalize.
function toTensor(bitmap) {
  const s = meta.img_size;
  const d = resizeTo(bitmap, s, s).getContext("2d").getImageData(0, 0, s, s).data;
  const x = new Float32Array(3 * s * s);
  for (let i = 0; i < s * s; i++) {
    const l = Math.round(0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]) / 255;
    for (let c = 0; c < 3; c++) x[c * s * s + i] = (l - meta.mean[c]) / meta.std[c];
  }
  return new ort.Tensor("float32", x, [1, 3, s, s]);
}

async function analyze(file) {
  if (!session) return;
  show("busy");
  $("pick-text").textContent = "Check another X-ray";
  try {
    const bitmap = await createImageBitmap(file);
    const t0 = performance.now();
    const out = await session.run({ image: toTensor(bitmap) });
    const ms = Math.round(performance.now() - t0);
    const facts = buildFacts(out.probs.data, out.cams.data, meta);
    current = { bitmap, cams: out.cams.data, facts, ms };
    render();
  } catch (e) {
    console.error(e);
    $("error-text").textContent = "Try a JPEG or PNG of a frontal chest X-ray.";
    show("error");
  }
}

function render() {
  const { facts, ms } = current;
  const flagged = facts.findings.filter((f) => f.flagged);

  const dec = $("decision");
  dec.className = `card decision ${facts.refer ? "refer" : "clear"}`;
  $("decision-kicker").textContent = facts.refer ? "Suggest referral" : "No referral suggested";
  $("decision-title").textContent = facts.refer
    ? (flagged.length ? `Check ${flagged.map((f) => f.name.toLowerCase()).join(", ")}` : "Get a radiologist read")
    : "Nothing flagged";
  $("decision-sub").textContent = facts.refer
    ? "Send this study to a radiologist for review."
    : "This doesn't rule out disease. Use clinical judgment.";

  drawXray();
  const chips = $("heat-chips");
  chips.replaceChildren();
  const order = [...facts.findings].sort((a, b) => b.p / b.thr - a.p / a.thr);
  order.forEach((f, i) => {
    const c = document.createElement("button");
    c.type = "button"; c.className = "chip"; c.setAttribute("role", "radio");
    c.textContent = f.name;
    c.setAttribute("aria-checked", i === 0 ? "true" : "false");
    c.onclick = () => {
      chips.querySelectorAll(".chip").forEach((x) => x.setAttribute("aria-checked", "false"));
      c.setAttribute("aria-checked", "true");
      drawHeat(meta.labels.indexOf(f.label));
    };
    chips.append(c);
  });
  drawHeat(meta.labels.indexOf(order[0].label));

  const ul = $("findings");
  ul.replaceChildren();
  for (const f of order) {
    // Threshold maps to 50%: [0, thr] -> [0, 50%], [thr, 1] -> [50%, 100%]
    const pos = f.p < f.thr ? 50 * f.p / f.thr : 50 + 50 * (f.p - f.thr) / (1 - f.thr);
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="f-row"><span class="f-name"></span><span class="pill ${f.flagged ? "on" : "off"}">${f.flagged ? "Flagged" : "Not flagged"}</span></div>
      <div class="bar ${f.flagged ? "on" : ""}" role="img"><i style="width:${pos.toFixed(1)}%"></i></div>
      <div class="f-meta"></div>`;
    li.querySelector(".f-name").textContent = f.name;
    li.querySelector(".bar").setAttribute("aria-label", `${f.name}: score ${f.p.toFixed(2)}, threshold ${f.thr.toFixed(2)}`);
    li.querySelector(".f-meta").textContent = `Score ${f.p.toFixed(2)} · threshold ${f.thr.toFixed(2)}${f.flagged && f.region ? ` · ${f.region}` : ""}`;
    ul.append(li);
  }

  renderNote(facts);
  setStatus(`Ready · analyzed in ${ms} ms`, "ready");
  show("result");
}

async function renderNote(facts) {
  const badge = $("note-badge");
  $("note").textContent = templateNote(facts);
  badge.textContent = "Template";
  badge.className = "badge";
  // The on-device LLM note (llm.js) is optional: show it only if the verifier passes.
  try {
    const { generateNote } = await import("./llm.js");
    const text = await generateNote(facts);
    const v = verify(text, facts);
    if (v.ok) {
      $("note").textContent = text;
      badge.textContent = "AI note · checked against model output";
      badge.className = "badge verified";
    } else {
      console.warn("LLM note rejected:", v.problems);
    }
  } catch { /* LLM not bundled or failed: template stays */ }
}

function drawXray() {
  const { bitmap } = current;
  const c = $("xray");
  const w = Math.min(1024, bitmap.width);
  c.width = w; c.height = Math.round(bitmap.height * w / bitmap.width);
  const g = c.getContext("2d");
  g.imageSmoothingQuality = "high";
  g.drawImage(bitmap, 0, 0, c.width, c.height);
}

function drawHeat(k) {
  const [h, w] = meta.cam_size;
  const cam = current.cams.subarray(k * h * w, (k + 1) * h * w);
  const small = new OffscreenCanvas(w, h);
  const img = small.getContext("2d").createImageData(w, h);
  for (let i = 0; i < h * w; i++) {
    const v = cam[i];
    const a = v <= 0.35 ? 0 : (v - 0.35) / 0.65;  // weak activation stays transparent
    // amber -> red ramp, max ~60% opacity so anatomy stays visible underneath
    img.data.set([255, Math.round(190 - 150 * a), Math.round(40 - 20 * a), Math.round(155 * Math.sqrt(a))], 4 * i);
  }
  small.getContext("2d").putImageData(img, 0, 0);
  const c = $("heat"), x = $("xray");
  c.width = x.width; c.height = x.height;
  const g = c.getContext("2d");
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = "high";
  g.drawImage(small, 0, 0, c.width, c.height);
}

$("pick").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) analyze(f);
  e.target.value = "";
});
$("about-link").addEventListener("click", (e) => { e.preventDefault(); $("about").showModal(); });
$("status").addEventListener("click", () => $("about").showModal());

show("empty");
init();
