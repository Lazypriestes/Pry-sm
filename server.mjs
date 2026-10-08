// Local server for the study: serves the static files and forwards readings and image generation
// to a backend. Usage (from the prototype folder):  node server.mjs   → http://localhost:8001
//
// BACKEND=local (default, free, nothing leaves this Mac)
//   readings:  Ollama vision model (VLM_LOCAL, default gemma3:4b)       http://127.0.0.1:11434
//   images:    tools/gen_server.py (SDXL-Turbo on the Mac's GPU)         http://127.0.0.1:8002
// BACKEND=fal (paid, FAL_KEY in .env): Gemini Flash readings, FLUX images
//
// No dependencies. Dotfiles (.env, .git) and tools/ are never served.
import { createServer } from "node:http";
import { readFile, stat, writeFile, mkdir } from "node:fs/promises";
import { join, extname, normalize, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 8001);
const BACKEND = (process.env.BACKEND || "local").toLowerCase();

// Local backend
const OLLAMA = process.env.OLLAMA_URL || "http://127.0.0.1:11434";
const VLM_LOCAL = process.env.VLM_LOCAL || "gemma3:4b";
const GEN_LOCAL = process.env.GEN_LOCAL_URL || "http://127.0.0.1:8002";
// Real-image matching (tools/match_server.py): photos containing a region shaped like the circled part.
const MATCH_LOCAL = process.env.MATCH_URL || "http://127.0.0.1:8003";

// fal backend. Image-to-image: FLUX-dev (~2 s). Outpainting with a mask: flux-lora-fill (~7 s;
// "fal-ai/flux-pro/v1/fill" is a little richer but ~15 s). Readings: Gemini Flash (~1.5 s).
const GEN_MODEL = process.env.GEN_MODEL || "fal-ai/flux/dev/image-to-image";
const GEN_STEPS = Number(process.env.GEN_STEPS || 14);
const FILL_MODEL = process.env.FILL_MODEL || "fal-ai/flux-lora-fill";
const VLM_MODEL = process.env.VLM_MODEL || "google/gemini-2.5-flash";

// The reading: what much larger thing could this fragment be a small part of?
const INTERPRET_SYSTEM = "You name what image fragments resemble. Reply with exactly three short noun phrases separated by commas. No other words, no explanation, no punctuation at the end.";
const INTERPRET_PROMPT = `Like seeing shapes in clouds, read this fragment by its shape, lines and structure, not its material or colour.
What much larger thing could it be a small part of? First something built or architectural, then something from nature, then something surprising.`;

let FAL_KEY = process.env.FAL_KEY;
try {
  for (const line of (await readFile(join(ROOT, ".env"), "utf8")).split("\n")) {
    const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.*)\s*$/);
    if (m && m[1] === "FAL_KEY") FAL_KEY = m[2];
  }
} catch (e) { /* no .env */ }

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".css": "text/css", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml", ".md": "text/plain" };

async function readBody(req, limit = 16e6) {
  let size = 0; const chunks = [];
  for await (const c of req) { size += c.length; if (size > limit) throw new Error("too large"); chunks.push(c); }
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}
function send(res, code, obj) {
  res.writeHead(code, { "Content-Type": "application/json", "Cache-Control": "no-store" });
  res.end(JSON.stringify(obj));
}
const isDataImage = s => typeof s === "string" && s.startsWith("data:image/");

// Robust to preambles ("Here are three…"), numbering and markdown.
function parseWords(text) {
  const line = text.trim().split("\n").map(l => l.trim()).filter(l => l.includes(",")).pop() || text;
  return line.split(",").map(w => w.replace(/[*_#".]/g, "").replace(/^\s*(\d+[.)]|-)\s*/, "").trim().toLowerCase())
    .filter(w => w && w.split(" ").length <= 4).slice(0, 3);
}

async function localGenReady() {
  try { const r = await fetch(`${GEN_LOCAL}/status`, { signal: AbortSignal.timeout(1500) }); return (await r.json()).ready === true; }
  catch (e) { return false; }
}

async function generate(req, res) {
  let b;
  try { b = await readBody(req); } catch (e) { return send(res, 400, { error: "bad request" }); }
  if (!isDataImage(b.image) || typeof b.prompt !== "string") return send(res, 400, { error: "image (data URL) and prompt required" });
  const fill = isDataImage(b.mask);
  const t = performance.now();
  try {
    if (BACKEND === "local") {
      const r = await fetch(`${GEN_LOCAL}/${fill ? "fill" : "img2img"}`, {
        method: "POST", signal: AbortSignal.timeout(120000), headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: b.image, mask: b.mask, prompt: b.prompt, strength: b.strength, seed: b.seed,
                               size: b.size, steps: b.steps, ref: b.ref })
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !j.image) return send(res, 502, { error: j.error || `local generator ${r.status}` });
      return send(res, 200, { url: j.image, model: j.model, mode: j.mode, seed: b.seed, ms: Math.round(performance.now() - t) });
    }
    if (!FAL_KEY) return send(res, 503, { error: "no FAL_KEY in .env" });
    const seed = Number.isInteger(b.seed) ? { seed: b.seed } : {};
    const model = fill ? FILL_MODEL : GEN_MODEL;
    const body = fill
      ? { image_url: b.image, mask_url: b.mask, prompt: b.prompt.slice(0, 500), num_images: 1, output_format: "jpeg", ...seed }
      : { image_url: b.image, prompt: b.prompt.slice(0, 500), strength: Math.min(0.95, Math.max(0.2, Number(b.strength) || 0.75)),
          num_inference_steps: GEN_STEPS, num_images: 1, enable_safety_checker: true, ...seed };
    const r = await fetch(`https://fal.run/${model}`, { method: "POST",
      headers: { Authorization: `Key ${FAL_KEY}`, "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({}));
    const url = j.images && j.images[0] && j.images[0].url;
    if (!r.ok || !url) return send(res, 502, { error: (j.detail && JSON.stringify(j.detail)) || `fal ${r.status}` });
    send(res, 200, { url, model, mode: fill ? "fill" : "img2img", seed: j.seed, ms: Math.round(performance.now() - t) });
  } catch (e) {
    send(res, 502, { error: String(e.message || e) });
  }
}

async function interpret(req, res) {
  let b;
  try { b = await readBody(req); } catch (e) { return send(res, 400, { error: "bad request" }); }
  if (!isDataImage(b.image)) return send(res, 400, { error: "image (data URL) required" });
  const t = performance.now();
  try {
    let raw, model;
    if (BACKEND === "local") {
      const r = await fetch(`${OLLAMA}/api/generate`, {
        method: "POST", signal: AbortSignal.timeout(60000),
        body: JSON.stringify({ model: VLM_LOCAL, system: INTERPRET_SYSTEM, prompt: INTERPRET_PROMPT,
          images: [b.image.split(",", 2)[1]], stream: false, keep_alive: "30m", options: { temperature: 0.8, num_predict: 30 } })
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || typeof j.response !== "string") return send(res, 502, { error: j.error || `ollama ${r.status}` });
      raw = j.response; model = VLM_LOCAL;
    } else {
      if (!FAL_KEY) return send(res, 503, { error: "no FAL_KEY in .env" });
      const r = await fetch("https://fal.run/openrouter/router/vision", {
        method: "POST", signal: AbortSignal.timeout(30000),
        headers: { Authorization: `Key ${FAL_KEY}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: VLM_MODEL, system_prompt: INTERPRET_SYSTEM, prompt: INTERPRET_PROMPT, image_urls: [b.image] })
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || typeof j.output !== "string") return send(res, 502, { error: (j.detail && JSON.stringify(j.detail)) || `fal ${r.status}` });
      raw = j.output; model = VLM_MODEL;
    }
    send(res, 200, { words: parseWords(raw), raw, model, ms: Math.round(performance.now() - t) });
  } catch (e) {
    send(res, 502, { error: String(e.message || e) });
  }
}

async function matchProxy(req, res) {
  let b;
  try { b = await readBody(req); } catch (e) { return send(res, 400, { error: "bad request" }); }
  if (!isDataImage(b.image)) return send(res, 400, { error: "image (data URL) required" });
  try {
    const r = await fetch(`${MATCH_LOCAL}/match`, { method: "POST", signal: AbortSignal.timeout(20000),
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: b.image, k: b.k || 6 }) });
    const j = await r.json().catch(() => ({}));
    send(res, r.status, j);
  } catch (e) { send(res, 502, { error: String(e.message || e) }); }
}
async function photoProxy(id, res) {
  if (!/^[\w-]+$/.test(id)) { res.writeHead(404); return res.end("not found"); }
  try {
    const r = await fetch(`${MATCH_LOCAL}/photo/${id}.jpg`, { signal: AbortSignal.timeout(10000) });
    if (!r.ok) throw new Error(r.status);
    res.writeHead(200, { "Content-Type": "image/jpeg", "Cache-Control": "max-age=86400" });
    res.end(Buffer.from(await r.arrayBuffer()));
  } catch (e) { res.writeHead(404); res.end("not found"); }
}
// Per-vase words, reference images and weights for "imagined" mode (edited in imagine.html).
async function saveImagine(req, res) {
  let b;
  try { b = await readBody(req, 2e6); } catch (e) { return send(res, 400, { error: "bad request" }); }
  if (!b || typeof b !== "object" || Array.isArray(b)) return send(res, 400, { error: "expected an object" });
  await writeFile(join(ROOT, "imagine.json"), JSON.stringify(b, null, 2));
  send(res, 200, { ok: true });
}
async function saveRef(req, res) {
  let b;
  try { b = await readBody(req); } catch (e) { return send(res, 400, { error: "bad request" }); }
  if (!isDataImage(b.image) || !/^[\w-]+$/.test(b.vase || "")) return send(res, 400, { error: "vase and image required" });
  const ext = b.image.startsWith("data:image/png") ? "png" : "jpg";
  await mkdir(join(ROOT, "imagine_refs"), { recursive: true });
  const path = `imagine_refs/${b.vase}_${Date.now()}.${ext}`;
  await writeFile(join(ROOT, path), Buffer.from(b.image.split(",", 2)[1], "base64"));
  send(res, 200, { path });
}

async function matchReady() {
  try { const r = await fetch(`${MATCH_LOCAL}/status`, { signal: AbortSignal.timeout(1500) }); const j = await r.json(); return j.ready ? j : false; }
  catch (e) { return false; }
}

createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  if (url.pathname === "/api/match" && req.method === "POST") return matchProxy(req, res);
  if (url.pathname === "/api/imagine" && req.method === "POST") return saveImagine(req, res);
  if (url.pathname === "/api/imagine/ref" && req.method === "POST") return saveRef(req, res);
  const photo = url.pathname.match(/^\/api\/photo\/([\w-]+)\.jpg$/);
  if (photo) return photoProxy(photo[1], res);
  if (url.pathname === "/api/interpret" && req.method === "POST") return interpret(req, res);
  if (url.pathname === "/api/generate" && req.method === "POST") return generate(req, res);
  if (url.pathname === "/api/status") {
    const generation = BACKEND === "local" ? await localGenReady() : !!FAL_KEY;
    const m = await matchReady();
    return send(res, 200, { backend: BACKEND, generation, matching: !!m, match_photos: m ? m.photos : 0,
      model: BACKEND === "local" ? "sdxl-turbo (local)" : GEN_MODEL,
      fill_model: BACKEND === "local" ? "sdxl-turbo inpaint (local)" : FILL_MODEL,
      vlm_model: BACKEND === "local" ? VLM_LOCAL : VLM_MODEL });
  }
  // Static files, inside ROOT only, no dotfiles, no tools/.
  let path = decodeURIComponent(url.pathname);
  if (path.endsWith("/")) path += "index.html";
  const file = normalize(join(ROOT, path));
  if (!file.startsWith(ROOT) || /\/\.|^\/tools\//.test(path)) { res.writeHead(404); return res.end("not found"); }
  try {
    if (!(await stat(file)).isFile()) throw new Error();
    res.writeHead(200, { "Content-Type": TYPES[extname(file).toLowerCase()] || "application/octet-stream" });
    res.end(await readFile(file));
  } catch (e) { res.writeHead(404); res.end("not found"); }
}).listen(PORT, () => {
  console.log(`study on http://localhost:${PORT}  ·  backend: ${BACKEND}` +
    (BACKEND === "local" ? `  (readings ${VLM_LOCAL} via Ollama, images ${GEN_LOCAL})` : ""));
});
