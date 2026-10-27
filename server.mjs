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
import { createHash } from "node:crypto";
import { homedir } from "node:os";
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
// Objects in a room photo (tools/room_server.py, Florence-2) and their histories (room.html).
const ROOM_LOCAL = process.env.ROOM_URL || "http://127.0.0.1:8004";
const LLM_LOCAL = process.env.LLM_LOCAL || "llama3.1:8b";
const SCAN_LOCAL = process.env.SCAN_LOCAL || "qwen2.5vl:7b";      // looks at each object (gemma3:4b missed too much)
const UA = { "User-Agent": "objview-research/0.1 (study prototype)" };

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
// Google Cloud Vision (Lens-style "web detection"): an API key from console.cloud.google.com, saved in .env
let GOOGLE_VISION_KEY = process.env.GOOGLE_VISION_KEY;
try {
  for (const line of (await readFile(join(ROOT, ".env"), "utf8")).split("\n")) {
    const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.*)\s*$/);
    if (m && m[1] === "FAL_KEY") FAL_KEY = m[2];
    if (m && m[1] === "GOOGLE_VISION_KEY") GOOGLE_VISION_KEY = m[2];
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
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: b.image, k: b.k || 6, library: b.library, context: b.context, exclude: b.exclude, vessels: b.vessels }) });
    const j = await r.json().catch(() => ({}));
    send(res, r.status, j);
  } catch (e) { send(res, 502, { error: String(e.message || e) }); }
}
// Photos whose overall look (CLIP) fits a description and/or an image: colour, surface and mood references.
async function similarProxy(req, res) {
  let b;
  try { b = await readBody(req); } catch (e) { return send(res, 400, { error: "bad request" }); }
  try {
    const r = await fetch(`${MATCH_LOCAL}/similar`, { method: "POST", signal: AbortSignal.timeout(30000), headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: b.text, image: isDataImage(b.image) ? b.image : undefined, minus: b.minus, library: b.library, k: b.k, embed: b.embed }) });
    send(res, r.status, await r.json().catch(() => ({})));
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

// ---- Room: objects and how they came to be -------------------------------------------------------
async function objectsProxy(req, res) {
  let b;
  try { b = await readBody(req, 30e6); } catch (e) { return send(res, 400, { error: "bad request" }); }
  if (!isDataImage(b.image)) return send(res, 400, { error: "image (data URL) required" });
  try {
    const r = await fetch(`${ROOM_LOCAL}/objects`, { method: "POST", signal: AbortSignal.timeout(400000),
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: b.image }) });
    send(res, r.status, await r.json().catch(() => ({ error: `room server ${r.status}` })));
  } catch (e) { send(res, 502, { error: String(e.message || e) }); }
}

const LINEAGE_SYSTEM = `You are a design historian. For an everyday object, you trace how it came to be and what else shares its form.
Answer only with JSON:
{"name": short name,
 "lineage": 5 ancestors, oldest first, ending with the modern version: [{"year": "c. 500 BC" or "1882", "name": the specific historical object or invention, "place": where, "search": 2-4 words naming the object itself, as it would be titled in a photo archive (never a museum or person name)}]. Only well-known, real objects; if unsure of a date, use a century,
 "cousins": 4 things from other worlds with the same form or working principle: [{"name", "why": under 10 words, "search": 2-4 words naming that thing}]. Think of shape first: blades, rings, cylinders, legs, frames,
 "story": 4 sentences, second person. The reader glanced past this object and thinks it is beneath their notice. Start with what they assume ("You think this is just..."). Then, coolly and precisely, trace how it trickled down from the earliest ancestor through specific inventors, workshops, cities and years, to the one in their room. End by pointing out that they did not choose it; it was chosen for them by all of that. Tone: dry, exacting, faintly amused, never warm. No exclamation marks, no "let me tell you".}`;
const LINEAGE_FILE = join(ROOT, "rooms", "lineage.json");
let lineageCache = null;
const lineagePending = new Map();
async function loadLineage() {
  if (!lineageCache) { try { lineageCache = JSON.parse(await readFile(LINEAGE_FILE, "utf8")); } catch (e) { lineageCache = {}; } }
  return lineageCache;
}
async function getJSON(url, ms = 10000) {
  try { const r = await fetch(url, { headers: UA, signal: AbortSignal.timeout(ms) }); return r.ok ? await r.json() : null; } catch (e) { return null; }
}
// Candidate pictures: the lead image of the matching Wikipedia article(s), plus Wikimedia Commons files.
async function candidates(item) {
  const q = encodeURIComponent;
  const [wp, cm] = await Promise.all([
    getJSON(`https://en.wikipedia.org/w/api.php?action=query&format=json&origin=*&generator=search&gsrlimit=2&gsrsearch=${q(item.name)}` +
            `&prop=pageimages|info&inprop=url&piprop=thumbnail&pithumbsize=480`),
    getJSON(`https://commons.wikimedia.org/w/api.php?action=query&format=json&origin=*&generator=search&gsrnamespace=6&gsrlimit=7` +
            `&gsrsearch=${q("filetype:bitmap " + (item.search || item.name))}&prop=imageinfo&iiprop=url|extmetadata` +
            `&iiextmetadatafilter=Artist|LicenseShortName&iiurlwidth=480`)]);
  const out = [];
  for (const p of Object.values(wp?.query?.pages || {}))
    if (p.thumbnail?.source) out.push({ src: p.thumbnail.source, link: p.fullurl, credit: `Wikipedia: ${p.title}` });
  for (const p of Object.values(cm?.query?.pages || {})) {
    const ii = p.imageinfo?.[0]; if (!ii?.thumburl) continue;
    const m = ii.extmetadata || {};
    const artist = String(m.Artist?.value || "").replace(/<[^>]+>/g, "").trim().slice(0, 60);
    out.push({ src: ii.thumburl, link: ii.descriptionurl, credit: [artist, m.LicenseShortName?.value].filter(Boolean).join(" · ") || "Wikimedia Commons" });
  }
  return out;
}
async function pictureFor(item) {
  const c = await candidates(item);
  if (!c.length) return null;
  try {
    const r = await fetch(`${MATCH_LOCAL}/rank`, { method: "POST", signal: AbortSignal.timeout(60000),
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: item.name, images: c.map(x => x.src) }) });
    const scores = (await r.json()).scores || [];
    // Up to two pictures per step, the ones CLIP thinks show it best.
    const good = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).filter(x => x[0] > 0.2).slice(0, 2);
    return good.length ? good.map(([s, i]) => ({ ...c[i], score: s })) : null;
  } catch (e) { return [c[0]]; }                                 // matcher not running: take the first one
}
async function makeLineage(name, desc) {
  const r = await fetch(`${OLLAMA}/api/generate`, { method: "POST", signal: AbortSignal.timeout(180000),
    body: JSON.stringify({ model: LLM_LOCAL, system: LINEAGE_SYSTEM, prompt: `The object: ${desc && desc !== name ? `${name} (${desc})` : name}`,
      stream: false, format: "json", keep_alive: "30m", options: { temperature: 0.6, num_predict: 900 } }) });
  const j = await r.json();
  const L = JSON.parse(j.response);
  const clean = (a, n) => (Array.isArray(a) ? a : []).filter(x => x && x.name).slice(0, n);
  L.lineage = clean(L.lineage, 6); L.cousins = clean(L.cousins, 5);
  // Three at a time: Wikimedia throttles bursts of image downloads.
  const items = [...L.lineage, ...L.cousins];
  for (let i = 0; i < items.length; i += 3)
    await Promise.all(items.slice(i, i + 3).map(async it => { it.images = await pictureFor(it); }));
  return { name: L.name || name, desc, lineage: L.lineage, cousins: L.cousins, story: String(L.story || ""), model: LLM_LOCAL, made: new Date().toISOString() };
}
// Cached in rooms/lineage.json by object name; edit an entry and set "checked": true to keep it.
async function lineage(req, res) {
  let b;
  try { b = await readBody(req); } catch (e) { return send(res, 400, { error: "bad request" }); }
  const name = String(b.name || "").trim().toLowerCase().slice(0, 80);
  if (!name) return send(res, 400, { error: "name required" });
  const cache = await loadLineage();
  if (cache[name] && !b.refresh) return send(res, 200, { ...cache[name], cached: true });
  if (cache[name]?.checked) return send(res, 200, { ...cache[name], cached: true });
  try {
    if (!lineagePending.has(name)) lineagePending.set(name, makeLineage(name, String(b.desc || "").slice(0, 160)).finally(() => lineagePending.delete(name)));
    const L = await lineagePending.get(name);
    cache[name] = L;
    await mkdir(dirname(LINEAGE_FILE), { recursive: true });
    await writeFile(LINEAGE_FILE, JSON.stringify(cache, null, 1));
    send(res, 200, L);
  } catch (e) { send(res, 502, { error: String(e.message || e) }); }
}

// A careful look at one object: form, surface, decoration, palette, material, what is certain and what is not.
const SCAN_PROMPT = `You are a careful museum cataloguer examining one object, shown cut out of its photograph on a dark background. Describe only what you can see, and say when something cannot be told from a photo.
Answer as JSON:
{"name": what it is, 2-4 words,
 "summary": one sentence: what it is and its overall character,
 "form": 3 short observations about shape and construction (parts like top, body, base, legs, edges),
 "surface": 3 short observations about colour, finish, texture, gloss,
 "decoration": up to 3 short observations describing any motifs (flowers, figures, lettering), patterns or ornament; empty list only if the surface is plain,
 "palette": 4-6 colour names as a designer would name them (e.g. "olive grey", "ivory"),
 "material": the most likely material, hedged if unsure,
 "made": how it was probably made, hedged,
 "resembles": 1-2 styles or traditions it resembles, phrased as comparisons ("consistent with ...", "recalls ..."), never as identifications,
 "interesting": 2 things a careful viewer would notice,
 "clear": 2 things that are certain from the photo,
 "uncertain": 2 things a photo cannot establish}
Each observation under 14 words.`;
const SCAN_FILE = join(ROOT, "rooms", "scans.json");
let scanCache = null;
const scanPending = new Map();
async function scan(req, res) {
  let b;
  try { b = await readBody(req); } catch (e) { return send(res, 400, { error: "bad request" }); }
  const key = String(b.key || "").slice(0, 120);
  if (!key) return send(res, 400, { error: "key required" });
  if (!scanCache) { try { scanCache = JSON.parse(await readFile(SCAN_FILE, "utf8")); } catch (e) { scanCache = {}; } }
  if (scanCache[key] && (!b.refresh || scanCache[key].checked)) return send(res, 200, { ...scanCache[key], cached: true });
  if (!isDataImage(b.image)) return send(res, 400, { error: "image (data URL) required" });
  try {
    if (!scanPending.has(key)) scanPending.set(key, (async () => {
      const hint = b.name ? ` A detector called it "${String(b.name).slice(0, 60)}"${b.desc && b.desc !== b.name ? ` ("${String(b.desc).slice(0, 120)}")` : ""}; correct it if wrong.` : "";
      const r = await fetch(`${OLLAMA}/api/generate`, { method: "POST", signal: AbortSignal.timeout(180000),
        body: JSON.stringify({ model: SCAN_LOCAL, prompt: SCAN_PROMPT + hint, images: [b.image.split(",", 2)[1]], stream: false, format: "json",
          keep_alive: "30m", options: { temperature: 0.3, num_predict: 800 } }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || `ollama ${r.status}`);
      return { ...JSON.parse(j.response), model: SCAN_LOCAL, made_at: new Date().toISOString() };
    })().finally(() => scanPending.delete(key)));
    const out = await scanPending.get(key);
    scanCache[key] = out;
    await mkdir(dirname(SCAN_FILE), { recursive: true });
    await writeFile(SCAN_FILE, JSON.stringify(scanCache, null, 1));
    send(res, 200, out);
  } catch (e) { send(res, 502, { error: String(e.message || e) }); }
}

// ---- Google Lens-style search (Cloud Vision web detection) ----------------------------------------
// What Google thinks the image shows, related topics, visually similar images on the web and pages that
// contain it. Results are cached by image hash outside the project, so the same crop is only sent once.
const LENS_CACHE = join(homedir(), ".local", "share", "objview-match", "lens_cache");
async function lens(req, res) {
  let b;
  try { b = await readBody(req); } catch (e) { return send(res, 400, { error: "bad request" }); }
  if (!isDataImage(b.image)) return send(res, 400, { error: "image (data URL) required" });
  const content = b.image.split(",", 2)[1];
  const key = createHash("sha1").update(content).digest("hex").slice(0, 20), file = join(LENS_CACHE, key + ".json");
  try { return send(res, 200, { ...JSON.parse(await readFile(file, "utf8")), cached: true }); } catch (e) { /* not cached */ }
  if (!GOOGLE_VISION_KEY) return send(res, 503, { error: "no GOOGLE_VISION_KEY in .env" });
  const t = performance.now();
  try {
    const r = await fetch(`https://vision.googleapis.com/v1/images:annotate?key=${encodeURIComponent(GOOGLE_VISION_KEY)}`, {
      method: "POST", signal: AbortSignal.timeout(20000), headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ requests: [{ image: { content }, features: [{ type: "WEB_DETECTION", maxResults: 20 }, { type: "LABEL_DETECTION", maxResults: 10 }],
                                          imageContext: { webDetectionParams: { includeGeoResults: false } } }] }) });
    const j = await r.json().catch(() => ({}));
    const a = j.responses?.[0] || {};
    if (!r.ok || a.error) return send(res, 502, { error: (a.error || j.error || {}).message || `google ${r.status}` });
    const w = a.webDetection || {};
    const out = {
      guess: (w.bestGuessLabels || []).map(l => l.label).filter(Boolean),
      entities: (w.webEntities || []).filter(e => e.description).map(e => ({ name: e.description, score: +(e.score || 0).toFixed(3) })),
      labels: (a.labelAnnotations || []).map(l => ({ name: l.description, score: +(l.score || 0).toFixed(3) })),
      similar: (w.visuallySimilarImages || []).map(i => i.url).filter(Boolean),
      matching: [...(w.fullMatchingImages || []), ...(w.partialMatchingImages || [])].map(i => i.url).filter(Boolean),
      pages: (w.pagesWithMatchingImages || []).map(p => ({ url: p.url, title: String(p.pageTitle || "").replace(/<[^>]+>/g, ""),
              image: (p.fullMatchingImages || p.partialMatchingImages || [])[0]?.url || null })),
      ms: Math.round(performance.now() - t), made: new Date().toISOString()
    };
    await mkdir(LENS_CACHE, { recursive: true });
    await writeFile(file, JSON.stringify(out));
    send(res, 200, out);
  } catch (e) { send(res, 502, { error: String(e.message || e) }); }
}

async function matchReady() {
  try { const r = await fetch(`${MATCH_LOCAL}/status`, { signal: AbortSignal.timeout(1500) }); const j = await r.json(); return j.ready ? j : false; }
  catch (e) { return false; }
}

createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  if (url.pathname === "/api/match" && req.method === "POST") return matchProxy(req, res);
  if (url.pathname === "/api/similar" && req.method === "POST") return similarProxy(req, res);
  if (url.pathname === "/api/imagine" && req.method === "POST") return saveImagine(req, res);
  if (url.pathname === "/api/objects" && req.method === "POST") return objectsProxy(req, res);
  if (url.pathname === "/api/lineage" && req.method === "POST") return lineage(req, res);
  if (url.pathname === "/api/scan" && req.method === "POST") return scan(req, res);
  if (url.pathname === "/api/lens" && req.method === "POST") return lens(req, res);
  if (url.pathname === "/api/imagine/ref" && req.method === "POST") return saveRef(req, res);
  const photo = url.pathname.match(/^\/api\/photo\/([\w-]+)\.jpg$/);
  if (photo) return photoProxy(photo[1], res);
  if (url.pathname === "/api/interpret" && req.method === "POST") return interpret(req, res);
  if (url.pathname === "/api/generate" && req.method === "POST") return generate(req, res);
  if (url.pathname === "/api/status") {
    const generation = BACKEND === "local" ? await localGenReady() : !!FAL_KEY;
    const m = await matchReady();
    let room = false;
    try { room = (await (await fetch(`${ROOM_LOCAL}/status`, { signal: AbortSignal.timeout(1500) })).json()).ready === true; } catch (e) { /* off */ }
    return send(res, 200, { backend: BACKEND, generation, matching: !!m, match_photos: m ? m.photos : 0, room_objects: room, lens: !!GOOGLE_VISION_KEY,
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
    const ext = extname(file).toLowerCase();
    // Pages, scripts and data must always be re-checked, so edits show up on a normal reload.
    const fresh = [".html", ".js", ".mjs", ".json", ".css"].includes(ext);
    res.writeHead(200, { "Content-Type": TYPES[ext] || "application/octet-stream",
                         "Cache-Control": fresh ? "no-cache" : "max-age=3600" });
    res.end(await readFile(file));
  } catch (e) { res.writeHead(404); res.end("not found"); }
}).listen(PORT, () => {
  console.log(`study on http://localhost:${PORT}  ·  backend: ${BACKEND}` +
    (BACKEND === "local" ? `  (readings ${VLM_LOCAL} via Ollama, images ${GEN_LOCAL})` : ""));
});
