// Local server for the study: serves the static files and forwards image generation to fal.ai,
// so the API key (in .env, never sent to the browser) stays on this machine.
// Usage (from the prototype folder):  node server.mjs        → http://localhost:8001
// No dependencies. Dotfiles (.env, .git) and tools/ are never served.
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { join, extname, normalize, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 8001);
// Image-to-image model. FLUX-dev gave the best "realistic but still the circled shape" results at
// ~2 s per image (SDXL-Lightning was as fast but mostly repainted the object). Any fal
// image-to-image endpoint taking image_url / prompt / strength works.
const GEN_MODEL = process.env.GEN_MODEL || "fal-ai/flux/dev/image-to-image";
const GEN_STEPS = Number(process.env.GEN_STEPS || 14);
// Outpainting model, used when the browser sends a mask: keeps the masked-out (black) part and
// invents everything else, so the circled part becomes one small piece of a larger scene.
// flux-lora-fill (FLUX-dev) ≈ 7 s; "fal-ai/flux-pro/v1/fill" is a little richer but ≈ 15 s.
const FILL_MODEL = process.env.FILL_MODEL || "fal-ai/flux-lora-fill";
// Interpretation: a vision-language model reads what a fragment could be part of, with an open
// vocabulary (CLIP could only pick from a fixed word list). ~1.5 s, a fraction of a cent.
const VLM_MODEL = process.env.VLM_MODEL || "google/gemini-2.5-flash";
const INTERPRET_PROMPT = `A small fragment cropped from an object. Like seeing shapes in clouds, read it by its shape, lines and structure, not its material or colour.
What much larger thing could this be a small part of? Give three answers: one built or architectural, one from nature, one surprising.
Each 1-3 words, comma-separated, nothing else.`;

let FAL_KEY = process.env.FAL_KEY;
try {
  for (const line of (await readFile(join(ROOT, ".env"), "utf8")).split("\n")) {
    const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.*)\s*$/);
    if (m && m[1] === "FAL_KEY") FAL_KEY = m[2];
  }
} catch (e) { /* no .env: generation disabled */ }

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

async function generate(req, res) {
  if (!FAL_KEY) return send(res, 503, { error: "no FAL_KEY in .env" });
  let b;
  try { b = await readBody(req); } catch (e) { return send(res, 400, { error: "bad request" }); }
  if (typeof b.image !== "string" || !b.image.startsWith("data:image/") || typeof b.prompt !== "string")
    return send(res, 400, { error: "image (data URL) and prompt required" });
  const fill = typeof b.mask === "string" && b.mask.startsWith("data:image/");
  const model = fill ? FILL_MODEL : GEN_MODEL;
  const seed = Number.isInteger(b.seed) ? { seed: b.seed } : {};
  const body = fill
    ? { image_url: b.image, mask_url: b.mask, prompt: b.prompt.slice(0, 500), num_images: 1,
        output_format: "jpeg", safety_tolerance: "2", ...seed }
    : { image_url: b.image, prompt: b.prompt.slice(0, 500),
        strength: Math.min(0.95, Math.max(0.2, Number(b.strength) || 0.75)),
        num_inference_steps: GEN_STEPS, num_images: 1, enable_safety_checker: true, ...seed };
  const t = performance.now();
  try {
    const r = await fetch(`https://fal.run/${model}`, {
      method: "POST",
      headers: { Authorization: `Key ${FAL_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const j = await r.json().catch(() => ({}));
    const url = j.images && j.images[0] && j.images[0].url;
    if (!r.ok || !url) return send(res, 502, { error: (j.detail && JSON.stringify(j.detail)) || `fal ${r.status}` });
    send(res, 200, { url, model, mode: fill ? "fill" : "img2img", seed: j.seed, strength: body.strength,
                     ms: Math.round(performance.now() - t) });
  } catch (e) {
    send(res, 502, { error: String(e.message || e) });
  }
}

async function interpret(req, res) {
  if (!FAL_KEY) return send(res, 503, { error: "no FAL_KEY in .env" });
  let b;
  try { b = await readBody(req); } catch (e) { return send(res, 400, { error: "bad request" }); }
  if (typeof b.image !== "string" || !b.image.startsWith("data:image/")) return send(res, 400, { error: "image (data URL) required" });
  const t = performance.now();
  try {
    const r = await fetch("https://fal.run/openrouter/router/vision", {
      method: "POST", signal: AbortSignal.timeout(30000),
      headers: { Authorization: `Key ${FAL_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: VLM_MODEL, prompt: INTERPRET_PROMPT, image_urls: [b.image] })
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok || typeof j.output !== "string") return send(res, 502, { error: (j.detail && JSON.stringify(j.detail)) || `fal ${r.status}` });
    const words = j.output.split(/,|\n/).map(w => w.trim().replace(/^[-*\d.\s]+/, "").replace(/[."]+$/g, "").toLowerCase())
      .filter(Boolean).slice(0, 3);
    send(res, 200, { words, raw: j.output, model: VLM_MODEL, ms: Math.round(performance.now() - t) });
  } catch (e) {
    send(res, 502, { error: String(e.message || e) });
  }
}

createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  if (url.pathname === "/api/interpret" && req.method === "POST") return interpret(req, res);
  if (url.pathname === "/api/generate" && req.method === "POST") return generate(req, res);
  if (url.pathname === "/api/status") return send(res, 200, { generation: !!FAL_KEY, model: GEN_MODEL, fill_model: FILL_MODEL, vlm_model: VLM_MODEL });
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
  console.log(`study on http://localhost:${PORT}  ·  generation ${FAL_KEY ? "on (" + GEN_MODEL + ")" : "off (no FAL_KEY)"}`);
});
