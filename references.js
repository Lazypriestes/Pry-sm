/* ============================================================
   VISUAL REFERENCES (CLIP + live search)
   Loaded on demand by the "Circle + references" end part.
   1. CLIP (transformers.js, in the browser) embeds the circled part twice:
      as it is, and abstracted (greyscale, contrast-stretched) so shape and
      tone count more than colour and material.
   2. CLIP names what the part reads like, from FORM_WORDS (e.g. "smoke").
   3. Openverse is searched live for those words (CC0 / public domain only).
   4. Every result is embedded in the browser (abstracted the same way) and
      ranked by how much its shape and tone look like the circled part.
   If the live search fails, the offline Met library (references/index.json)
   is used instead, when present.
   ============================================================ */
const TF_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const MODEL = "Xenova/clip-vit-base-patch32";
const DTYPE = "q8";
const OPENVERSE = "https://api.openverse.org/v1/images/";
const PER_WORD = 8;           // results fetched per word
const MAX_PER_WORD = 3;       // variety: at most this many shown results from one word

// What a part of an object can read as. Edit freely.
export const FORM_WORDS = [
  "smoke", "swirl", "whirlpool", "spiral", "flame", "ocean wave", "water ripples", "clouds", "fog",
  "ink in water", "marble", "wood grain", "tree bark", "roots", "bare branches", "leaf veins",
  "feather", "fern", "flower petals", "seashell", "fish scales", "honeycomb", "cracked earth",
  "dripping liquid", "crystals", "sand dunes", "rock strata", "coral", "lichen", "bubbles", "frost",
  "galaxy", "aurora", "lava", "stripes", "moss", "rust", "ice", "spider web", "seed pods", "pine cone",
  "waterfall", "river delta", "canyon", "mountain ridge", "storm clouds", "lightning", "moon surface",
  "pebbles", "jellyfish", "tentacles", "butterfly wing", "tiger fur", "leopard spots", "honey",
  "amber", "glass", "silk fabric", "knot", "rope", "basket weave", "embroidery", "brushstroke",
  "ink wash painting", "mosaic", "stained glass", "arch", "dome", "spiral staircase", "fingerprint",
  "topographic map", "tree rings", "eye", "skin", "hair", "bone", "flower bud", "mushroom gills",
  // structures and built forms, so geometric parts can read as more than a texture
  "door", "doorway", "window", "picture frame", "gate", "column", "pillar", "panel", "shutters",
  "tiles", "grid", "brick wall", "corridor", "ladder", "fence", "bridge", "tower", "staircase",
  "temple facade", "room", "mirror", "book spine", "maze", "map", "screen", "curtain"
];

let ready = null;
let S = null;                 // { tf, processor, vision, tokenizer, text, words, met }
const searchCache = new Map();

function unit(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function decode(b64) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
function zscores(a) {
  const m = a.reduce((x, y) => x + y, 0) / a.length;
  const sd = Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length) || 1;
  return a.map(x => (x - m) / sd);
}

// Greyscale + 2–98% contrast stretch: keeps shape and tone, drops colour and material.
export function abstractCanvas(src) {
  const cv = document.createElement("canvas"); cv.width = src.width; cv.height = src.height;
  const ctx = cv.getContext("2d");
  ctx.drawImage(src, 0, 0);
  const id = ctx.getImageData(0, 0, cv.width, cv.height), d = id.data, n = d.length / 4;
  const g = new Float32Array(n);
  for (let i = 0; i < n; i++) g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
  const sorted = Float32Array.from(g).sort();
  const lo = sorted[Math.floor(n * 0.02)], hi = sorted[Math.floor(n * 0.98)], span = Math.max(1, hi - lo);
  for (let i = 0; i < n; i++) { const v = Math.max(0, Math.min(255, (g[i] - lo) / span * 255)); d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; }
  ctx.putImageData(id, 0, 0);
  return cv;
}

// onProgress(fraction 0..1) while the model downloads (first time only; then cached by the browser).
export function loadReferences(onProgress = () => {}) {
  if (ready) return ready;
  ready = (async () => {
    const tf = await import(TF_URL);
    tf.env.allowLocalModels = false;
    const files = {};
    const progress_callback = p => {
      if (p.status === "progress" && p.total) {
        files[p.file] = [p.loaded, p.total];
        const v = Object.values(files), done = v.reduce((a, f) => a + f[0], 0), all = v.reduce((a, f) => a + f[1], 0);
        onProgress(all ? done / all : 0);
      }
    };
    const [processor, vision, tokenizer, text] = await Promise.all([
      tf.AutoProcessor.from_pretrained(MODEL, { progress_callback }),
      tf.CLIPVisionModelWithProjection.from_pretrained(MODEL, { dtype: DTYPE, progress_callback }),
      tf.AutoTokenizer.from_pretrained(MODEL, { progress_callback }),
      tf.CLIPTextModelWithProjection.from_pretrained(MODEL, { dtype: DTYPE, progress_callback })
    ]);
    S = { tf, processor, vision, tokenizer, text };
    S.words = [];
    for (const w of FORM_WORDS) S.words.push({ word: w, vec: await embedText(`a photo of ${w}`) });
    // Offline fallback library (optional).
    try {
      const index = await (await fetch("references/index.json")).json();
      S.met = { items: index.items.map(({ emb, ...rest }) => ({ ...rest, image: "references/" + rest.file, source: "The Met (CC0)" })),
                vecs: index.items.map(it => decode(it.emb)) };
    } catch (e) { S.met = null; }
    onProgress(1);
    return { model: MODEL, dtype: DTYPE, words: FORM_WORDS.length, offline_fallback: !!S.met };
  })();
  return ready;
}

async function toRaw(canvas) {
  const blob = await new Promise(r => canvas.toBlob(r, "image/png"));
  return S.tf.RawImage.fromBlob(blob);
}
async function embedImageLike(src) {
  return (await embedBatch([src]))[0];
}
// Several canvases through CLIP in one pass (much faster than one at a time).
async function embedBatch(canvases) {
  const raws = await Promise.all(canvases.map(toRaw));
  const { image_embeds } = await S.vision(await S.processor(raws));
  const d = image_embeds.dims[1], out = [];
  for (let i = 0; i < canvases.length; i++) out.push(unit(Float32Array.from(image_embeds.data.slice(i * d, (i + 1) * d))));
  return out;
}
function toCanvas(img, size = 224) {
  const cv = document.createElement("canvas"); cv.width = cv.height = size;
  const ctx = cv.getContext("2d"), k = Math.max(size / img.naturalWidth, size / img.naturalHeight);
  const w = img.naturalWidth * k, h = img.naturalHeight * k;
  ctx.drawImage(img, (size - w) / 2, (size - h) / 2, w, h);      // centre-crop, like CLIP's own preprocessing
  return cv;
}

export async function embedText(str) {
  const inputs = S.tokenizer([str], { padding: true, truncation: true });
  const { text_embeds } = await S.text(inputs);
  return unit(Float32Array.from(text_embeds.data));
}

// The circled part → { colour, shape } embeddings.
export async function embedPart(canvas) {
  await ready;
  return { colour: await embedImageLike(canvas), shape: await embedImageLike(abstractCanvas(canvas)) };
}

// Top-k form words for one embedding (for inspection / tuning).
export function topWords(vec, k = 6) {
  return S.words.map(w => [w.word, +dot(vec, w.vec).toFixed(4)]).sort((a, b) => b[1] - a[1]).slice(0, k);
}

// What the part reads like: 2 words from its shape, 1 from its colour/material, without repeats.
export function readsLike(part, typed = null) {
  const top = vec => S.words.map(w => [w.word, dot(vec, w.vec)]).sort((a, b) => b[1] - a[1]).map(x => x[0]);
  const shape = top(part.shape), colour = top(part.colour);
  // Shape leads (2 words), colour/material adds one, unless it repeats.
  const out = shape.slice(0, 2);
  for (const w of colour) { if (!out.includes(w)) { out.push(w); break; } }
  if (typed) out.unshift(typed);              // the participant's own words lead the search
  return out;
}

function loadImage(url) {
  return new Promise((res, rej) => {
    const im = new Image(); im.crossOrigin = "anonymous";
    im.onload = () => res(im); im.onerror = () => rej(new Error("image failed: " + url));
    im.src = url;
  });
}

// Live search for one word: Openverse results, each embedded once (cached for the session).
async function searchWord(word) {
  if (searchCache.has(word)) return searchCache.get(word);
  const p = (async () => {
    const url = `${OPENVERSE}?q=${encodeURIComponent(word)}&license=cc0,pdm&page_size=${PER_WORD}&mature=false`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Openverse ${r.status}`);
    const { results = [] } = await r.json();
    const loaded = (await Promise.all(results.map(res =>
      loadImage(res.thumbnail).then(im => ({ res, cv: toCanvas(im) }), () => null)))).filter(Boolean);
    if (!loaded.length) return [];
    // One CLIP pass for all images of this word, abstracted like the circled part.
    const embs = await embedBatch(loaded.map(l => abstractCanvas(l.cv)));
    return loaded.map(({ res }, i) => ({
      id: "ov_" + res.id, word, image: res.thumbnail, full: res.url, title: res.title || word,
      creator: res.creator || "", license: res.license, landing: res.foreign_landing_url || "",
      source: "Openverse", shape: embs[i]
    }));
  })();
  searchCache.set(word, p);
  p.catch(() => searchCache.delete(word));
  return p;
}

// Rank candidates by shape/tone likeness to the circled part, dropping near-duplicates.
function rankCandidates(part, cands, k) {
  if (!cands.length) return [];
  const ss = cands.map(c => dot(part.shape, c.shape));
  const score = zscores(ss);
  const order = score.map((s, i) => i).sort((a, b) => score[b] - score[a]);
  const picked = [], perWord = {};
  for (const i of order) {
    if (picked.length >= k) break;
    const w = cands[i].word;
    if ((perWord[w] || 0) >= MAX_PER_WORD) continue;
    if (picked.some(j => dot(cands[i].shape, cands[j].shape) > 0.95)) continue;
    picked.push(i); perWord[w] = (perWord[w] || 0) + 1;
  }
  return picked.map(i => {
    const { shape, ...c } = cands[i];
    return { ...c, score: +score[i].toFixed(3), shape_sim: +ss[i].toFixed(4) };
  });
}

// Main entry: words + ranked references for a circled part.
// onPartial(results) is called as each word's search finishes, so results appear progressively.
export async function findReferences(part, { typed = null, k = 8, onWords = () => {}, onPartial = () => {} } = {}) {
  await ready;
  const words = readsLike(part, typed);
  onWords(words);                                   // the words are known right away; photos follow
  const all = [];
  let failures = 0;
  await Promise.all(words.map(async w => {
    try { all.push(...await searchWord(w)); onPartial(rankCandidates(part, all, k), words); }
    catch (e) { failures++; }
  }));
  if (!all.length && S.met) {                       // offline fallback: the Met library
    const cands = S.met.items.map((it, i) => ({ ...it, word: it.theme, shape: S.met.vecs[i] }));
    return { words, results: rankCandidates(part, cands, k), offline: true };
  }
  return { words, results: rankCandidates(part, all, k), offline: false, failed_words: failures };
}
