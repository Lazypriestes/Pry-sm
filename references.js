/* ============================================================
   VISUAL REFERENCES (CLIP)
   Loaded on demand by the "Circle + references" end part.
   - CLIP (transformers.js, in the browser) embeds the part the participant
     circled, and any words they typed.
   - references/index.json holds precomputed embeddings of the reference
     library (built by tools/build-index.mjs with the same model + dtype).
   - rank() returns the most similar references, mixing in the words if any.
   ============================================================ */
const TF_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const MODEL = "Xenova/clip-vit-base-patch32";
const DTYPE = "q8";
const TEXT_WEIGHT = 0.4;      // share of the ranking from the typed words, when there are any
const MAX_PER_THEME = 2;      // variety: at most this many results from one library theme

let ready = null;
let S = null;                 // { tf, processor, vision, tokenizer, text, items, M, dims }

function unit(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}
function decode(b64) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
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
    const [processor, vision, tokenizer, text, index] = await Promise.all([
      tf.AutoProcessor.from_pretrained(MODEL, { progress_callback }),
      tf.CLIPVisionModelWithProjection.from_pretrained(MODEL, { dtype: DTYPE, progress_callback }),
      tf.AutoTokenizer.from_pretrained(MODEL, { progress_callback }),
      tf.CLIPTextModelWithProjection.from_pretrained(MODEL, { dtype: DTYPE, progress_callback }),
      fetch("references/index.json", { cache: "no-store" }).then(r => { if (!r.ok) throw new Error("references/index.json missing"); return r.json(); })
    ]);
    const items = index.items.map(({ emb, ...rest }) => rest);
    const vecs = index.items.map(it => decode(it.emb));
    S = { tf, processor, vision, tokenizer, text, items, vecs, dims: vecs[0].length };
    onProgress(1);
    return { count: items.length, model: MODEL, dtype: DTYPE };
  })();
  return ready;
}

export async function embedCanvas(canvas) {
  await ready;
  const blob = await new Promise(r => canvas.toBlob(r, "image/png"));
  const image = await S.tf.RawImage.fromBlob(blob);
  const { image_embeds } = await S.vision(await S.processor(image));
  return unit(Float32Array.from(image_embeds.data));
}

export async function embedText(str) {
  await ready;
  const inputs = S.tokenizer([str], { padding: true, truncation: true });
  const { text_embeds } = await S.text(inputs);
  return unit(Float32Array.from(text_embeds.data));
}

function sims(q) {
  return S.vecs.map(v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * q[i]; return s; });
}
function zscores(a) {
  const m = a.reduce((x, y) => x + y, 0) / a.length;
  const sd = Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length) || 1;
  return a.map(x => (x - m) / sd);
}

// imageVec: embedding of the circled part. textVec: embedding of the words, or null.
// Image–image and text–image similarities live on different scales, so each is z-scored before mixing.
// Near-duplicates are skipped so the k results are varied.
export function rank(imageVec, textVec, k = 4) {
  const si = sims(imageVec), st = textVec ? sims(textVec) : null;
  const zi = zscores(si), zt = st ? zscores(st) : null;
  const score = zi.map((z, i) => (zt ? (1 - TEXT_WEIGHT) * z + TEXT_WEIGHT * zt[i] : z));
  const order = score.map((s, i) => i).sort((a, b) => score[b] - score[a]);
  // Keep the set varied: no near-identical images, no repeated title from the same culture
  // (museums hold long series of almost the same object), and at most MAX_PER_THEME per theme.
  const picked = [], perTheme = {};
  for (const i of order) {
    if (picked.length >= k) break;
    const it = S.items[i];
    if ((perTheme[it.theme] || 0) >= MAX_PER_THEME) continue;
    const dup = picked.some(j => {
      const o = S.items[j];
      if (o.title === it.title && o.culture === it.culture) return true;
      let s = 0; const a = S.vecs[i], b = S.vecs[j];
      for (let d = 0; d < a.length; d++) s += a[d] * b[d];
      return s > 0.9;
    });
    if (dup) continue;
    picked.push(i);
    perTheme[it.theme] = (perTheme[it.theme] || 0) + 1;
  }
  return picked.map(i => ({
    ...S.items[i],
    score: +score[i].toFixed(3),
    image_sim: +si[i].toFixed(4),
    ...(st ? { text_sim: +st[i].toFixed(4) } : {})
  }));
}
