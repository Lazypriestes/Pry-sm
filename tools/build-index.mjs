// Computes CLIP image embeddings for the reference library and for each catalogue object.
// Usage (from the prototype folder):  cd tools && npm install && node build-index.mjs
// Writes references/index.json (references + embeddings) and references/vases.json.
// The browser uses the same model and dtype (see references.js) so the vectors are comparable.
import { readFile, writeFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { AutoProcessor, CLIPVisionModelWithProjection, RawImage } from "@huggingface/transformers";

export const MODEL = "Xenova/clip-vit-base-patch32";
export const DTYPE = "q8";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const REF = join(ROOT, "references");

const processor = await AutoProcessor.from_pretrained(MODEL);
const vision = await CLIPVisionModelWithProjection.from_pretrained(MODEL, { dtype: DTYPE });

async function embed(path) {
  const image = await RawImage.read(path);
  const { image_embeds } = await vision(await processor(image));
  const v = Float32Array.from(image_embeds.data);
  const n = Math.hypot(...v);
  for (let i = 0; i < v.length; i++) v[i] /= n;          // unit length: dot product = cosine similarity
  return Buffer.from(v.buffer).toString("base64");
}

const meta = JSON.parse(await readFile(join(REF, "meta.json"), "utf8"));
const items = [];
for (const [n, it] of meta.items.entries()) {
  try { items.push({ ...it, emb: await embed(join(REF, it.file)) }); }
  catch (e) { console.warn(`\nskip ${it.id}: ${e.message}`); }
  process.stdout.write(`\rreferences ${n + 1}/${meta.items.length}`);
}
console.log();
await writeFile(join(REF, "index.json"), JSON.stringify({
  model: MODEL, dtype: DTYPE, dims: 512, encoding: "base64 float32, unit length",
  source: meta.source, built_at: new Date().toISOString(), items
}));

const cat = JSON.parse(await readFile(join(ROOT, "catalogue.json"), "utf8"));
const vases = {};
for (const [n, it] of cat.items.entries()) {
  vases[it.id] = await embed(join(ROOT, it.image));
  process.stdout.write(`\rvases ${n + 1}/${cat.items.length}`);
}
console.log();
await writeFile(join(REF, "vases.json"), JSON.stringify({ model: MODEL, dtype: DTYPE, catalogue_version: cat.version, emb: vases }));
console.log(`index: ${items.length} references, ${Object.keys(vases).length} vases`);
