// Builds the reference library from The Met's Open Access collection (public domain / CC0).
// Usage:  node tools/fetch-met.mjs [perTheme=20]
// Writes references/img/met_<id>.jpg and references/meta.json. Re-running skips what is already there.
// Then run tools/build-index.mjs to compute the CLIP embeddings.
import { mkdir, writeFile, readFile, access } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "references");
const IMG = join(OUT, "img");
const API = "https://collectionapi.metmuseum.org/public/collection/v1";
const PER_THEME = Number(process.argv[2] || 20);

// Themes chosen to sit *behind* parts of a vase: surfaces, motifs, structures, natural forms.
const THEMES = [
  "ornament", "textile pattern", "architectural detail", "botanical", "shell", "insect",
  "bird", "dragon", "landscape", "mosaic", "metalwork", "lacquer", "flower", "wave",
  "mask", "scroll", "lattice", "fish", "tree", "jewelry"
];
// Skip vessels so a circled vase part doesn't just match another vase.
const SKIP_NAMES = /\b(vase|jar|bottle|bowl|ewer|jug|pitcher|cup|beaker|amphora|urn)\b/i;

const sleep = ms => new Promise(r => setTimeout(r, ms));
const exists = p => access(p).then(() => true, () => false);

// The API throttles bursts (403/429); back off and retry rather than silently skipping.
async function getJSON(url, tries = 6) {
  for (let t = 0; t < tries; t++) {
    try {
      const r = await fetch(url);
      if (r.ok) return await r.json();
      if (r.status === 404) return null;
    } catch (e) { /* network hiccup: retry */ }
    await sleep(2000 * (t + 1));
  }
  console.warn(`\ngave up on ${url}`);
  return null;
}

await mkdir(IMG, { recursive: true });
const metaPath = join(OUT, "meta.json");
const meta = (await exists(metaPath)) ? JSON.parse(await readFile(metaPath, "utf8")) : { source: "The Metropolitan Museum of Art Open Access (CC0)", items: [] };
const have = new Set(meta.items.map(i => i.objectID));

for (const theme of THEMES) {
  const already = meta.items.filter(i => i.theme === theme).length;
  if (already >= PER_THEME) { console.log(`${theme}: ${already} already`); continue; }
  const search = await getJSON(`${API}/search?hasImages=true&isPublicDomain=true&q=${encodeURIComponent(theme)}`);
  const ids = (search && search.objectIDs) || [];
  let got = already;
  const todo = ids.filter(id => !have.has(id));
  if (!ids.length) console.warn(`${theme}: search returned nothing`);
  for (let start = 0; start < todo.length && got < PER_THEME; start += 4) {
    // 4 records at a time, then keep the usable ones in order.
    const batch = await Promise.all(todo.slice(start, start + 4).map(id => getJSON(`${API}/objects/${id}`)));
    await sleep(250);
    for (const o of batch) {
      if (got >= PER_THEME) break;
      if (!o || !o.isPublicDomain || !o.primaryImageSmall || have.has(o.objectID)) continue;
      if (SKIP_NAMES.test(o.objectName || "") || SKIP_NAMES.test(o.title || "")) continue;
      const id = o.objectID;
      const file = `img/met_${id}.jpg`;
      try {
        const r = await fetch(o.primaryImageSmall);
        if (!r.ok) continue;
        await writeFile(join(OUT, file), Buffer.from(await r.arrayBuffer()));
      } catch (e) { continue; }
      meta.items.push({
        id: `met_${id}`, objectID: id, file, theme,
        title: o.title || "", date: o.objectDate || "", culture: o.culture || "",
        artist: o.artistDisplayName || "", medium: o.medium || "", name: o.objectName || "",
        department: o.department || "", url: o.objectURL || ""
      });
      have.add(id); got++;
      process.stdout.write(`\r${theme}: ${got}/${PER_THEME}   `);
    }
  }
  console.log();
  await writeFile(metaPath, JSON.stringify(meta, null, 1));
}
console.log(`total ${meta.items.length} references`);
