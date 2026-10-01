// Removes the photo background from every catalogue object (fal.ai BiRefNet v2) and saves
// transparent PNGs to images/cut/<id>.png, then records them in catalogue.json as "cutout".
// Usage (from the prototype folder):  node tools/cutout.mjs        (re-running skips done ones)
import { readFile, writeFile, mkdir, access } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const env = Object.fromEntries((await readFile(join(ROOT, ".env"), "utf8")).trim().split("\n").map(l => l.split(/=(.*)/s).slice(0, 2)));
const exists = p => access(p).then(() => true, () => false);
const catPath = join(ROOT, "catalogue.json");
const cat = JSON.parse(await readFile(catPath, "utf8"));
await mkdir(join(ROOT, "images", "cut"), { recursive: true });

async function cut(item) {
  const out = `images/cut/${item.id}.png`;
  if (!(await exists(join(ROOT, out)))) {
    const img = "data:image/jpeg;base64," + (await readFile(join(ROOT, item.image))).toString("base64");
    for (let t = 0; t < 3; t++) {
      try {
        const r = await fetch("https://fal.run/fal-ai/birefnet/v2", {
          method: "POST", signal: AbortSignal.timeout(120000),
          headers: { Authorization: `Key ${env.FAL_KEY}`, "Content-Type": "application/json" },
          body: JSON.stringify({ image_url: img, output_format: "png" })
        });
        const j = await r.json();
        if (!j.image) throw new Error(JSON.stringify(j).slice(0, 120));
        await writeFile(join(ROOT, out), Buffer.from(await (await fetch(j.image.url)).arrayBuffer()));
        break;
      } catch (e) { if (t === 2) { console.warn(`\n${item.id}: ${e.message}`); return; } }
    }
  }
  item.cutout = out;
}

let done = 0;
const queue = cat.items.slice();
await Promise.all(Array.from({ length: 4 }, async () => {
  while (queue.length) { await cut(queue.shift()); process.stdout.write(`\rcutouts ${++done}/${cat.items.length}`); }
}));
console.log();
await writeFile(catPath, JSON.stringify(cat, null, 2));
console.log(`${cat.items.filter(i => i.cutout).length} cutouts recorded in catalogue.json`);
