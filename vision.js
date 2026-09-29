/* ============================================================
   IMAGE ANALYSIS for the dissection end parts (and analysis.html)
   analyzeImage(img) finds the object's outline against the photo
   background, its colour palette, the most textured patch ("pattern")
   and the most visually distinct patch ("interesting element").
   Needs the page served over http(s): canvas pixels can't be read
   from file:// pages.
   ============================================================ */
const hex = c => "#" + c.map(v => Math.round(v).toString(16).padStart(2, "0")).join("");

// Returns null if the canvas can't be read (e.g. page opened as a file:// URL).
function analyzeImage(img) {
  const L = 360;
  const k = L / Math.max(img.naturalWidth, img.naturalHeight);
  const w = Math.max(1, Math.round(img.naturalWidth * k)), h = Math.max(1, Math.round(img.naturalHeight * k));
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  let px;
  try { px = ctx.getImageData(0, 0, w, h).data; } catch (e) { return null; }
  const N = w * h;

  // Background colour: median of the outer ring of pixels.
  const ring = [[], [], []];
  for (let x = 0; x < w; x++) for (const y of [0, 1, h - 2, h - 1]) for (let c = 0; c < 3; c++) ring[c].push(px[(y * w + x) * 4 + c]);
  for (let y = 0; y < h; y++) for (const x of [0, 1, w - 2, w - 1]) for (let c = 0; c < 3; c++) ring[c].push(px[(y * w + x) * 4 + c]);
  const bg = ring.map(a => a.sort((p, q) => p - q)[a.length >> 1]);

  // Background model that copes with studio gradients: each row's (or column's) own edge colour,
  // used when both ends of that row/column agree; otherwise the global median.
  const at = (x, y) => { const o = (y * w + x) * 4; return [px[o], px[o + 1], px[o + 2]]; };
  const d3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const avg = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  const rowBg = [], colBg = [];
  for (let y = 0; y < h; y++) { const l = at(0, y), r = at(w - 1, y); rowBg.push(d3(l, r) < 40 ? avg(l, r) : null); }
  for (let x = 0; x < w; x++) { const t = at(x, 0), b = at(x, h - 1); colBg.push(d3(t, b) < 40 ? avg(t, b) : null); }

  // Flood-fill background from the border; everything not reached is the object (holes filled).
  const bgLike = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const x = i % w, y = (i / w) | 0, c = [px[i * 4], px[i * 4 + 1], px[i * 4 + 2]];
    let d = d3(c, bg);
    if (rowBg[y]) d = Math.min(d, d3(c, rowBg[y]));
    if (colBg[x]) d = Math.min(d, d3(c, colBg[x]));
    bgLike[i] = d < 32 ? 1 : 0;
  }
  const outside = new Uint8Array(N), queue = new Int32Array(N);
  let qh = 0, qt = 0;
  const push = i => { if (!outside[i] && bgLike[i]) { outside[i] = 1; queue[qt++] = i; } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (qh < qt) {
    const i = queue[qh++], x = i % w, y = (i / w) | 0;
    if (x > 0) push(i - 1); if (x < w - 1) push(i + 1);
    if (y > 0) push(i - w); if (y < h - 1) push(i + w);
  }
  let mask = new Uint8Array(N), area = 0;
  for (let i = 0; i < N; i++) if (!outside[i]) { mask[i] = 1; area++; }
  const maskOk = area > 0.03 * N && area < 0.97 * N;
  if (!maskOk) { mask.fill(1); area = N; }

  // Bounding box and centroid.
  let x0 = w, y0 = h, x1 = 0, y1 = 0, sx = 0, sy = 0;
  for (let i = 0; i < N; i++) if (mask[i]) {
    const x = i % w, y = (i / w) | 0;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    sx += x; sy += y;
  }
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;

  // Grey, gradient (Sobel) and eroded interior so the silhouette edge doesn't count as texture.
  const gray = new Float32Array(N);
  for (let i = 0; i < N; i++) gray[i] = 0.3 * px[i * 4] + 0.59 * px[i * 4 + 1] + 0.11 * px[i * 4 + 2];
  const grad = new Float32Array(N);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x, g = gray;
    const gx = g[i - w + 1] + 2 * g[i + 1] + g[i + w + 1] - g[i - w - 1] - 2 * g[i - 1] - g[i + w - 1];
    const gy = g[i + w - 1] + 2 * g[i + w] + g[i + w + 1] - g[i - w - 1] - 2 * g[i - w] - g[i - w + 1];
    grad[i] = Math.hypot(gx, gy);
  }
  let interior = mask.slice();
  for (let it = 0; it < 4; it++) {
    const next = new Uint8Array(N);
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      next[i] = interior[i] && interior[i - 1] && interior[i + 1] && interior[i - w] && interior[i + w] ? 1 : 0;
    }
    interior = next;
  }

  // Integral images for fast window sums.
  const W = w + 1;
  const I = () => new Float64Array(W * (h + 1));
  const iG = I(), iI = I(), iM = I(), iR = I(), iGr = I(), iB = I();
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, j = (y + 1) * W + x + 1, m = mask[i], n = interior[i];
    const add = (A, v) => { A[j] = v + A[j - 1] + A[j - W] - A[j - W - 1]; };
    add(iG, n ? grad[i] : 0); add(iI, n); add(iM, m);
    add(iR, m ? px[i * 4] : 0); add(iGr, m ? px[i * 4 + 1] : 0); add(iB, m ? px[i * 4 + 2] : 0);
  }
  const sum = (A, x, y, s) => A[(y + s) * W + x + s] - A[y * W + x + s] - A[(y + s) * W + x] + A[y * W + x];
  const tot = (A) => A[h * W + w];
  const mean = [tot(iR) / area, tot(iGr) / area, tot(iB) / area];

  // Candidate windows inside the object.
  const s = Math.max(10, Math.min(Math.round(0.22 * Math.max(bw, bh)), bw, bh, w, h));
  const step = Math.max(2, Math.round(s / 4));
  const wins = [];
  for (let y = y0; y + s <= Math.min(h, y0 + bh + 1); y += step)
    for (let x = x0; x + s <= Math.min(w, x0 + bw + 1); x += step) {
      const m = sum(iM, x, y, s);
      if (m < 0.8 * s * s) continue;
      const n = sum(iI, x, y, s);
      const texture = n > 0.4 * s * s ? sum(iG, x, y, s) / n : 0;
      const c = [sum(iR, x, y, s) / m, sum(iGr, x, y, s) / m, sum(iB, x, y, s) / m];
      const distinct = Math.hypot(c[0] - mean[0], c[1] - mean[1], c[2] - mean[2]);
      wins.push({ x: x + s / 2, y: y + s / 2, texture, distinct });
    }
  if (!wins.length) wins.push({ x: x0 + bw / 2, y: y0 + bh / 2, texture: 0, distinct: 0 });
  const pat = wins.reduce((a, b) => (b.texture > a.texture ? b : a));
  // Interesting element: the most visually distinct patch, away from the pattern patch,
  // among patches with at least median texture (so flat background can't win).
  const medTex = wins.map(v => v.texture).sort((a, b) => a - b)[wins.length >> 1];
  const far = wins.filter(v => v.texture >= medTex && Math.hypot(v.x - pat.x, v.y - pat.y) >= 1.2 * s);
  const det = (far.length ? far : wins).reduce((a, b) =>
    (b.distinct + 0.25 * b.texture > a.distinct + 0.25 * a.texture ? b : a));

  // Colour palette: k-means on object pixels.
  const samples = [];
  const stride = Math.max(1, Math.floor(area / 5000));
  let seen = 0;
  for (let i = 0; i < N; i++) if (mask[i] && (seen++ % stride === 0))
    samples.push([px[i * 4], px[i * 4 + 1], px[i * 4 + 2], i % w, (i / w) | 0]);
  samples.sort((a, b) => (a[0] + a[1] + a[2]) - (b[0] + b[1] + b[2]));
  const K = Math.min(5, samples.length);
  let cents = Array.from({ length: K }, (_, j) => samples[Math.floor((j + 0.5) * samples.length / K)].slice(0, 3));
  let assign = new Int32Array(samples.length);
  for (let it = 0; it < 10; it++) {
    samples.forEach((p, n) => {
      let best = 0, bd = Infinity;
      cents.forEach((c, j) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2; if (d < bd) { bd = d; best = j; } });
      assign[n] = best;
    });
    const acc = cents.map(() => [0, 0, 0, 0]);
    samples.forEach((p, n) => { const a = acc[assign[n]]; a[0] += p[0]; a[1] += p[1]; a[2] += p[2]; a[3]++; });
    cents = acc.map((a, j) => (a[3] ? [a[0] / a[3], a[1] / a[3], a[2] / a[3]] : cents[j]));
  }
  const clusters = cents.map((c, j) => ({ rgb: c, n: 0, sx: 0, sy: 0 }));
  samples.forEach((p, n) => { const c = clusters[assign[n]]; c.n++; c.sx += p[3]; c.sy += p[4]; });
  const palette = clusters.filter(c => c.n).sort((a, b) => b.n - a.n)
    .map(c => ({ hex: hex(c.rgb), share: +(c.n / samples.length).toFixed(3), x: c.sx / c.n / w, y: c.sy / c.n / h }));

  // Shape: flat silhouette.
  const shapeCv = document.createElement("canvas");
  shapeCv.width = w; shapeCv.height = h;
  const sImg = shapeCv.getContext("2d").createImageData(w, h);
  for (let i = 0; i < N; i++) {
    const o = i * 4;
    sImg.data[o] = sImg.data[o + 1] = sImg.data[o + 2] = mask[i] ? 30 : 244; sImg.data[o + 3] = 255;
  }
  shapeCv.getContext("2d").putImageData(sImg, 0, 0);

  // Anchor for the shape line: the outline's left edge, 40% of the way down.
  const ry = Math.round(y0 + 0.4 * bh);
  let rx = x0;
  for (let x = 0; x < w; x++) if (mask[ry * w + x]) { rx = x; break; }

  const L2 = Math.max(w, h);
  return {
    maskOk, shapeCv,
    raw: { w, h, mask, interior, grad },       // for analysis.html visualisations
    bbox: { x: x0 / w, y: y0 / h, w: bw / w, h: bh / h },
    anchors: {
      shape: { x: rx / w, y: ry / h },
      colour: palette.length ? { x: palette[0].x, y: palette[0].y } : { x: 0.5, y: 0.5 }
    },
    pattern: { x: pat.x / w, y: pat.y / h, size: s / L2,
               texture: +pat.texture.toFixed(1), median_texture: +medTex.toFixed(1) },
    detail: { x: det.x / w, y: det.y / h, size: s / L2 },
    palette
  };
}
