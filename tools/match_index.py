"""Builds the shape index for "real matches": which region of which photo looks like a circled part.

For every photo in ~/.local/share/objview-match/images (Unsplash Lite, research use, kept outside the
project and never redistributed):
  1. greyscale + contrast stretch, then an edge drawing (dark lines on white);
  2. DINOv2-small patch features of its *line drawing* (edges), so shape counts, not objects;
  3. region descriptors: mean of the patch features in sliding windows of 5x5 and 9x9 patches;
  4. an edge-direction histogram (HOG) per patch, used to re-check the *layout* of the best
     candidates: which way the lines run, cell by cell (plus a compact PCA copy of the features).

Run with the image-model venv (outside iCloud):
  ~/.local/share/objview-gen/.venv/bin/python tools/match_index.py
Writes ~/.local/share/objview-match/index/ (about 1.5 GB for 25k photos).
"""
import csv, json, os, sys, time
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from transformers import AutoModel

ROOT = os.path.expanduser("~/.local/share/objview-match")
# Libraries: images folder, metadata TSV (Unsplash Lite columns), index folder.
LIBRARIES = {
    "unsplash": (os.path.join(ROOT, "images"), os.path.join(ROOT, "lite/photos.tsv000"), os.path.join(ROOT, "index")),
    "arena":    (os.path.join(ROOT, "arena/images"), os.path.join(ROOT, "arena/photos.tsv"), os.path.join(ROOT, "arena/index")),
}
LIBRARY = os.environ.get("LIBRARY", "unsplash")
IMG_DIR, META_TSV, OUT = LIBRARIES[LIBRARY]
MODEL = "facebook/dinov2-small"
SHORT = 224                      # short side in pixels → 16 patches of 14 px
WINDOWS = [(5, 3), (9, 4)]       # (size in patches, stride)
PCA_DIMS = 64
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def grey(img, short=SHORT):
    """Greyscale, 2-98% contrast stretch, resized so both sides are multiples of 14."""
    g = ImageOps.autocontrast(img.convert("L"), cutoff=2)
    w, h = g.size
    k = short / min(w, h)
    W, H = max(14, round(w * k / 14) * 14), max(14, round(h * k / 14) * 14)
    return g.resize((W, H), Image.BICUBIC)


def gradients(g):
    a = np.asarray(g.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32)
    gx = np.zeros_like(a); gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]; gy[1:-1, :] = a[2:, :] - a[:-2, :]
    return gx, gy


def shape_image(img, short=SHORT):
    """What the model sees: a line drawing of the image (dark edges on white). Colour, material and
    "this is an object" mostly drop out; the lines that make the shape stay."""
    g = grey(img, short)
    gx, gy = gradients(g)
    mag = np.hypot(gx, gy)
    mag = np.clip(mag / (np.percentile(mag, 97) + 1e-6), 0, 1)
    line = (255 * (1 - mag)).astype(np.uint8)
    return Image.fromarray(line).convert("RGB")


def hog_cells(img, short=SHORT, bins=9):
    """Edge-direction histogram per 14 px cell (unsigned 0-180°, weighted by edge strength): (gh, gw, bins)."""
    g = grey(img, short)
    gx, gy = gradients(g)
    mag = np.hypot(gx, gy); ang = (np.degrees(np.arctan2(gy, gx)) + 180) % 180
    b = np.minimum((ang / (180 / bins)).astype(int), bins - 1)
    gh, gw = g.size[1] // 14, g.size[0] // 14
    out = np.zeros((gh, gw, bins), np.float32)
    cy = np.minimum(np.arange(g.size[1]) // 14, gh - 1); cx = np.minimum(np.arange(g.size[0]) // 14, gw - 1)
    np.add.at(out, (cy[:, None].repeat(g.size[0], 1), cx[None, :].repeat(g.size[1], 0), b), mag)
    return out


def to_tensor(imgs):
    a = np.stack([np.asarray(i, dtype=np.float32) / 255.0 for i in imgs])
    t = torch.from_numpy(a).permute(0, 3, 1, 2)
    return (t - MEAN) / STD


@torch.no_grad()
def patch_features(model, img):
    """(H/14, W/14, 384) patch features, L2-normalised."""
    x = to_tensor([img]).to(DEVICE, torch.float16)
    tok = model(pixel_values=x).last_hidden_state[0, 1:].float()       # drop CLS
    gh, gw = img.size[1] // 14, img.size[0] // 14
    tok = torch.nn.functional.normalize(tok, dim=-1)
    return tok.reshape(gh, gw, -1).cpu().numpy()


def windows(gh, gw):
    for s, stride in WINDOWS:
        if s > gh or s > gw: continue
        ys = list(range(0, gh - s + 1, stride)); xs = list(range(0, gw - s + 1, stride))
        if ys[-1] != gh - s: ys.append(gh - s)
        if xs[-1] != gw - s: xs.append(gw - s)
        for y in ys:
            for x in xs:
                yield x, y, s


def main():
    os.makedirs(OUT, exist_ok=True)
    csv.field_size_limit(10**9)
    meta = {r["photo_id"]: r for r in csv.DictReader(open(META_TSV, newline=""), delimiter="\t")}
    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()

    photos, win_vecs, win_meta, tok_chunks, tok_offsets, hog_chunks = [], [], [], [], [], []
    offset, t0 = 0, time.time()
    for n, f in enumerate(files):
        try:
            img = Image.open(os.path.join(IMG_DIR, f)); img.load()
        except Exception:
            continue
        si = shape_image(img)
        feats = patch_features(model, si)                      # gh, gw, 384
        gh, gw, _ = feats.shape
        pid = f[:-4]; m = meta.get(pid, {})
        idx = len(photos)
        photos.append({"id": pid, "w": img.size[0], "h": img.size[1], "gw": gw, "gh": gh,
                       "desc": (m.get("photo_description") or m.get("ai_description") or "").strip(),
                       "ai": (m.get("ai_description") or "").strip(),
                       "by": " ".join(x for x in [m.get("photographer_first_name", ""), m.get("photographer_last_name", "")] if x).strip(),
                       "url": m.get("photo_url", ""), **({"channel": m["channel"]} if m.get("channel") else {})})
        for x, y, s in windows(gh, gw):
            v = feats[y:y + s, x:x + s].reshape(-1, feats.shape[-1]).mean(0)
            win_vecs.append(v / (np.linalg.norm(v) + 1e-8))
            win_meta.append((idx, x, y, s))
        tok_chunks.append(feats.reshape(-1, feats.shape[-1]).astype(np.float16))
        hog_chunks.append(hog_cells(img).reshape(-1, 9).astype(np.float16))
        tok_offsets.append(offset); offset += gh * gw
        if (n + 1) % 1000 == 0:
            print(f"{n + 1}/{len(files)} photos · {len(win_vecs)} regions · {time.time() - t0:.0f}s", flush=True)

    tokens = np.concatenate(tok_chunks).astype(np.float32)
    # PCA basis for the compact layout features (fit on a sample of patches).
    sample = tokens[np.random.default_rng(0).choice(len(tokens), min(200000, len(tokens)), replace=False)]
    mu = sample.mean(0)
    _, _, vt = np.linalg.svd(sample - mu, full_matrices=False)
    basis = vt[:PCA_DIMS].T.astype(np.float32)                 # 384 x 64
    small = ((tokens - mu) @ basis).astype(np.float16)

    np.save(os.path.join(OUT, "win_vecs.npy"), np.stack(win_vecs).astype(np.float16))
    np.save(os.path.join(OUT, "win_meta.npy"), np.array(win_meta, dtype=np.int32))
    np.save(os.path.join(OUT, "tokens_pca.npy"), small)
    np.save(os.path.join(OUT, "hog.npy"), np.concatenate(hog_chunks))
    np.save(os.path.join(OUT, "tok_offsets.npy"), np.array(tok_offsets, dtype=np.int64))
    np.save(os.path.join(OUT, "pca_mu.npy"), mu.astype(np.float32))
    np.save(os.path.join(OUT, "pca_basis.npy"), basis)
    json.dump({"model": MODEL, "short": SHORT, "windows": WINDOWS, "photos": photos,
               "source": "Unsplash Lite Dataset 1.4.0 (research use; images not redistributed)"},
              open(os.path.join(OUT, "photos.json"), "w"))
    print(f"done: {len(photos)} photos, {len(win_vecs)} regions in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
