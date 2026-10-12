"""Real-image matching for the study: finds photos containing a region shaped like the circled part.

  POST /match  {image: dataURL (square crop of the part), k?}  -> {matches: [...], ms}
       each match: {id, score, coarse, layout, box: [x, y, w, h] as fractions of the photo,
                    w, h (photo px), desc, by, url}
  GET  /photo/<id>.jpg                                          -> the photo (local file)
  GET  /status                                                  -> {ready, photos, regions}

Uses the index built by tools/match_index.py (same greyscale + DINOv2 features). server.mjs
forwards /api/match and /api/photo/... here. Run with the image-model venv:
  ~/.local/share/objview-gen/.venv/bin/python tools/match_server.py
"""
import base64, io, json, os, re, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(__file__))
from match_index import shape_image, hog_cells, patch_features, MODEL, DEVICE, ROOT, LIBRARIES  # noqa: E402

PORT = 8003
COARSE_K = 400                    # candidates from the fast search, re-checked for layout
GRID = 4                          # layout check on a 4x4 grid of cells
# A vase should turn into something *else*: skip matches that are themselves vessels.
VESSEL = re.compile(r"\b(vases?|vessels?|pots?|pottery|ceramics?|stoneware|porcelain|bottles?|jars?|cups?|urns?|amphora|jugs?|pitchers?)\b", re.I)
state = {"ready": False, "error": None}
LOCK = threading.Lock()


def load():
    """Load every library that has been indexed (Unsplash, Are.na, ...)."""
    try:
        t = time.time()
        state["libs"] = {}
        for name, (img_dir, _, idx) in LIBRARIES.items():
            if not os.path.exists(os.path.join(idx, "photos.json")): continue
            state["libs"][name] = {
                "img_dir": img_dir,
                "photos": json.load(open(os.path.join(idx, "photos.json")))["photos"],
                "win": torch.from_numpy(np.load(os.path.join(idx, "win_vecs.npy"))).to(DEVICE),
                "win_meta": np.load(os.path.join(idx, "win_meta.npy")),
                "hog": np.load(os.path.join(idx, "hog.npy"), mmap_mode="r"),
                "offsets": np.load(os.path.join(idx, "tok_offsets.npy")),
                # CLIP "what is this a picture of" per photo (tools/clip_index.py), for context
                "clip": np.load(os.path.join(idx, "clip.npy")).astype(np.float32) if os.path.exists(os.path.join(idx, "clip.npy")) else None,
            }
        state["model"] = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
        state["clip"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16).to(DEVICE).eval()
        state["clip_proc"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        state["vase_dir"] = clip_text("a photo of a vase")          # removed from the whole-vase look
        flag_vessels()
        state["ready"] = True
        print("ready: " + ", ".join(f"{n} {len(L['photos'])} photos" for n, L in state["libs"].items()) + f" in {time.time() - t:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        state["error"] = repr(e)
        print("load failed:", e, flush=True)


def grid_cells(feats, n=GRID):
    """Average a (h, w, d) feature block into n x n cells, each L2-normalised -> (n*n, d)."""
    h, w, d = feats.shape
    ys = np.linspace(0, h, n + 1).round().astype(int); xs = np.linspace(0, w, n + 1).round().astype(int)
    cells = []
    for i in range(n):
        for j in range(n):
            c = feats[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)].reshape(-1, d).mean(0)
            cells.append(c / (np.linalg.norm(c) + 1e-8))
    return np.stack(cells)


@torch.no_grad()
def clip_text(text):
    t = state["clip_proc"](text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    e = state["clip"].text_projection(state["clip"].text_model(**t).pooler_output).float()[0].cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-8)


@torch.no_grad()
def clip_image(img):
    x = state["clip_proc"](images=[img.convert("RGB")], return_tensors="pt")["pixel_values"].to(DEVICE, torch.float16)
    e = state["clip"].visual_projection(state["clip"].vision_model(pixel_values=x).pooler_output).float()[0].cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-8)


# Visual vessel filter: CLIP looks at every library photo once; anything that reads mainly as a
# vase, pot, bowl, jar or bottle is never offered as a match (titles alone miss untitled images).
VESSEL_LABELS = ["a vase", "a ceramic pot", "a bowl", "a jar", "a bottle", "a vessel", "pottery", "a teapot", "a cup"]
OTHER_LABELS = ["a landscape", "a building", "architecture", "a plant", "a flower", "an animal", "a person", "a street",
                "the sky", "water", "a texture", "fabric", "a sculpture", "furniture", "food", "a painting", "a pattern",
                "a machine", "a room", "rocks", "trees", "the sea", "a city", "an object"]


def flag_vessels():
    labels = VESSEL_LABELS + OTHER_LABELS
    T = np.stack([clip_text(f"a photo of {l}") for l in labels])            # (labels, 512)
    for name, L in state["libs"].items():
        if L["clip"] is None: L["vessel"] = np.zeros(len(L["photos"]), bool); continue
        logits = 100 * L["clip"] @ T.T
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        L["vessel"] = p[:, :len(VESSEL_LABELS)].sum(1) > 0.5
        print(f"{name}: {int(L['vessel'].sum())} photos look like vessels and are skipped", flush=True)


def context_vector(ctx):
    """One direction for the vase's context: its description (era, material, origin, production, style)
    plus the look of the whole vase with the generic idea of "a vase" taken out (so matches don't drift
    towards other vases)."""
    parts = []
    if ctx.get("text"):
        # A "world" the vase belongs to, not its material: "a photo evoking Yuan dynasty, South China, 1300s".
        parts.append(0.6 * clip_text("a photo evoking " + str(ctx["text"])[:250]))
    if isinstance(ctx.get("whole"), str) and ctx["whole"].startswith("data:image/"):
        w = clip_image(Image.open(io.BytesIO(base64.b64decode(ctx["whole"].split(",", 1)[1]))))
        w = w - (w @ state["vase_dir"]) * state["vase_dir"]
        parts.append(0.4 * w / (np.linalg.norm(w) + 1e-8))
    if not parts: return None
    v = sum(parts)
    return v / (np.linalg.norm(v) + 1e-8)


def zs(a):
    a = np.asarray(a, np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def edge_layout(cells_raw, n=GRID):
    """Where the edges are: total edge strength in each of n x n cells, normalised (n*n,)."""
    h, w, _ = cells_raw.shape
    ys = np.linspace(0, h, n + 1).round().astype(int); xs = np.linspace(0, w, n + 1).round().astype(int)
    m = np.array([cells_raw[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)].sum()
                  for i in range(n) for j in range(n)], np.float32)
    return m / (np.linalg.norm(m) + 1e-8), float(cells_raw.sum(-1).mean())


def search(L, name, qv, q_grid, q_where, q_energy):
    sims = L["win"] @ torch.from_numpy(qv.astype(np.float16)).to(DEVICE)
    top = torch.topk(sims.float(), min(COARSE_K, sims.shape[0])).indices.cpu().numpy()
    coarse = sims.float().cpu().numpy()
    scored = []
    for i in top:
        p, x, y, s = L["win_meta"][i]
        ph = L["photos"][p]
        o = L["offsets"][p]
        hog = np.asarray(L["hog"][o:o + ph["gh"] * ph["gw"]], dtype=np.float32).reshape(ph["gh"], ph["gw"], -1)
        block = hog[y:y + s, x:x + s]
        direction = float((grid_cells(block) * q_grid).sum(1).mean())      # which way the lines run
        where, energy = edge_layout(block)
        location = float(where @ q_where)                                     # where the edges are
        layout = 0.6 * direction + 0.4 * location
        # Penalise regions with far less (or far more) going on than the circled part.
        ratio = min(energy, q_energy) / (max(energy, q_energy) + 1e-6)
        score = (0.5 * float(coarse[i]) + 0.5 * layout) * (0.75 + 0.25 * ratio)
        scored.append((score, float(coarse[i]), layout, name, p, x, y, s))
    return scored


def match(img, k=6, library="both", ctx=None):
    img = img.convert("RGB").resize((224, 224))
    feats = patch_features(state["model"], shape_image(img, short=224))       # 16, 16, 384
    qv = feats.reshape(-1, feats.shape[-1]).mean(0); qv /= np.linalg.norm(qv) + 1e-8
    q_hog = hog_cells(img, short=224)
    q_grid = grid_cells(q_hog)                                                 # 16 cells x 9 directions
    q_where, q_energy = edge_layout(q_hog)
    names = [n for n in state["libs"] if library in ("both", n)] or list(state["libs"])
    scored = [r for n in names for r in search(state["libs"][n], n, qv, q_grid, q_where, q_energy)]
    # Context: tilt the ranking towards photos that fit the vase (shape still leads; weight 0 = shape only).
    weight = float((ctx or {}).get("weight") or 0)
    cvec = context_vector(ctx) if ctx and weight > 0 else None
    ctx_sim = {}
    if cvec is not None:
        cs = [float(state["libs"][r[3]]["clip"][r[4]] @ cvec) if state["libs"][r[3]]["clip"] is not None else 0.0 for r in scored]
        z_shape, z_ctx = zs([r[0] for r in scored]), zs(cs)
        ctx_sim = {(r[3], r[4]): c for r, c in zip(scored, cs)}
        scored = [(float((1 - weight) * a + weight * b),) + r[1:] + (r[0],) for r, a, b in zip(scored, z_shape, z_ctx)]
    scored.sort(reverse=True)
    out, seen = [], set()
    for score, c, lay, name, p, x, y, s, *shape_score in scored:                # one region per photo
        if (name, p) in seen: continue
        seen.add((name, p))
        ph = state["libs"][name]["photos"][p]
        if VESSEL.search(" ".join([ph.get("desc", ""), ph.get("ai", ""), ph.get("channel", "")])): continue
        if state["libs"][name].get("vessel") is not None and state["libs"][name]["vessel"][p]: continue
        desc = ph["desc"]
        if name == "arena" and (not desc or re.search(r"\.(jpe?g|png|gif|webp)|^[\w-]{0,40}\d{4,}", desc, re.I)):
            desc = f"from “{ph.get('channel', '').replace('-', ' ').rsplit(' ', 1)[0]}”" if ph.get("channel") else ""
        out.append({"id": ph["id"], "library": name, "score": round(score, 4), "coarse": round(c, 4), "layout": round(lay, 4),
                    **({"shape_score": round(shape_score[0], 4), "context_sim": round(ctx_sim.get((name, p), 0.0), 4)} if shape_score else {}),
                    "box": [x / ph["gw"], y / ph["gh"], s / ph["gw"], s / ph["gh"]],
                    "w": ph["w"], "h": ph["h"], "desc": desc, "by": ph["by"], "url": ph["url"]})
        if len(out) >= k: break
    return out


_remote = {}


def remote_url(lib, pid):
    """Where the photo lives online (Unsplash CDN, or the Are.na image URL saved by import_arena.py)."""
    if lib not in _remote:
        import csv
        csv.field_size_limit(10**9)
        tsv = LIBRARIES[lib][1]
        rows = csv.DictReader(open(tsv, newline=""), delimiter="\t") if os.path.exists(tsv) else []
        if lib == "unsplash":
            _remote[lib] = {r["photo_id"]: r["photo_image_url"] + "?w=1280&q=80&fm=jpg" for r in rows if r.get("photo_image_url")}
        else:
            _remote[lib] = {r["photo_id"]: r["image_url"] for r in rows if r.get("image_url")}
    return _remote[lib].get(pid)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            libs = state.get("libs", {})
            return self._send(200, {"ready": state["ready"], "error": state["error"],
                                    "photos": sum(len(L["photos"]) for L in libs.values()),
                                    "libraries": {n: len(L["photos"]) for n, L in libs.items()}})
        if self.path.startswith("/photo/") and self.path.endswith(".jpg"):
            pid = os.path.basename(self.path)[:-4]
            lib = "arena" if pid.startswith("arena_") else "unsplash"
            f = os.path.join(LIBRARIES[lib][0], pid + ".jpg")
            if "/" not in pid and not os.path.exists(f) and remote_url(lib, pid):
                # Not kept locally: send the browser to the image on Are.na's / Unsplash's own servers.
                self.send_response(302); self.send_header("Location", remote_url(lib, pid)); self.end_headers(); return
            if "/" not in pid and os.path.exists(f):
                data = open(f, "rb").read()
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "max-age=86400")
                self.end_headers(); self.wfile.write(data); return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/match": return self._send(404, {"error": "not found"})
        if not state["ready"]: return self._send(503, {"error": state["error"] or "index still loading"})
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            img = Image.open(io.BytesIO(base64.b64decode(b["image"].split(",", 1)[1])))
            t = time.time()
            with LOCK:
                res = match(img, int(b.get("k") or 6), str(b.get("library") or "both"), b.get("context"))
            self._send(200, {"matches": res, "ms": round((time.time() - t) * 1000)})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    print(f"matching on http://127.0.0.1:{PORT} (loading index…)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
