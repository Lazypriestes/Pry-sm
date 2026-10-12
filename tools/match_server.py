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
from transformers import AutoModel

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
            }
        state["model"] = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
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


def match(img, k=6, library="both"):
    img = img.convert("RGB").resize((224, 224))
    feats = patch_features(state["model"], shape_image(img, short=224))       # 16, 16, 384
    qv = feats.reshape(-1, feats.shape[-1]).mean(0); qv /= np.linalg.norm(qv) + 1e-8
    q_hog = hog_cells(img, short=224)
    q_grid = grid_cells(q_hog)                                                 # 16 cells x 9 directions
    q_where, q_energy = edge_layout(q_hog)
    names = [n for n in state["libs"] if library in ("both", n)] or list(state["libs"])
    scored = sorted((r for n in names for r in search(state["libs"][n], n, qv, q_grid, q_where, q_energy)), reverse=True)
    out, seen = [], set()
    for score, c, lay, name, p, x, y, s in scored:                               # one region per photo
        if (name, p) in seen: continue
        seen.add((name, p))
        ph = state["libs"][name]["photos"][p]
        if VESSEL.search(" ".join([ph.get("desc", ""), ph.get("ai", ""), ph.get("channel", "")])): continue
        desc = ph["desc"]
        if name == "arena" and (not desc or re.search(r"\.(jpe?g|png|gif|webp)|^[\w-]{0,40}\d{4,}", desc, re.I)):
            desc = f"from “{ph.get('channel', '').replace('-', ' ').rsplit(' ', 1)[0]}”" if ph.get("channel") else ""
        out.append({"id": ph["id"], "library": name, "score": round(score, 4), "coarse": round(c, 4), "layout": round(lay, 4),
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
                res = match(img, int(b.get("k") or 6), str(b.get("library") or "both"))
            self._send(200, {"matches": res, "ms": round((time.time() - t) * 1000)})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    print(f"matching on http://127.0.0.1:{PORT} (loading index…)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
