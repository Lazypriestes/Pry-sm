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
import base64, io, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(__file__))
from match_index import shape_image, hog_cells, patch_features, MODEL, DEVICE, ROOT, IMG_DIR  # noqa: E402

PORT = 8003
IDX = os.path.join(ROOT, "index")
COARSE_K = 400                    # candidates from the fast search, re-checked for layout
GRID = 4                          # layout check on a 4x4 grid of cells
state = {"ready": False, "error": None}
LOCK = threading.Lock()


def load():
    try:
        t = time.time()
        meta = json.load(open(os.path.join(IDX, "photos.json")))
        state["photos"] = meta["photos"]
        state["win"] = torch.from_numpy(np.load(os.path.join(IDX, "win_vecs.npy"))).to(DEVICE)
        state["win_meta"] = np.load(os.path.join(IDX, "win_meta.npy"))
        state["hog"] = np.load(os.path.join(IDX, "hog.npy"), mmap_mode="r")
        state["offsets"] = np.load(os.path.join(IDX, "tok_offsets.npy"))
        state["mu"] = np.load(os.path.join(IDX, "pca_mu.npy"))
        state["basis"] = np.load(os.path.join(IDX, "pca_basis.npy"))
        state["model"] = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
        state["ready"] = True
        print(f"ready: {len(state['photos'])} photos, {len(state['win_meta'])} regions in {time.time() - t:.0f}s", flush=True)
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


def match(img, k=6):
    img = img.convert("RGB").resize((224, 224))
    feats = patch_features(state["model"], shape_image(img, short=224))       # 16, 16, 384
    qv = feats.reshape(-1, feats.shape[-1]).mean(0); qv /= np.linalg.norm(qv) + 1e-8
    q_grid = grid_cells(hog_cells(img, short=224))                             # 16 cells x 9 directions
    # 1. fast: cosine against every region descriptor, on the GPU
    sims = state["win"] @ torch.from_numpy(qv.astype(np.float16)).to(DEVICE)
    top = torch.topk(sims.float(), min(COARSE_K, sims.shape[0])).indices.cpu().numpy()
    coarse = sims.float().cpu().numpy()
    # 2. careful: does the arrangement inside the region match too?
    scored = []
    for i in top:
        p, x, y, s = state["win_meta"][i]
        ph = state["photos"][p]
        o = state["offsets"][p]
        hog = np.asarray(state["hog"][o:o + ph["gh"] * ph["gw"]], dtype=np.float32).reshape(ph["gh"], ph["gw"], -1)
        layout = float((grid_cells(hog[y:y + s, x:x + s]) * q_grid).sum(1).mean())
        scored.append((0.5 * float(coarse[i]) + 0.5 * layout, float(coarse[i]), layout, p, x, y, s))
    scored.sort(reverse=True)
    out, seen = [], set()
    for score, c, lay, p, x, y, s in scored:                                     # one region per photo
        if p in seen: continue
        seen.add(p)
        ph = state["photos"][p]
        out.append({"id": ph["id"], "score": round(score, 4), "coarse": round(c, 4), "layout": round(lay, 4),
                    "box": [x / ph["gw"], y / ph["gh"], s / ph["gw"], s / ph["gh"]],
                    "w": ph["w"], "h": ph["h"], "desc": ph["desc"], "by": ph["by"], "url": ph["url"]})
        if len(out) >= k: break
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            return self._send(200, {"ready": state["ready"], "error": state["error"],
                                    "photos": len(state.get("photos", [])), "regions": len(state.get("win_meta", []))})
        if self.path.startswith("/photo/") and self.path.endswith(".jpg"):
            pid = os.path.basename(self.path)[:-4]
            f = os.path.join(IMG_DIR, pid + ".jpg")
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
                res = match(img, int(b.get("k") or 6))
            self._send(200, {"matches": res, "ms": round((time.time() - t) * 1000)})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    print(f"matching on http://127.0.0.1:{PORT} (loading index…)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
