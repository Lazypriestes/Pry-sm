"""Finds the objects in a photo of a room (for room.html): name, box and outline of each.

  POST /objects {image: dataURL}  -> {objects: [{id, name, desc, box: [x, y, w, h], poly: [[x, y], ...]}], caption, ms}
       box and poly are fractions of the photo's width / height
  GET  /status                    -> {ready, error}

Florence-2-large (MIT) runs three passes and merges them: object detection (short names), dense region
captions (descriptions), and grounding of the phrases in a detailed caption (things the first two miss).
Each object then gets an outline (region-to-segmentation). Results are cached per photo, so a room is
only read once. server.mjs forwards /api/objects here. Run with the image-model venv:
  ~/.local/share/objview-gen/.venv/bin/python tools/room_server.py
"""
import base64, hashlib, io, json, os, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

PORT = 8004
MODEL = "florence-community/Florence-2-large"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CACHE = os.path.expanduser("~/.local/share/objview-match/rooms/cache")
# The room itself is not an object: surfaces and whole-scene phrases are dropped.
SKIP = re.compile(r"^(the |a |an |two |several |some )?(room|living room|bedroom|kitchen|walls?|floor|ceiling|house|interior|"
                  r"image|photo|scene|decorative items?|decorations?|items?|things?|objects?|space|area|corner|side)$", re.I)
state = {"ready": False, "error": None}
LOCK = threading.Lock()


def load():
    try:
        t = time.time()
        state["model"] = Florence2ForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
        state["proc"] = AutoProcessor.from_pretrained(MODEL)
        state["ready"] = True
        print(f"ready in {time.time() - t:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        state["error"] = repr(e)
        print("load failed:", e, flush=True)


@torch.no_grad()
def run(img, task, text="", beams=3, tokens=1024):
    proc = state["proc"]
    x = proc(text=task + text, images=img, return_tensors="pt").to(DEVICE, torch.float16)
    ids = state["model"].generate(**x, max_new_tokens=tokens, num_beams=beams)
    return proc.post_process_generation(proc.batch_decode(ids, skip_special_tokens=False)[0], task=task, image_size=img.size)[task]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (union + 1e-6)


def short_name(phrase):
    """'wooden armchair with gray cushion' -> 'armchair'-ish: drop articles and trailing clauses."""
    p = re.sub(r"^(the|a|an|two|several|some)\s+", "", phrase.strip().lower())
    p = re.split(r"\s+(with|on|in|of|that|and|hanging|above|below|next)\s+", p)[0]
    return p.strip(" .,")


def objects(img):
    W, H = img.size
    cands = []                                                   # (box, name, desc, source)
    od = run(img, "<OD>")
    for b, l in zip(od["bboxes"], od["labels"]): cands.append((b, l.lower(), "", "od"))
    dense = run(img, "<DENSE_REGION_CAPTION>")
    for b, l in zip(dense["bboxes"], dense["labels"]): cands.append((b, short_name(l), l, "dense"))
    caption = run(img, "<MORE_DETAILED_CAPTION>")
    g = run(img, "<CAPTION_TO_PHRASE_GROUNDING>", caption)
    for b, l in zip(g["bboxes"], g["labels"]): cands.append((b, short_name(l), "", "phrase"))

    merged = []
    for box, name, desc, src in cands:
        area = (box[2] - box[0]) * (box[3] - box[1]) / (W * H)
        if not name or SKIP.match(name) or area > 0.45 or area < 0.0015: continue
        for m in merged:
            if iou(m["xyxy"], box) > 0.55:
                if src == "od" or (m["src"] != "od" and len(name) < len(m["name"])): m["name"], m["src"] = name, src
                if desc and not m["desc"]: m["desc"] = desc
                break
        else:
            merged.append({"xyxy": box, "name": name, "desc": desc, "src": src})

    # Two "fireplace"s: name the second by its description ("cast iron fireplace insert").
    names = [m["name"] for m in merged]
    for m in merged:
        if names.count(m["name"]) > 1 and m["desc"] and short_name(m["desc"]) != m["name"]: m["name"] = short_name(m["desc"])

    out = []
    for n, m in enumerate(merged):
        x1, y1, x2, y2 = m["xyxy"]
        loc = "".join(f"<loc_{int(max(0, min(999, v)))}>" for v in (x1 / W * 999, y1 / H * 999, x2 / W * 999, y2 / H * 999))
        poly = []
        try:
            seg = run(img, "<REGION_TO_SEGMENTATION>", loc, beams=1, tokens=2400)   # outlines are long; greedy is ~3x faster
            pts = max(seg["polygons"][0], key=len) if seg["polygons"] and seg["polygons"][0] else []
            poly = [[round(pts[i] / W, 4), round(pts[i + 1] / H, 4)] for i in range(0, len(pts) - 1, 2)]
        except Exception as e:  # noqa: BLE001
            print("outline failed:", m["name"], e, flush=True)
        if len(poly) < 3:                                        # fall back to the box
            poly = [[x1 / W, y1 / H], [x2 / W, y1 / H], [x2 / W, y2 / H], [x1 / W, y2 / H]]
        out.append({"id": f"o{n}", "name": m["name"], "desc": m["desc"] or m["name"],
                    "box": [round(x1 / W, 4), round(y1 / H, 4), round((x2 - x1) / W, 4), round((y2 - y1) / H, 4)], "poly": poly})
    return out, caption


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status": return self._send(200, {"ready": state["ready"], "error": state["error"]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/objects": return self._send(404, {"error": "not found"})
        if not state["ready"]: return self._send(503, {"error": state["error"] or "model still loading"})
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            raw = base64.b64decode(b["image"].split(",", 1)[1])
            key = hashlib.sha1(raw).hexdigest()[:16]
            path = os.path.join(CACHE, key + ".json")
            if os.path.exists(path): return self._send(200, {**json.load(open(path)), "cached": True})
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img.thumbnail((1600, 1600))
            t = time.time()
            with LOCK:
                objs, caption = objects(img)
            res = {"objects": objs, "caption": caption, "ms": round((time.time() - t) * 1000)}
            os.makedirs(CACHE, exist_ok=True)
            json.dump(res, open(path, "w"))
            print(f"{key}: {len(objs)} objects in {res['ms']} ms: " + ", ".join(o["name"] for o in objs), flush=True)
            self._send(200, res)
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    print(f"room objects on http://127.0.0.1:{PORT} (loading Florence-2…)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
