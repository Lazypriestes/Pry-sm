"""Bakes everything room.html shows into rooms/objects_content.json, so the page needs no models at all:
per object the scan (qwen2.5vl), its history and cousins with pictures (llama3.1 + Wikimedia), and
same-shape photos from the matching library, and creative references from Are.na / Unsplash: photos shaped
like each part of the object (as in the vase study), and photos that share its colours, surface and mood.

  node server.mjs  +  tools/match_server.py  must be running (Ollama too, for anything not cached yet)
  ~/.local/share/objview-gen/.venv/bin/python tools/build_objects_content.py
"""
import base64, io, json, os, urllib.request
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
API = "http://127.0.0.1:8001/api"


def post(path, body, timeout=600, tries=2):
    for t in range(tries):
        try:
            req = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:  # noqa: BLE001
            print(f"  {path} failed ({e}){' retrying' if t + 1 < tries else ''}", flush=True)
    return {}


def on_square(path, size, bg):
    im = Image.open(os.path.join(ROOT, path)).convert("RGBA")
    im.thumbnail((int(size * 0.92), int(size * 0.92)))
    sq = Image.new("RGB", (size, size), bg); sq.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
    b = io.BytesIO(); sq.save(b, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def parts(o, n=4):
    """Square crops of the object (top, middle, bottom, a detail), saved to rooms/parts/ for the page."""
    im = Image.open(os.path.join(ROOT, o["src"])).convert("RGBA")
    x0, y0, x1, y1 = im.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()
    w, h = x1 - x0, y1 - y0
    side = int(min(w, h) * (0.7 if h > 1.3 * w or w > 1.3 * h else 0.5))
    cx = x0 + w / 2
    spots = [(cx, y0 + side / 2), (cx, y0 + h / 2), (cx, y1 - side / 2), (x0 + w * 0.35, y0 + h * 0.4)]
    out = []
    os.makedirs(os.path.join(ROOT, "rooms", "parts"), exist_ok=True)
    for i, (px, py) in enumerate(spots[:n]):
        s = side if i < 3 else int(side * 0.6)
        box = (int(px - s / 2), int(py - s / 2), int(px + s / 2), int(py + s / 2))
        crop = im.crop(box)
        if sum(a > 128 for a in crop.getchannel("A").getdata()) / (s * s) < 0.35: continue
        sq = Image.new("RGB", crop.size, (255, 255, 255)); sq.paste(crop, (0, 0), crop); sq = sq.resize((256, 256))
        path = f"rooms/parts/{o['id']}_{i}.jpg"; sq.save(os.path.join(ROOT, path), quality=88)
        b = io.BytesIO(); sq.save(b, "JPEG", quality=90)
        out.append((path, "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()))
    return out


def slim(ms):
    return [{k: m[k] for k in ("id", "library", "box", "desc", "by")} for m in ms]


objs = json.load(open(os.path.join(ROOT, "rooms", "objects.json")))["objects"]
out = {}
for o in objs:
    print(o["id"], flush=True)
    scan = post("/scan", {"key": f"objects:{o['id']}", "name": o["name"], "desc": o.get("desc"), "image": on_square(o["src"], 640, (27, 27, 26))})
    lin = post("/lineage", {"name": o["name"], "desc": o.get("desc")})
    shapes = post("/match", {"image": on_square(o["src"], 256, (255, 255, 255)), "k": 8, "library": "both",
                             "exclude": o["name"], "vessels": "vase" not in o["name"].lower()}).get("matches", [])
    vessel = "vase" in o["name"].lower()
    part_refs = []
    for path, data in parts(o):
        ms = post("/match", {"image": data, "k": 2, "library": "both", "exclude": o["name"], "vessels": not vessel}).get("matches", [])
        part_refs += [{"part": path, **m} for m in slim(ms)]
    pal = ", ".join(scan.get("palette") or [])
    colours = post("/similar", {"text": f"a photo in the colours {pal}", "library": "arena", "k": 6}).get("matches", []) if pal else []
    surf = "; ".join(scan.get("surface") or [])
    surface = post("/similar", {"text": f"close-up of a surface: {surf}", "library": "arena", "k": 4}).get("matches", []) if surf else []
    mood = post("/similar", {"image": on_square(o["src"], 256, (128, 128, 128)), "minus": f"a photo of a {o['name']}",
                             "library": "arena", "k": 5}).get("matches", [])
    out[o["id"]] = {"parts": part_refs, "colours": slim(colours), "surface": slim(surface), "mood": slim(mood),"scan": scan, "lineage": lin.get("lineage", []), "cousins": lin.get("cousins", []),
                    "story": lin.get("story", ""),
                    "shapes": [{k: m[k] for k in ("id", "library", "box", "desc", "by")} for m in shapes]}
json.dump(out, open(os.path.join(ROOT, "rooms", "objects_content.json"), "w"), indent=1)
print("done", {k: sum(len(v[s]) for s in ("lineage", "cousins", "shapes", "parts", "colours", "surface", "mood")) for k, v in out.items()})
