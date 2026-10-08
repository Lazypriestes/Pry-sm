"""Adds vases from the Cleveland Museum of Art Open Access collection (CC0) to the catalogue.

  ~/.local/share/objview-gen/.venv/bin/python tools/fetch_vases_cleveland.py [how_many=40]

Picks a varied set (materials, cultures, centuries), downloads the print-size photo, resizes it to
1400 px like the other vases, removes the background locally (tools/cutout_local.py), keeps only
objects at least 500 px tall, and appends them to catalogue.json as vase_58, vase_59, ...
"""
import io, json, os, random, sys, urllib.request
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from cutout_local import cutout  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
API = "https://openaccess-api.clevelandart.org/api/artworks/"
WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MIN_H = 500


def get(url, tries=4):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "objview-research"}), timeout=60) as r:
                return r.read()
        except Exception:
            pass
    return None


def search(kind):
    out, skip = [], 0
    while True:
        d = get(f"{API}?q=vase&has_image=1&cc0=1&type={urllib.parse.quote(kind)}&limit=100&skip={skip}")
        if not d: break
        rows = json.loads(d).get("data", [])
        out += rows; skip += 100
        if len(rows) < 100: break
    return out


import urllib.parse  # noqa: E402

cat_path = os.path.join(ROOT, "catalogue.json")
cat = json.load(open(cat_path))
have = {it.get("source", {}).get("accession") for it in cat["items"] + cat.get("excluded", [])}
nums = [int(it["id"].split("_")[1]) for it in cat["items"] + cat.get("excluded", [])]
next_n = max(nums) + 1

pool = []
for kind in ["Ceramic", "Glass", "Metalwork", "Jade", "Enamel"]:
    rows = search(kind)
    rows = [a for a in rows if "vase" in (a.get("title") or "").lower() and (a.get("images") or {}).get("print")]
    print(f"{kind}: {len(rows)} vases", flush=True)
    pool += rows
random.seed(7); random.shuffle(pool)
# Spread over cultures so the set isn't dominated by one tradition.
by_culture = {}
for a in pool:
    key = ((a.get("culture") or ["?"])[0] if isinstance(a.get("culture"), list) else a.get("culture") or "?").split(",")[0]
    by_culture.setdefault(key, []).append(a)
order = []
while any(by_culture.values()):
    for k in list(by_culture):
        if by_culture[k]: order.append(by_culture[k].pop())
added = []
for a in order:
    if len(added) >= WANT: break
    acc = a.get("accession_number")
    if acc in have: continue
    raw = get(a["images"]["print"]["url"])
    if not raw: continue
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((1400, 1400), Image.LANCZOS)
    cut = cutout(img)
    box = cut.getchannel("A").point(lambda v: 255 if v > 12 else 0).getbbox()
    if not box or box[3] - box[1] < MIN_H or (box[2] - box[0]) * (box[3] - box[1]) < 0.05 * img.size[0] * img.size[1]:
        continue
    vid = f"vase_{next_n:02d}"; next_n += 1
    img.save(os.path.join(ROOT, f"images/{vid}.jpg"), quality=90)
    cut.save(os.path.join(ROOT, f"images/cut/{vid}.png"))
    culture = a.get("culture") or []
    item = {"id": vid, "image": f"images/{vid}.jpg", "cutout": f"images/cut/{vid}.png",
            "tags": {"category": "vase", "type": a.get("type"), "material": None, "era": a.get("creation_date")},
            "object_px": [box[2] - box[0], box[3] - box[1]],
            "source": {"museum": "Cleveland Museum of Art", "license": "CC0", "accession": acc, "title": a.get("title"),
                       "culture": culture[0] if isinstance(culture, list) and culture else culture, "date": a.get("creation_date"),
                       "url": a.get("url")}}
    cat["items"].append(item); added.append(item)
    print(f"{vid}: {a.get('title')} · {item['source']['culture']} · {a.get('creation_date')} · {box[3] - box[1]}px", flush=True)

cat["version"] = f"vases-{len(cat['items'])}-v1"
json.dump(cat, open(cat_path, "w"), indent=2)
print(f"added {len(added)}; catalogue now has {len(cat['items'])} vases", flush=True)
