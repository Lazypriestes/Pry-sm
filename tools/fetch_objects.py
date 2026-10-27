"""Everyday objects for room.html as cut-out PNGs: one Wikimedia Commons photo per object (picked by CLIP),
background removed with BiRefNet.

  ~/.local/share/objview-gen/.venv/bin/python tools/fetch_objects.py ["ceiling fan" "fire extinguisher" ...]

Needs tools/match_server.py running (CLIP ranking). Writes images/objects/<slug>.png and adds them to
rooms/objects.json with their source and licence. Existing entries are kept.
"""
import io, json, os, re, sys, urllib.parse, urllib.request
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from cutout_local import cutout  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "images", "objects")
LIST = os.path.join(ROOT, "rooms", "objects.json")
UA = {"User-Agent": "objview-research/0.1 (study prototype)"}
DEFAULT = ["ceiling fan", "fire extinguisher", "wooden armchair", "table lamp", "rotary telephone", "kettle"]


def get(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def commons(query, n=12):
    q = urllib.parse.quote("filetype:bitmap " + query)
    d = json.loads(get(f"https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search&gsrnamespace=6"
                       f"&gsrlimit={n}&gsrsearch={q}&prop=imageinfo&iiprop=url|extmetadata|size"
                       f"&iiextmetadatafilter=Artist|LicenseShortName&iiurlwidth=1024"))
    out = []
    for p in (d.get("query") or {}).get("pages", {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        if not ii.get("thumburl") or min(ii.get("width", 0), ii.get("height", 0)) < 500: continue
        m = ii.get("extmetadata") or {}
        out.append({"src": ii["thumburl"], "link": ii.get("descriptionurl"), "title": p["title"],
                    "artist": re.sub(r"<[^>]+>", "", (m.get("Artist") or {}).get("value", "")).strip()[:80],
                    "license": (m.get("LicenseShortName") or {}).get("value", "")})
    return out


def rank(text, urls):
    body = json.dumps({"text": text, "images": urls}).encode()
    req = urllib.request.Request("http://127.0.0.1:8003/rank", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["scores"]


def main():
    names = sys.argv[1:] or DEFAULT
    os.makedirs(OUT, exist_ok=True)
    data = json.load(open(LIST)) if os.path.exists(LIST) else {"objects": []}
    have = {o["name"] for o in data["objects"]}
    for name in names:
        if name in have: print("have", name); continue
        cands = commons(f"{name} white background") + commons(name)
        cands = list({c["src"]: c for c in cands}.values())[:20]
        if not cands: print("nothing for", name); continue
        scores = rank(f"a product photo of a single {name} on a plain white background", [c["src"] for c in cands])
        # best-ranked photo whose object is whole (does not run off the edge of the photo)
        for best in sorted(range(len(cands)), key=lambda i: -scores[i])[:6]:
            c = cands[best]
            img = Image.open(io.BytesIO(get(c["src"], 60))).convert("RGB")
            cut = cutout(img)
            a = cut.getchannel("A").load(); W, H = cut.size
            edge = [a[x, y] for x in range(0, W, 4) for y in (0, H - 1)] + [a[x, y] for y in range(0, H, 4) for x in (0, W - 1)]
            if sum(v > 128 for v in edge) / len(edge) < 0.01: break
            print(f"  {name}: skip {c['title'][:50]} (cut off at the edge)", flush=True)
        bbox = cut.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()
        if bbox: cut = cut.crop(bbox)
        slug = re.sub(r"\W+", "_", name).strip("_")
        cut.save(os.path.join(OUT, slug + ".png"))
        data["objects"].append({"id": slug, "name": name, "src": f"images/objects/{slug}.png",
                                "source": {k: c[k] for k in ("link", "title", "artist", "license")}})
        print(f"{name}: {c['title']} ({c['license']}, score {scores[best]})", flush=True)
        json.dump(data, open(LIST, "w"), indent=1)


if __name__ == "__main__":
    main()
