"""Builds a second matching library from public Are.na channels (images people collected on themes
like form, shapes, texture, patterns, natural forms, architecture details).

  ~/.local/share/objview-gen/.venv/bin/python tools/import_arena.py [max_images=12000]
  ~/.local/share/objview-gen/.venv/bin/python tools/import_arena.py --channels slug-a slug-b ...

Images go to ~/.local/share/objview-match/arena/images (outside the project, never redistributed:
they belong to their makers), metadata to ~/.local/share/objview-match/arena/photos.tsv in the same
columns tools/match_index.py reads. Then index it with:
  LIBRARY=arena ~/.local/share/objview-gen/.venv/bin/python tools/match_index.py
Re-running skips images already downloaded.
"""
import csv, json, os, sys, time, urllib.parse, urllib.request, concurrent.futures as cf

ROOT = os.path.expanduser("~/.local/share/objview-match/arena")
IMG = os.path.join(ROOT, "images")
API = "https://api.are.na/v2"
QUERIES = ["form", "shapes", "texture", "patterns", "natural forms", "architecture details", "sculpture",
           "geometry", "ornament", "surfaces", "structure", "organic forms", "materials", "objects",
           "landscape", "botanical", "minerals", "folds", "repetition", "brutalism"]
MIN_BLOCKS, PER_CHANNEL = 80, 400


def get_json(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "objview-research"}), timeout=40) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1.5 * (t + 1))
    return None


def channels_for(query):
    d = get_json(f"{API}/search/channels?q={urllib.parse.quote(query)}&per=20") or {}
    return [c["slug"] for c in d.get("channels", []) if c.get("length", 0) >= MIN_BLOCKS and c.get("status") in ("public", "closed")]


def image_blocks(slug, limit=PER_CHANNEL):
    out, page = [], 1
    while len(out) < limit:
        d = get_json(f"{API}/channels/{slug}/contents?per=100&page={page}")
        rows = (d or {}).get("contents") or []
        if not rows: break
        for b in rows:
            im = b.get("image") or {}
            url = (im.get("display") or {}).get("url") or (im.get("large") or {}).get("url")
            if b.get("class") == "Image" and url:
                out.append({"id": f"arena_{b['id']}", "url": url, "title": (b.get("title") or b.get("generated_title") or "").strip(),
                            "desc": (b.get("description") or "").strip(), "channel": slug,
                            "by": ((b.get("source") or {}).get("provider") or {}).get("name") or (b.get("user") or {}).get("full_name", ""),
                            "link": f"https://www.are.na/block/{b['id']}"})
        page += 1
    return out[:limit]


def download(b):
    path = os.path.join(IMG, b["id"] + ".jpg")
    if os.path.exists(path) and os.path.getsize(path) > 2000: return True
    try:
        with urllib.request.urlopen(urllib.request.Request(b["url"], headers={"User-Agent": "objview-research"}), timeout=40) as r:
            data = r.read()
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if min(im.size) < 300: return False                      # too small to match on
        im.thumbnail((640, 640)); im.save(path, quality=85)
        return True
    except Exception:
        return False


def main():
    os.makedirs(IMG, exist_ok=True)
    args = sys.argv[1:]
    if args[:1] == ["--channels"]:
        slugs, cap = args[1:], 10**9
    else:
        cap = int(args[0]) if args else 12000
        slugs = list(dict.fromkeys(s for q in QUERIES for s in channels_for(q)))
    print(f"{len(slugs)} channels", flush=True)
    blocks, seen = [], set()
    for s in slugs:
        if len(blocks) >= cap: break
        for b in image_blocks(s):
            if b["id"] not in seen: seen.add(b["id"]); blocks.append(b)
        print(f"  {s}: total {len(blocks)}", flush=True)
    blocks = blocks[:cap]
    ok = []
    with cf.ThreadPoolExecutor(16) as ex:
        for b, good in zip(blocks, ex.map(download, blocks)):
            if good: ok.append(b)
            if len(ok) % 1000 == 0 and good: print(f"downloaded {len(ok)}", flush=True)
    # Same columns as Unsplash Lite's photos.tsv, so the indexer reads both the same way.
    with open(os.path.join(ROOT, "photos.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["photo_id", "photo_url", "photo_description", "ai_description", "photographer_first_name", "photographer_last_name", "channel", "image_url"])
        for b in ok:
            w.writerow([b["id"], b["link"], b["title"] or b["desc"][:120], b["desc"][:200], b["by"], "", b["channel"], b["url"]])
    print(f"done: {len(ok)} images from {len(slugs)} channels", flush=True)


if __name__ == "__main__":
    main()
