"""Adds a "context" block to every vase in catalogue.json: era, material, origin, production, style.

Museum records (Cleveland) provide era and origin; Gemma 3 (via Ollama) looks at the photo and fills
in the rest. Anything Gemma guessed is marked "estimated" so it can be corrected by hand. Existing
values that were edited by hand ("checked": true) are never overwritten.

  ~/.local/share/objview-gen/.venv/bin/python tools/estimate_context.py
"""
import base64, io, json, os, sys, urllib.request
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROMPT = ("You are a museum cataloguer. Look at this vase and estimate, as best you can: "
          "era (a date range like 'c. 1880-1910'), material (e.g. 'glazed stoneware', 'blown glass', 'porcelain'), "
          "origin (region or culture), production (how it was made and decorated, e.g. 'wheel-thrown, carved, celadon glaze'), "
          "and style (movement or tradition). Answer as JSON with those five keys, short values.")
KEYS = ["era", "material", "origin", "production", "style"]


def ask(path, known):
    im = Image.open(path).convert("RGB"); im.thumbnail((768, 768))
    b = io.BytesIO(); im.save(b, "JPEG", quality=88)
    hint = "" if not known else " Known facts: " + "; ".join(f"{k}: {v}" for k, v in known.items()) + ". Keep them."
    req = {"model": "gemma3:4b", "prompt": PROMPT + hint, "images": [base64.b64encode(b.getvalue()).decode()],
           "stream": False, "format": "json", "keep_alive": "30m", "options": {"temperature": 0.2, "num_predict": 200}}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/generate", json.dumps(req).encode()), timeout=180))
    try:
        out = json.loads(r["response"])
    except Exception:
        return {}
    return {k: str(out.get(k, "")).strip() for k in KEYS if out.get(k)}


cat_path = os.path.join(ROOT, "catalogue.json")
cat = json.load(open(cat_path))
for n, it in enumerate(cat["items"]):
    ctx = it.get("context") or {}
    if ctx.get("checked"): continue
    src = it.get("source") or {}
    known = {k: v for k, v in {"era": src.get("date"), "origin": src.get("culture")}.items() if v}
    guess = ask(os.path.join(ROOT, it["image"]), known)
    ctx = {k: known.get(k) or guess.get(k, "") for k in KEYS}
    ctx["estimated"] = [k for k in KEYS if not known.get(k) and ctx.get(k)]
    if src.get("title"): ctx["title"] = src["title"]
    it["context"] = ctx
    print(f"{it['id']}: " + " · ".join(ctx[k] for k in KEYS if ctx.get(k)), flush=True)
    if (n + 1) % 10 == 0: json.dump(cat, open(cat_path, "w"), indent=2)
json.dump(cat, open(cat_path, "w"), indent=2)
print("done", flush=True)
