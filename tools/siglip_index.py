"""SigLIP 2 image embeddings for every photo in the matching libraries: the newer, stronger replacement for
CLIP (tools/clip_index.py) for "feels like", colour and surface references.

  ~/.local/share/objview-gen/.venv/bin/python tools/siglip_index.py            # all libraries
  LIBRARY=arena ~/.local/share/objview-gen/.venv/bin/python tools/siglip_index.py

Writes <index folder>/siglip.npy (one 768-number unit vector per photo, same order as photos.json).
"""
import json, os, sys, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.dirname(__file__))
from match_index import LIBRARIES, DEVICE  # noqa: E402

SIGLIP = "google/siglip2-base-patch16-224"


def features(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out


def main():
    model = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to(DEVICE).eval()
    proc = AutoProcessor.from_pretrained(SIGLIP)
    names = [os.environ["LIBRARY"]] if os.environ.get("LIBRARY") else list(LIBRARIES)
    for name in names:
        img_dir, _, idx = LIBRARIES[name]
        pj = os.path.join(idx, "photos.json")
        if not os.path.exists(pj): continue
        photos = json.load(open(pj))["photos"]
        out, t, batch = np.zeros((len(photos), 768), np.float16), time.time(), 64
        for s in range(0, len(photos), batch):
            ims = []
            for p in photos[s:s + batch]:
                try: ims.append(Image.open(os.path.join(img_dir, p["id"] + ".jpg")).convert("RGB"))
                except Exception: ims.append(Image.new("RGB", (224, 224), (128, 128, 128)))
            with torch.no_grad():
                x = proc(images=ims, return_tensors="pt")["pixel_values"].to(DEVICE, torch.float16)
                e = features(model.get_image_features(pixel_values=x)).float()
                e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
            out[s:s + len(ims)] = e.astype(np.float16)
            if (s // batch) % 50 == 0: print(f"{name}: {s + len(ims)}/{len(photos)} · {time.time() - t:.0f}s", flush=True)
        np.save(os.path.join(idx, "siglip.npy"), out)
        print(f"{name}: done {len(photos)} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
