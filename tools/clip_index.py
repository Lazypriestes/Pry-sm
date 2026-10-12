"""CLIP image embeddings for every photo in the matching libraries ("what is this a picture of"),
so matches can lean towards photos that fit a vase's context (era, material, origin, the whole vase).

  ~/.local/share/objview-gen/.venv/bin/python tools/clip_index.py            # all libraries
  LIBRARY=arena ~/.local/share/objview-gen/.venv/bin/python tools/clip_index.py

Writes <index folder>/clip.npy (one 512-number unit vector per photo, same order as photos.json).
"""
import json, os, sys, time
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(__file__))
from match_index import LIBRARIES, DEVICE  # noqa: E402

CLIP = "openai/clip-vit-base-patch32"


def main():
    model = CLIPModel.from_pretrained(CLIP, torch_dtype=torch.float16).to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(CLIP)
    names = [os.environ["LIBRARY"]] if os.environ.get("LIBRARY") else list(LIBRARIES)
    for name in names:
        img_dir, _, idx = LIBRARIES[name]
        pj = os.path.join(idx, "photos.json")
        if not os.path.exists(pj): continue
        photos = json.load(open(pj))["photos"]
        out, t, batch = np.zeros((len(photos), 512), np.float16), time.time(), 64
        for s in range(0, len(photos), batch):
            ims = []
            for p in photos[s:s + batch]:
                try: ims.append(Image.open(os.path.join(img_dir, p["id"] + ".jpg")).convert("RGB"))
                except Exception: ims.append(Image.new("RGB", (224, 224), (128, 128, 128)))
            with torch.no_grad():
                x = proc(images=ims, return_tensors="pt")["pixel_values"].to(DEVICE, torch.float16)
                e = model.visual_projection(model.vision_model(pixel_values=x).pooler_output).float()
                e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
            out[s:s + len(ims)] = e.astype(np.float16)
            if (s // batch) % 50 == 0: print(f"{name}: {s + len(ims)}/{len(photos)} · {time.time() - t:.0f}s", flush=True)
        np.save(os.path.join(idx, "clip.npy"), out)
        print(f"{name}: done {len(photos)} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
