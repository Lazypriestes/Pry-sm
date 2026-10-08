"""Background removal on this Mac (BiRefNet, MIT licence) — the free, local replacement for tools/cutout.mjs.

  ~/.local/share/objview-gen/.venv/bin/python tools/cutout_local.py IN.jpg OUT.png [IN2.jpg OUT2.png ...]

Writes RGBA PNGs: the object on a transparent background, same size as the input.
"""
import sys, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SIZE = 1024
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_model = None


def model():
    global _model
    if _model is None:
        _model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        _model = _model.to(DEVICE).eval().half() if DEVICE == "mps" else _model.eval()
    return _model


@torch.no_grad()
def cutout(img):
    img = img.convert("RGB")
    x = torch.from_numpy(np.asarray(img.resize((SIZE, SIZE), Image.BICUBIC), dtype=np.float32) / 255.0)
    x = ((x.permute(2, 0, 1)[None] - MEAN) / STD).to(DEVICE, next(model().parameters()).dtype)
    pred = model()(x)[-1].sigmoid()[0, 0].float().cpu().numpy()
    alpha = Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)
    out = img.copy(); out.putalpha(alpha)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    for src, dst in zip(args[0::2], args[1::2]):
        t = time.time()
        cutout(Image.open(src)).save(dst)
        print(f"{dst}  {time.time() - t:.1f}s", flush=True)
