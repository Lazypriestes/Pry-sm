"""Local image generation for the study (no cloud, no cost).

SDXL-Turbo on Apple Silicon (MPS). server.mjs forwards to this when GEN_BACKEND=local.
  POST /img2img  {image: dataURL, prompt, strength?, seed?}         -> {image: dataURL, ms}
  POST /fill     {image: dataURL, mask: dataURL, prompt, seed?}     -> {image: dataURL, ms}
       mask: white = invent, black = keep
  GET  /status                                                      -> {ready, model, device}

Run with the venv made for it (outside iCloud):
  ~/.local/share/objview-gen/.venv/bin/python tools/gen_server.py
The model (~7 GB) downloads to ~/.cache/huggingface on first start.
"""
import base64, io, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting

MODEL = "stabilityai/sdxl-turbo"
PORT = 8002
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32
LOCK = threading.Lock()          # one generation at a time on the GPU
state = {"ready": False, "error": None}


def load():
    try:
        t = time.time()
        img2img = AutoPipelineForImage2Image.from_pretrained(MODEL, torch_dtype=DTYPE, variant="fp16").to(DEVICE)
        fill = AutoPipelineForInpainting.from_pipe(img2img)
        img2img.set_progress_bar_config(disable=True)
        fill.set_progress_bar_config(disable=True)
        state.update(img2img=img2img, fill=fill)
        # Warm-up so the first participant doesn't wait for kernel compilation.
        blank = Image.new("RGB", (512, 512), (128, 128, 128))
        img2img("warm up", image=blank, strength=0.5, num_inference_steps=2, guidance_scale=0.0)
        wide = Image.new("RGB", (768, 480), (128, 128, 128))                 # typical outpainting frame
        fill("warm up", image=wide, mask_image=Image.new("L", wide.size, 255), strength=0.99,
             num_inference_steps=4, guidance_scale=0.0, width=768, height=480)
        state["ready"] = True
        print(f"ready on {DEVICE} in {time.time() - t:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001 — report any load failure through /status
        state["error"] = repr(e)
        print("load failed:", e, flush=True)


def decode(data_url):
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


def encode(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fit(img, long_side):
    """Resize so the long side is `long_side`, both sides multiples of 8 (what the model needs)."""
    w, h = img.size
    k = long_side / max(w, h)
    return img.resize((max(8, round(w * k / 8) * 8), max(8, round(h * k / 8) * 8)), Image.LANCZOS)


def run(kind, b):
    gen = torch.Generator("cpu").manual_seed(int(b.get("seed") or 0) % (2**31))
    prompt = str(b.get("prompt", ""))[:400]
    if kind == "img2img":
        image = fit(decode(b["image"]).convert("RGB"), 512)
        strength = min(0.95, max(0.2, float(b.get("strength") or 0.75)))
        steps = max(2, round(2 / strength + 0.5))      # turbo: steps × strength ≥ ~2 effective steps
        out = state["img2img"](prompt, image=image, strength=strength, num_inference_steps=steps,
                               guidance_scale=0.0, generator=gen).images[0]
    else:
        image = fit(decode(b["image"]).convert("RGB"), int(b.get("size") or 768))
        mask = decode(b["mask"]).convert("L").resize(image.size)
        out = state["fill"](prompt, image=image, mask_image=mask, strength=0.99, num_inference_steps=4,
                            guidance_scale=0.0, width=image.size[0], height=image.size[1], generator=gen).images[0]
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            return self._send(200, {"ready": state["ready"], "error": state["error"], "model": MODEL, "device": DEVICE})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        kind = self.path.strip("/")
        if kind not in ("img2img", "fill"):
            return self._send(404, {"error": "not found"})
        if not state["ready"]:
            return self._send(503, {"error": state["error"] or "model still loading"})
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            t = time.time()
            with LOCK:
                out = run(kind, b)
            self._send(200, {"image": encode(out), "ms": round((time.time() - t) * 1000), "model": MODEL, "mode": kind})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    print(f"local generation on http://127.0.0.1:{PORT} (loading {MODEL}…)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
