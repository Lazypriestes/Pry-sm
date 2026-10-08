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
from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoencoderKL

MODEL = "stabilityai/sdxl-turbo"
# SDXL's own VAE has to run in float32 (it overflows in fp16), which roughly doubled memory use.
# This drop-in VAE is fixed to run in fp16.
VAE = "madebyollin/sdxl-vae-fp16-fix"
PORT = 8002
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32
LOCK = threading.Lock()          # one generation at a time on the GPU
state = {"ready": False, "error": None}


def load():
    try:
        t = time.time()
        vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=DTYPE)
        img2img = AutoPipelineForImage2Image.from_pretrained(MODEL, vae=vae, torch_dtype=DTYPE, variant="fp16").to(DEVICE)
        # (attention slicing would save memory but replaces IP-Adapter's attention layers, so it's off)
        # IP-Adapter: lets a reference image steer the look (per-vase references, imagine.json).
        # Loaded before the inpainting pipe is made so both share it. Scale 0 = no reference.
        img2img.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                                weight_name="ip-adapter_sdxl_vit-h.safetensors", image_encoder_folder="models/image_encoder")
        img2img.set_ip_adapter_scale(0.0)
        fill = AutoPipelineForInpainting.from_pipe(img2img)
        img2img.set_progress_bar_config(disable=True)
        fill.set_progress_bar_config(disable=True)
        state.update(img2img=img2img, fill=fill)
        # Warm-up so the first participant doesn't wait for kernel compilation.
        blank = Image.new("RGB", (512, 512), (128, 128, 128))
        img2img("warm up", image=blank, strength=0.5, num_inference_steps=2, guidance_scale=0.0, ip_adapter_image=blank)
        wide = Image.new("RGB", (1280, 800), (128, 128, 128))                # typical outpainting frame
        fill("warm up", image=wide, mask_image=Image.new("L", wide.size, 255), strength=0.99,
             num_inference_steps=6, guidance_scale=0.0, width=1280, height=800, ip_adapter_image=blank)
        free_gpu()
        state["ready"] = True
        print(f"ready on {DEVICE} in {time.time() - t:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001 — report any load failure through /status
        state["error"] = repr(e)
        print("load failed:", e, flush=True)


def free_gpu():
    """Hand cached GPU memory back to macOS after each image (otherwise the MPS cache only grows)."""
    if DEVICE == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


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


NO_REF = Image.new("RGB", (224, 224), (128, 128, 128))


def reference(b):
    """The reference image for this request and how strongly it steers (0-1). No reference = scale 0."""
    ref = b.get("ref") or {}
    if isinstance(ref.get("image"), str) and ref["image"].startswith("data:image/"):
        return decode(ref["image"]).convert("RGB"), min(1.0, max(0.0, float(ref.get("weight", 0.5))))
    return NO_REF, 0.0


def run(kind, b):
    gen = torch.Generator("cpu").manual_seed(int(b.get("seed") or 0) % (2**31))
    prompt = str(b.get("prompt", ""))[:400]
    ref_img, ref_scale = reference(b)
    state["img2img"].set_ip_adapter_scale(ref_scale)
    if kind == "img2img":
        image = fit(decode(b["image"]).convert("RGB"), 512)
        strength = min(0.95, max(0.2, float(b.get("strength") or 0.75)))
        steps = max(2, round(2 / strength + 0.5))      # turbo: steps × strength ≥ ~2 effective steps
        out = state["img2img"](prompt, image=image, strength=strength, num_inference_steps=steps,
                               guidance_scale=0.0, generator=gen, ip_adapter_image=ref_img).images[0]
    else:
        image = fit(decode(b["image"]).convert("RGB"), min(1536, int(b.get("size") or 1280)))
        mask = decode(b["mask"]).convert("L").resize(image.size)
        steps = min(12, max(2, int(b.get("steps") or 6)))
        out = state["fill"](prompt, image=image, mask_image=mask, strength=0.99, num_inference_steps=steps,
                            guidance_scale=0.0, width=image.size[0], height=image.size[1], generator=gen,
                            ip_adapter_image=ref_img).images[0]
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
                try:
                    out = run(kind, b)
                finally:
                    free_gpu()
            self._send(200, {"image": encode(out), "ms": round((time.time() - t) * 1000), "model": MODEL, "mode": kind})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    print(f"local generation on http://127.0.0.1:{PORT} (loading {MODEL}…)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
