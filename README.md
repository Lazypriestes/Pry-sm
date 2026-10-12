# pry-sm — aesthetic foreclosure prototype

A browser study about the moment someone dismisses an object at a glance.

1. **Field** — cut-out vases drift in, 2–3 clearly visible at a time, and fade away. Participants click the ones they'd keep looking at; everything else counts as skipped.
2. **Explore** — for some skipped vases, the participant circles any part. The part is matched to **real photos with the same shape** (or, optionally, an **imagined** scene is generated around it), lined up under the part so the vase dissolves into it.
3. **Field again** — same vases, new order, to see whether choices changed.

Everything is logged (timings, clicks, circles, matches/generations) and downloaded as JSON at the end.

## Run (macOS, Apple Silicon)

```bash
# 1. shape matching (needs the local Unsplash Lite index, see below)
~/.local/share/objview-gen/.venv/bin/python tools/match_server.py
# 2. optional: local image generation for "imagined" mode (SDXL-Turbo + IP-Adapter, ~9 GB RAM)
~/.local/share/objview-gen/.venv/bin/python tools/gen_server.py
# 3. the study
node server.mjs            # → http://localhost:8001
```

Readings ("what could this part belong to") use Gemma 3 via [Ollama](https://ollama.com) (`ollama pull gemma3:4b`).
Settings (gear on the start screen) choose real matches or imagined, timings and counts.
`imagine.html` edits per-vase words and reference images (with weights) for imagined mode.

## Pages

| | |
|---|---|
| `index.html` | the study |
| `imagine.html` | per-vase words / references / weights for imagined mode |
| `analysis.html` | machine reading of the vases (outline, palette, texture) |
| `classic.html` | the earlier version of the study (browse → replay → end parts) |

## Data and images

- `images/` — vase photos and background-removed cut-outs. Vases 58+ are from the Cleveland Museum of Art Open Access (CC0); sources are recorded per vase in `catalogue.json`.
- `references/` — 400 CC0 images from The Met Open Access (used by `classic.html`).
- **Not in this repository:** the Unsplash Lite photos used for shape matching (research use only, not redistributable — download with `tools/` scripts into `~/.local/share/objview-match/`), model weights, API keys (`.env`), and private reference images (`imagine_refs/`).

Shape matching credits: Unsplash Lite Dataset 1.4.0 (unsplash.com/data).
