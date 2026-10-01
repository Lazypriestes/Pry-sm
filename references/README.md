# Reference library

400 public-domain (CC0) images from The Metropolitan Museum of Art Open Access,
20 per theme, used by the "Circle + references" end part.

- `img/` — the images (resized to 480px)
- `meta.json` — title, date, culture, medium, department, Met URL per image
- `index.json` — the same plus a CLIP embedding per image (Xenova/clip-vit-base-patch32, q8)
- `vases.json` — a CLIP embedding per catalogue object

To rebuild or extend (from the prototype folder):

    node tools/fetch-met.mjs 20          # downloads; re-running skips what is there
    cd tools && npm install && node build-index.mjs

Themes are listed at the top of `tools/fetch-met.mjs`. Images of your own can be added by
appending entries to `meta.json` (with `id`, `file`, `theme`, `title`) and rebuilding the index.
