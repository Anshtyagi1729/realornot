# Spot the Fake Photo — REAL vs PHOTO-OF-A-SCREEN

Given one image, output a score in [0,1] (0 = real photo, 1 = photo of a
screen).

## Headline numbers

| metric | value |
|---|---|
| accuracy (leave-one-group-out, 5 screens + 4 real groups) | **0.815** (66/81) |
| ROC-AUC | **0.891** |
| latency, 12 MP photo, laptop CPU | **~350 ms** (of which ~110 ms decode) |
| latency, 0.5 MP image | **~120 ms** |
| model size | ~200 floats (few KB) |
| cost on-device | **$0** |
| cost cloud (CPU only) | **≈ $2–15 per million images** all-in |

## Setup (uv)
```bash
uv init
uv add numpy pillow scikit-learn joblib pillow-heif
```
(`pillow-heif` only needed if your photos are iPhone HEIC.)

## Data layout
```
data/
├── real/     <group>__<n>.<ext>   e.g. kitchen__01.jpg
└── screen/   <group>__<n>.<ext>   e.g. dell_monitor__01.jpg
```
The prefix before `__` is the **group** — one group per distinct screen and
one per scene/location. Group-disjoint validation holds whole groups out, so
the reported accuracy is on displays/scenes the model never saw. A random
split leaks and reports fake 100% on this data.

`organize.py` (if included) flattens a `raw/<class>/<group>/*` structure into
this layout automatically.

## Train
```bash
uv run python train.py --data ./data --group
```
Extracts features, trains a logistic-regression model, prints group-disjoint
validation accuracy + AUC, picks a decision threshold at ≤2% FPR, writes
`model.pkl`. Delete `features.npz` if you change the data and want to
recompute.

## Predict
```bash
uv run python predict.py some_image.jpg
# -> 0.9312
```
Prints only the score to stdout (diagnostics to stderr).

## Live demo (optional)
Runs the exact same model in the browser — same features, same weights, all
client-side, nothing uploaded. Includes a live frequency-spectrum view of the
analyzed patch, so you can *see* moiré peaks appear when the camera looks at
a screen.

```bash
uv run python export_weights.py             # -> weights.json (repo root)
mv weights.json fe/weights.json             # demo reads it from fe/
uv run python fe/serve.py                   # serves fe/ and opens the browser
```
Or on your phone (same wifi): run `uv run python fe/serve.py` on the laptop,
then open `http://<laptop-ip>:8000/index.html`. Note: camera needs HTTP(S)
served (not `file://`) and `weights.json` must sit next to `index.html`.

## Files
- `features.py` — physics feature extractor (FFT / residual / colour) on
  full-resolution patches
- `train.py` — feature extraction, training, group-disjoint validation,
  threshold selection
- `predict.py` — one-line predictor (the required entry point)
- `export_weights.py` — dumps trained model to `weights.json` for the demo
- `fe/index.html` — browser demo running the same model in JS (feature
  extraction + inference, all client-side)
- `fe/serve.py` — local dev server for `fe/`
- `organize.py` — flattens `raw/<class>/<group>/*` into the `data/` layout
