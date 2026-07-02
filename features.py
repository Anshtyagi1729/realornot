"""
features.py — physically-motivated features for REAL vs PHOTO-OF-A-SCREEN.

Design principle: encode the *recapture process*, not the scene content, so the
classifier generalises to unseen phones / screens / scenes.

We work on FULL-RESOLUTION patches (never a globally downsampled image), because
the strongest signal — moire from screen-grid x sensor-grid aliasing — lives in
high spatial frequencies that downsampling destroys.

Per patch we compute:
  1. Frequency (FFT) features:
       - radial power-spectrum band ratios (low / mid / high energy)
       - "peakiness": strongest anomalous periodic peak after radial whitening
         -> moire (LCD), subpixel structure, or halftone (printout) all show up
       - number of such peaks
  2. High-pass residual features (patch minus Gaussian-blurred patch):
       - std / mean-abs / kurtosis of the residual (sensor-noise character)
       - residual autocorrelation peak (screen grid is periodic -> peak > 0-lag)
  3. Colour features:
       - saturation stats, fraction near-clipped, banding / gamut compression
         (screens posterise; recaptures lose unique colours)

Features are aggregated over patches (mean / max / p90) into one image vector.
Only numpy + Pillow are required — no scipy, no NN runtime.
"""

import numpy as np
import pillow_heif
from PIL import Image, ImageFilter

pillow_heif.register_heif_opener()
# ---- config -----------------------------------------------------------------
PATCH = 256  # patch side in NATIVE pixels (must stay full-res)
MAX_PATCHES = 10  # cap for speed; features are aggregated over the image
SAFETY_SIDE = 6000  # only shrink absurdly large images (memory guard);
# normal phone photos pass through at NATIVE resolution so
# high-frequency moire/grid signal is preserved intact
RADIAL_BINS = 40
RNG = np.random.default_rng(0)


def _load_rgb(path):
    img = Image.open(path).convert("RGB")
    # EXIF orientation
    try:
        from PIL import ImageOps

        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    w, h = img.size
    scale = max(w, h) / SAFETY_SIDE
    if scale > 1.0:  # only for pathological inputs; rare
        img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)  # H,W,3 in [0,255]


def _luma(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _patch_coords(h, w):
    """Grid of patch top-left coords covering the image, capped to MAX_PATCHES."""
    p = min(PATCH, h, w)
    if p < 32:
        return [(0, 0, h, w)]
    ys = list(range(0, max(1, h - p + 1), p))
    xs = list(range(0, max(1, w - p + 1), p))
    if ys[-1] != h - p:
        ys.append(h - p)
    if xs[-1] != w - p:
        xs.append(w - p)
    coords = [(y, x, y + p, x + p) for y in ys for x in xs]
    if len(coords) > MAX_PATCHES:
        idx = RNG.choice(len(coords), MAX_PATCHES, replace=False)
        coords = [coords[i] for i in idx]
    return coords


_GEO = {}  # per-shape cache: radius bins, counts, window, axis mask


def _geometry(shape):
    g = _GEO.get(shape)
    if g is not None:
        return g
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    rr = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_idx = (rr / rr.max() * (RADIAL_BINS - 1)).astype(np.int32)
    r_flat = r_idx.ravel()
    cnt = np.maximum(np.bincount(r_flat, minlength=RADIAL_BINS), 1)
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    mask = np.ones((h, w), dtype=bool)
    icy, icx, band = h // 2, w // 2, 6
    mask[icy - band : icy + band + 1, :] = False
    mask[:, icx - band : icx + band + 1] = False
    g = (r_idx, r_flat, cnt, win, mask)
    _GEO[shape] = g
    return g


def _radial_profile(mag, r_flat, cnt):
    prof = np.bincount(r_flat, mag.ravel(), minlength=RADIAL_BINS) / cnt
    return prof


def _fft_features(gray, geo):
    r_idx, r_flat, cnt, win, axmask = geo
    g = (gray - gray.mean()) * win  # Hann window suppresses edge leakage
    F = np.fft.fftshift(np.fft.fft2(g))
    logmag = np.log1p(np.abs(F))

    prof = _radial_profile(logmag, r_flat, cnt)

    # band energy ratios (exclude DC bin 0)
    total = prof[1:].sum() + 1e-8
    low = prof[1 : RADIAL_BINS // 4].sum()
    mid = prof[RADIAL_BINS // 4 : RADIAL_BINS // 2].sum()
    high = prof[RADIAL_BINS // 2 :].sum()

    # Whiten: divide magnitude by the mean magnitude at its radius, so a bright
    # *periodic* peak (moire / grid / halftone) stands out above the natural
    # 1/f falloff. Natural photos are smooth here; recaptures spike.
    expected = prof[np.clip(r_idx, 0, RADIAL_BINS - 1)]
    whitened = logmag / (expected + 1e-6)

    # ignore DC neighbourhood and the axes (JPEG blockiness lives on the axes)
    vals = whitened[axmask]

    peak = float(vals.max())
    p999 = float(np.percentile(vals, 99.9))
    n_peaks = float((vals > (vals.mean() + 5 * vals.std())).sum())

    return np.array(
        [
            low / total,
            mid / total,
            high / total,
            (high + mid) / (low + 1e-6),
            peak,
            p999,
            np.log1p(n_peaks),
            float(prof[1:].std() / (prof[1:].mean() + 1e-6)),
        ],
        dtype=np.float32,
    )


def _residual_features(gray):
    im = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32)
    res = gray - blur

    r = res.ravel()
    std = float(r.std())
    mad = float(np.mean(np.abs(r)))
    m = r.mean()
    s = r.std() + 1e-6
    kurt = float(np.mean(((r - m) / s) ** 4) - 3.0)

    # autocorrelation of the residual via FFT; a periodic screen grid produces a
    # secondary peak away from the zero-lag spike. Use a centre crop — periodicity
    # is global, so a 128px window captures it at 1/4 the FFT cost.
    c = res[:128, :128] if min(res.shape) >= 128 else res
    R = c - c.mean()
    Fc = np.fft.fft2(R)
    ac = np.fft.fftshift(np.fft.ifft2(Fc * np.conj(Fc)).real)
    ac = ac / (ac.max() + 1e-8)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    ac[cy - 3 : cy + 4, cx - 3 : cx + 4] = 0.0  # remove zero-lag spike
    ac_peak = float(ac.max())

    return np.array([std, mad, kurt, ac_peak], dtype=np.float32)


def _color_features(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)

    clipped = float(np.mean(mx > 250))  # blown highlights (screen glare)
    dark = float(np.mean(mx < 12))

    # banding / gamut compression: unique 6-bit-per-channel colours, normalised.
    # Subsample (every 2nd px) — this is a ratio estimate, full resolution is waste.
    q = (rgb[::2, ::2] / 4).astype(np.int32)
    codes = (q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]
    uniq = np.unique(codes).size / codes.size  # low -> posterised (screen-ish)

    return np.array(
        [
            float(sat.mean()),
            float(sat.std()),
            clipped,
            dark,
            uniq,
        ],
        dtype=np.float32,
    )


def _patch_vector(rgb_patch):
    gray = _luma(rgb_patch)
    geo = _geometry(gray.shape)
    return np.concatenate(
        [
            _fft_features(gray, geo),
            _residual_features(gray),
            _color_features(rgb_patch),
        ]
    )


_PATCH_DIM = 8 + 4 + 5  # keep in sync with the three blocks above


def _agg_names():
    base = (
        [
            f"fft_{n}"
            for n in [
                "low",
                "mid",
                "high",
                "hi_lo",
                "peak",
                "p999",
                "npeaks",
                "prof_cv",
            ]
        ]
        + [f"res_{n}" for n in ["std", "mad", "kurt", "acpeak"]]
        + [f"col_{n}" for n in ["sat_m", "sat_s", "clip", "dark", "uniq"]]
    )
    names = []
    for stat in ("mean", "max", "p90"):
        names += [f"{stat}__{b}" for b in base]
    return names


FEATURE_NAMES = _agg_names()
FEATURE_DIM = len(FEATURE_NAMES)


def extract_features(path):
    """Return a 1-D float32 feature vector for one image (aggregated over patches)."""
    rgb = _load_rgb(path)
    h, w = rgb.shape[:2]
    vecs = []
    for y0, x0, y1, x1 in _patch_coords(h, w):
        vecs.append(_patch_vector(rgb[y0:y1, x0:x1]))
    V = np.stack(vecs, axis=0)  # (n_patches, PATCH_DIM)
    agg = np.concatenate(
        [
            V.mean(axis=0),
            V.max(axis=0),
            np.percentile(V, 90, axis=0),
        ]
    ).astype(np.float32)
    agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0)
    return agg


if __name__ == "__main__":
    import sys

    v = extract_features(sys.argv[1])
    print("dim:", v.shape[0])
    for n, x in zip(FEATURE_NAMES, v):
        print(f"{n:22s} {x: .4f}")
