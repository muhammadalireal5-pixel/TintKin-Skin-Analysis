"""Deterministic pixel-based skin measures (raw values; lower = healthier).

All measures are computed on a face normalised to a fixed width and are made independent of the
shot's lighting: the skin's median colour is balanced to neutral, L is flattened against a smooth
illumination gradient and rescaled so the median skin lightness is 60. They are therefore relative
to the person's own skin, so skin colour and exposure don't move them.
Raw values are turned into 0-100 scores in scoring.py via calibration.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import face as F

L_REF = 60.0  # median skin lightness after normalisation
CORE_ERODE = 15  # px at 512 wide: how far in from the skin-mask border the spot/tone measures start
EVAL_ERODE = 11  # extra px the spot/tone statistics are taken inside that core


def _neutralize(img_u8: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Float BGR 0..1 with the skin median made neutral (von Kries in linear light, luminance kept)."""
    lin = (img_u8.astype(np.float32) / 255.0) ** 2.2
    med = np.maximum(np.array([np.median(lin[..., c][sel]) for c in range(3)], np.float32), 1e-4)
    target = float(np.exp(np.log(med).mean()))
    gains = np.clip(target / med, 0.6, 1.6)
    return (np.clip(lin * gains, 0, 1) ** (1 / 2.2)).astype(np.float32)


def _flatten(L: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide out a smooth (quadratic) lighting gradient fitted on skin, rescale so the median skin L is L_REF.

    Returns (flattened L, per-pixel factor that was applied) so colour channels can be scaled the same way."""
    h, w = L.shape
    ys, xs = np.nonzero(sel)
    if len(ys) > 30000:
        k = np.random.default_rng(0).choice(len(ys), 30000, replace=False)
        ys, xs = ys[k], xs[k]

    def basis(x, y):
        x, y = x / w - 0.5, y / h - 0.5
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)

    coef, *_ = np.linalg.lstsq(basis(xs.astype(np.float32), ys.astype(np.float32)),
                               np.log(L[ys, xs] + 5.0), rcond=None)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    fit = basis(gx, gy) @ coef
    factor = 1.0 / np.exp(fit)
    flat = (L + 5.0) * factor
    k = L_REF / max(float(np.median(flat[sel])), 1e-4)
    return flat * k, factor * k


def _prep(f: F.Face):
    """(skin mask, bool selector, flattened L, a*, b*) for an already size-normalised face."""
    skin = f.skin_mask_cached()
    sel = skin > 0
    lab = cv2.cvtColor(_neutralize(f.image, sel), cv2.COLOR_BGR2Lab)
    L, factor = _flatten(lab[..., 0], sel)
    # chroma grows with lightness, so scale a*/b* by the same factor as L: exposure then doesn't move colour measures
    return skin, sel, L, lab[..., 1] * factor, lab[..., 2] * factor


def _masked_blur(x: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that ignores pixels outside the mask (normalised convolution)."""
    mf = (m > 0).astype(np.float32)
    num = cv2.GaussianBlur(x * mf, (0, 0), sigma)
    den = cv2.GaussianBlur(mf, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)


def _ridge(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Dark-line (valley) response from Hessian eigenvalues, a Frangi-like filter."""
    g = cv2.GaussianBlur(gray, (0, 0), sigma)
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    tmp = np.sqrt(((dxx - dyy) / 2) ** 2 + dxy ** 2)
    l1 = (dxx + dyy) / 2 + tmp  # larger eigenvalue (positive for dark valleys)
    l2 = (dxx + dyy) / 2 - tmp
    resp = np.where(l1 > 0, l1 - np.abs(l2) * 0.5, 0)
    return np.maximum(resp, 0) * sigma ** 2


NOSE = [168, 193, 122, 129, 98, 97, 2, 326, 327, 358, 351, 417]


def _core(f: F.Face, skin: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Interior skin for tone/spot measures: the mask border (hairline, beard line, eyelids, nostrils, lips) is where
    hair shadow and 3D shading sit, and detectors fire there rather than on pigmentation. Also drops the under-eye
    (dark circles are not pigment spots) and the nose (nostril/side shading)."""
    k = lambda n: np.ones((n, n), np.uint8)
    eyes = cv2.dilate(f.poly_mask(F.LEFT_UNDER_EYE) | f.poly_mask(F.RIGHT_UNDER_EYE), k(21)) > 0
    nose = cv2.dilate(f.poly_mask(NOSE), k(9)) > 0
    interior = cv2.erode(sel.astype(np.uint8), k(CORE_ERODE)) > 0
    core = sel & interior & ~eyes & ~nose
    return core if core.sum() > 4000 else sel  # tiny faces/occlusion: fall back to the whole mask


def compute_raw(face: F.Face, return_maps: bool = False):
    """Raw measures (and, with return_maps, per-metric 0..1 float maps over the size-normalised image)."""
    f = face.normalized(512)
    skin, sel, L, a, b = _prep(f)
    area = max(int(sel.sum()), 1)
    core = _core(f, skin, sel)
    skin_core = skin * core
    # measured further in than the blurs use: shadow from the nose wing / brows / beard bleeds a few px past the core edge
    inner = cv2.erode(core.astype(np.uint8), np.ones((EVAL_ERODE, EVAL_ERODE), np.uint8)) > 0
    if inner.sum() < 3000:
        inner = core

    # --- Redness (internal, not a TintKin metric): excess a* over the person's own median ---
    a_s = _masked_blur(a, skin, 3)
    a_med = float(np.median(a_s[sel]))
    excess = np.clip(a_s - (a_med + 3.0), 0, None)
    redness = 0.5 * float((excess[sel] > 0).mean()) + 0.5 * float(excess[sel].mean() / 8.0)

    # --- Dark spots: patches darker than their neighbourhood (pigmentation) ---
    bg = _masked_blur(L, skin_core, 25)
    fine = _masked_blur(L, skin_core, 2.0)
    depth = np.clip(bg - fine - 4.0, 0, None)
    # blobs only: remove thin lines (wrinkles/hair) with an opening
    spot = cv2.morphologyEx((depth > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    spot_depth = depth * spot
    dark_spots = float(spot[inner].mean()) + float(spot_depth[inner].mean() / 10.0)

    # --- Blemishes (pimples): small, sharp red-or-dark blobs ---
    # median-filtered first: JPEG blocking and sensor noise make 1-3 px specks that are not blemishes
    a_m, L_m = cv2.medianBlur(a.astype(np.float32), 5), cv2.medianBlur(L.astype(np.float32), 5)
    blem_signal = (cv2.GaussianBlur(a_m, (0, 0), 1.5) - cv2.GaussianBlur(a_m, (0, 0), 6)) * 1.0 \
        + (cv2.GaussianBlur(L_m, (0, 0), 6) - cv2.GaussianBlur(L_m, (0, 0), 1.5)) * 0.3
    blem = ((blem_signal > 4.0) & sel).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(blem)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    sizes = sizes[(sizes >= 6) & (sizes <= 400)]
    blemish = float(len(sizes)) / (area / 10000.0) / 10.0  # blobs per 10k skin px, scaled

    # --- Radiance: uniform, luminous tone = low-frequency L and colour variation (lighting gradient removed) ---
    L_low = _masked_blur(L, skin_core, 12)
    a_low = _masked_blur(a, skin_core, 12)
    b_low = _masked_blur(b, skin_core, 12)
    tone_L = float(L_low[inner].std() / max(L_low[inner].mean(), 1)) * 4
    tone_chroma = float(np.hypot(a_low[inner].std(), b_low[inner].std()) / 15.0) * 2
    radiance_unevenness = tone_L + tone_chroma

    # --- Wrinkles: ridge density in forehead, under-eye, cheek (nasolabial) zones ---
    gray = L / 100.0
    resp = np.maximum.reduce([_ridge(gray, s) for s in (1.5, 2.5, 3.5)])
    zones = {
        "forehead": F.FOREHEAD, "eye_l": F.LEFT_UNDER_EYE, "eye_r": F.RIGHT_UNDER_EYE,
        "cheek_l": F.LEFT_CHEEK, "cheek_r": F.RIGHT_CHEEK,
    }
    wr = {}
    for name, idx in zones.items():
        zm = cv2.bitwise_and(f.poly_mask(idx), skin) > 0
        wr[name] = float(resp[zm].mean()) if zm.sum() > 200 else 0.0
    wrinkle = 0.35 * wr["forehead"] + 0.25 * (wr["eye_l"] + wr["eye_r"]) / 2 \
        + 0.4 * (wr["cheek_l"] + wr["cheek_r"]) / 2
    wrinkle_cheek = (wr["cheek_l"] + wr["cheek_r"]) / 2

    # --- Texture + pores need real detail: measured on a face kept at native resolution (512..1024 px
    # wide, never upsampled past its source) with sigmas/areas scaled so they cover the same physical size ---
    dw = int(np.clip(face.face_width, 512, 1024))
    fd = face.normalized(dw) if dw != 512 else f
    ks = dw / 512.0
    if fd is f:
        skin_d, sel_d, Ld = skin, sel, L
    else:
        skin_d, sel_d, Ld, _, _ = _prep(fd)

    # --- Texture roughness (pores/evenness) for firmness/age context ---
    hf = Ld - cv2.GaussianBlur(Ld, (0, 0), 2.5 * ks)
    texture = float(np.abs(hf[sel_d]).mean())

    # --- Pores: small dark DoG blobs on cheeks and nose, as density per 10k (512-equivalent) px ---
    nose = [168, 193, 122, 129, 98, 97, 2, 326, 327, 358, 351, 417]
    pore_m = (cv2.bitwise_and(fd.poly_mask(F.LEFT_CHEEK) | fd.poly_mask(F.RIGHT_CHEEK) | fd.poly_mask(nose), skin_d)) > 0
    dog = cv2.GaussianBlur(Ld, (0, 0), 3.0 * ks) - cv2.GaussianBlur(Ld, (0, 0), 1.2 * ks)
    pore_blobs = ((dog > 1.5) & pore_m).astype(np.uint8)
    ok = max(2, int(round(2 * ks)))
    pore_blobs = cv2.morphologyEx(pore_blobs, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8))
    pn, _, pstats, _ = cv2.connectedComponentsWithStats(pore_blobs)
    psz = pstats[1:, cv2.CC_STAT_AREA] / ks ** 2
    psz = psz[(psz >= 3) & (psz <= 60)]
    pores = float(len(psz)) / max(pore_m.sum() / ks ** 2 / 10000.0, 1.0) / 20.0

    # --- Jawline definition: sagging proxy (edge strength along lower contour) ---
    jaw_pts = f.landmarks[F.JAW].astype(np.int32)
    gx = cv2.Sobel(cv2.GaussianBlur(L, (0, 0), 2), cv2.CV_32F, 0, 1, ksize=3)
    H, W = gx.shape
    jaw_edge = float(np.mean([abs(gx[min(max(y + dy, 0), H - 1), min(max(x, 0), W - 1)])
                              for x, y in jaw_pts for dy in (3, 6, 9)])) / L_REF * 60.0  # relative, averaged over a band
    jaw_sag = 1.0 / (1.0 + jaw_edge)  # lower = firmer

    raw = dict(redness=redness, dark_spots=dark_spots, blemish=blemish,
               radiance_unevenness=radiance_unevenness, tone_L=tone_L, tone_chroma=tone_chroma,
               wrinkle=wrinkle, wrinkle_cheek=wrinkle_cheek, pores=pores,
               wrinkle_forehead=wr["forehead"], texture=texture,
               jaw_definition=jaw_edge, jaw_sag=jaw_sag, skin_area_px=float(area))
    if not return_maps:
        return raw

    zone_any = np.zeros(sel.shape, np.uint8)
    for idx in zones.values():
        zone_any |= f.poly_mask(idx)
    cheeks = f.poly_mask(F.LEFT_CHEEK) | f.poly_mask(F.RIGHT_CHEEK)
    jaw_band = cv2.dilate(f.poly_mask(F.JAW), np.ones((9, 9), np.uint8))
    maps = {
        "wrinkle": np.clip(resp / 0.04, 0, 1) * (cv2.bitwise_and(zone_any, skin) > 0),
        "age_spot": np.clip(spot_depth / 12.0, 0, 1) * sel,
        "radiance": np.clip(np.abs(L_low - float(np.median(L_low[sel]))) / 10.0, 0, 1) * sel,
        "firmness": np.clip(resp / 0.04, 0, 1) * (cv2.bitwise_and(cheeks | jaw_band, skin) > 0),
    }
    return raw, {k: v.astype(np.float32) for k, v in maps.items()}
