"""Input quality gate: reject photos that would give meaningless scores."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import face as F
from .face import Face


@dataclass
class QualityReport:
    ok: bool
    issues: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    confidence: float = 1.0   # 0..1; scans below COMPARABLE_CONFIDENCE shouldn't be compared with others
    comparable: bool = True


# how much each issue lowers the confidence in a score (and its usefulness for tracking over time)
ISSUE_PENALTY = {
    "face_too_small": 0.30, "blurry": 0.40, "too_dark": 0.40, "overexposed": 0.30, "uneven_lighting": 0.25,
    "face_not_frontal": 0.25, "head_tilted": 0.10, "skin_smoothed": 0.35, "multiple_faces": 0.30, "not_enough_skin_visible": 1.0,
    "no_face": 1.0,
}
COMPARABLE_CONFIDENCE = 0.7


def _report(issues: list[str], stats: dict) -> "QualityReport":
    conf = max(0.0, 1.0 - sum(ISSUE_PENALTY.get(i, 0.2) for i in issues))
    return QualityReport(not issues, issues, stats, round(conf, 2), conf >= COMPARABLE_CONFIDENCE)


def _skin_detail_ratio(face: Face) -> float:
    f = face.normalized(512)
    g = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hf = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.5))
    skin = f.skin_mask_cached() > 0
    ring = np.zeros(g.shape, np.uint8)
    for grp in (F.LEFT_EYE, F.RIGHT_EYE, F.LEFT_BROW, F.RIGHT_BROW, F.LIPS):
        h = f.poly_mask(grp)
        ring |= cv2.dilate(h, np.ones((7, 7), np.uint8)) - cv2.erode(h, np.ones((7, 7), np.uint8))
    ring = (ring > 0) & ~skin
    if not skin.any() or not ring.any():
        return 1.0
    return float(hf[skin].mean() / max(float(hf[ring].mean()), 1e-6))


def check_quality(face: Face | None) -> QualityReport:
    if face is None:
        return _report(["no_face"], {})
    issues: list[str] = []
    img = face.image
    h, w = img.shape[:2]
    x0, y0, x1, y1 = face.bbox
    face_frac = ((x1 - x0) * (y1 - y0)) / (h * w)
    if face.n_faces > 1:
        issues.append("multiple_faces")
    if face.face_width < 256:  # absolute pixels: the face is later normalised to 512 px, so < 256 means > 2x upsampling
        issues.append("face_too_small")

    mask = face.skin_mask_cached()
    if mask.sum() < 255 * 2000:
        issues.append("not_enough_skin_visible")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sel = mask > 0
    # sharpness over the whole face box (eyes, brows, lips included) at a fixed 512 px width, so smooth
    # retouched skin doesn't read as blur and the value is comparable across image sizes
    fx0, fy0, fx1, fy1 = face.bbox
    crop = gray[fy0:fy1, fx0:fx1]
    sc = 512 / max(fx1 - fx0, 1)
    crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    sharp_norm = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
    bright = float(gray[sel].mean()) if sel.any() else 0.0
    contrast = float(gray[sel].std()) if sel.any() else 0.0

    if sharp_norm < 10:  # measured: real blur (sigma 1.5) gives 3-9, sharp or retouched faces 15-600
        issues.append("blurry")
    if bright < 70:
        issues.append("too_dark")
    if bright > 215:
        issues.append("overexposed")

    # left/right lighting imbalance (strong side light distorts redness/shadows)
    mid = int(face.landmarks[168, 0])
    l, r = gray[:, :mid][mask[:, :mid] > 0], gray[:, mid:][mask[:, mid:] > 0]
    imbalance = abs(float(l.mean()) - float(r.mean())) / max(bright, 1) if l.size and r.size else 0.0
    if imbalance > 0.25:
        issues.append("uneven_lighting")

    # head pose: nose vs. cheek distances (yaw) and eyes line (roll)
    lc, rc, nose = face.landmarks[234], face.landmarks[454], face.landmarks[1]
    yaw = abs((nose[0] - lc[0]) / max(rc[0] - lc[0], 1) - 0.5)
    le, re = face.landmarks[33], face.landmarks[263]
    roll = abs(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
    if yaw > 0.12:
        issues.append("face_not_frontal")
    if roll > 15:
        issues.append("head_tilted")

    # strong beauty-filter / smoothing check: fine detail on skin vs. on eyes, brows and lips (which filters
    # keep sharp). Measured: heavily smoothed copies of 17 photos all give <= 0.24, natural photos >= 0.24
    # (mostly 0.3-0.8), so only strong smoothing (< 0.22) is flagged; mild filters pass.
    detail = _skin_detail_ratio(face)
    if detail < 0.22:
        issues.append("skin_smoothed")

    stats = dict(face_fraction=round(face_frac, 3), sharpness=round(sharp_norm, 1), skin_detail=round(detail, 3),
                 brightness=round(bright, 1), contrast=round(contrast, 1),
                 lighting_imbalance=round(imbalance, 3), yaw=round(float(yaw), 3),
                 roll=round(float(roll), 1))
    return _report(issues, stats)
