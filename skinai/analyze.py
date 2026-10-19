"""End-to-end analysis: image bytes -> quality report + 0-100 scores + skin age."""
from __future__ import annotations

from . import age as A
from . import face as F
from . import metrics as M
from . import quality as Q
from . import scoring as S

PIPELINE_VERSION = "0.3.0"


def analyze(data: bytes, real_age: float | None = None, with_maps: bool = False) -> dict:
    """with_maps=True adds "_maps" (per-metric float maps in the analysed image's pixel space) and "_image_hw"."""
    img = F.load_image(data)
    face = F.detect_face(img)
    q = Q.check_quality(face)
    meta = {"pipeline_version": PIPELINE_VERSION, "calibration_version": S.calibration_version()}
    if face is None or "not_enough_skin_visible" in q.issues:
        return {"ok": False, "issues": q.issues, "quality": q.stats, "confidence": q.confidence,
                "comparable": False, **meta}
    if with_maps:
        raw, maps = M.compute_raw(face, return_maps=True)
    else:
        raw, maps = M.compute_raw(face), None
    scores = S.score_raw(raw)
    face_age = A.predict_age(face.crop_face(224, margin=0.15, beard_free=True))
    out = {
        "ok": q.ok,                      # False => scores returned but unreliable, see issues
        "issues": q.issues,
        "quality": q.stats,
        "confidence": q.confidence,      # 0..1
        "comparable": q.comparable,      # False => don't compare this scan with earlier ones
        "scores": scores,
        "raw": {k: round(float(v), 4) for k, v in raw.items()},
        "face_age": round(face_age, 1),
        "skin_age": round(S.skin_age(face_age, scores, real_age), 1),
        **meta,
    }
    if with_maps:
        out["_maps"], out["_image_hw"] = maps, img.shape[:2]
    return out
