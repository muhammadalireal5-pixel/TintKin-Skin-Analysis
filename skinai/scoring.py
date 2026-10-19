"""Turn raw measures into 0-100 scores (higher = better) via percentile calibration.

calibration.json holds, per raw measure, the p5/p95 observed over a reference face set
(scripts/calibrate.py). A raw value at p5 (healthiest) scores ~100, at p95 scores ~0.

Output metrics follow the TintKin app (the four YouCam metrics it stores): wrinkles, firmness, spots,
radiance, plus an overall score. Smoothness, pores, redness and clarity are kept as extras.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CALIB_PATH = Path(__file__).resolve().parents[1] / "models" / "calibration.json"

# TintKin metric -> {raw measure: weight}. Each raw measure is calibrated (p5/p95) before weighting.
FOUR = {
    "wrinkles": {"wrinkle": 1.0},
    "spots": {"dark_spots": 0.75, "blemish": 0.25},  # blemish counts are noisy
    "radiance": {"tone_L": 0.5, "tone_chroma": 0.5},
    # firmness has no direct pixel measure; this is a proxy (cheek lines, texture, jaw softness)
    "firmness": {"wrinkle_cheek": 0.4, "texture": 0.3, "jaw_sag": 0.3},
}
# extras: raw measure -> output score name
EXTRAS = {"redness": "redness", "texture": "smoothness", "pores": "pores"}
CLARITY_PARTS = {"blemish": 0.4, "dark_spots": 0.3, "radiance_unevenness": 0.3}

# fallback ranges if no calibration file exists yet (rough, replaced by calibrate.py)
DEFAULT_CALIB = {
    "redness": [0.0, 0.6], "texture": [1.0, 4.0], "wrinkle": [0.0, 0.05],
    "pores": [0.0, 1.0], "blemish": [0.0, 1.5], "dark_spots": [0.0, 0.5],
    "radiance_unevenness": [0.05, 0.5], "tone_L": [0.1, 0.6], "tone_chroma": [0.05, 0.5],
    "wrinkle_cheek": [0.0, 0.05], "jaw_sag": [0.08, 0.5],
}
HEALTH_CENTER_KEY = "_health_center"
SHIFT_PER_POINT, SHIFT_CAP = 0.12, 5.0  # years per health point / max years; loosen after calibrating on real selfies


def load_calibration() -> dict[str, list[float]]:
    if CALIB_PATH.exists():
        return {**DEFAULT_CALIB, **json.loads(CALIB_PATH.read_text())}
    return dict(DEFAULT_CALIB)


def to_score(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 50.0
    return float(100.0 * (1.0 - np.clip((value - lo) / (hi - lo), 0.0, 1.0)))


def _weighted(raw: dict[str, float], calib: dict, parts: dict[str, float]) -> float | None:
    got = {k: w for k, w in parts.items() if k in raw}
    if not got:
        return None
    tot = sum(got.values())
    return sum(w * to_score(raw[k], *calib[k]) for k, w in got.items()) / tot


def score_raw(raw: dict[str, float], calib: dict[str, list[float]] | None = None) -> dict[str, float]:
    """-> {wrinkles, firmness, spots, radiance, overall, smoothness, pores, redness, clarity}"""
    calib = calib or load_calibration()
    out = {}
    for name, parts in FOUR.items():
        v = _weighted(raw, calib, parts)
        if v is not None:
            out[name] = v
    if out:
        out["overall"] = float(np.mean([out[k] for k in FOUR if k in out]))
    for k, name in EXTRAS.items():
        if k in raw:
            out[name] = to_score(raw[k], *calib[k])
    clarity = _weighted(raw, calib, CLARITY_PARTS)
    if clarity is not None:
        out["clarity"] = clarity
    return {k: round(float(v), 1) for k, v in out.items()}


def health(scores: dict[str, float]) -> float:
    return float(np.mean([scores.get(k, 50.0) for k in FOUR]))


def skin_age(face_age: float, scores: dict[str, float], real_age: float | None = None,
             calib: dict | None = None) -> float:
    """Apparent skin age that moves with the skin scores.

    The shift is how far the skin health (mean of the four metrics) is from the calibration set's median,
    SHIFT_PER_POINT years per point, capped at +-SHIFT_CAP years. With the user's real age the estimate is anchored to it
    (70% real age, 30% face-model age) and kept within +-10 years of it; without it, it is face age + shift.
    """
    calib = calib or load_calibration()
    center = float(calib.get(HEALTH_CENTER_KEY, [50.0])[0])
    shift = float(np.clip((center - health(scores)) * SHIFT_PER_POINT, -SHIFT_CAP, SHIFT_CAP))
    if real_age is None:
        return float(max(10.0, face_age + shift))
    base = 0.7 * real_age + 0.3 * face_age
    return float(np.clip(base + shift, max(10.0, real_age - 10.0), real_age + 10.0))


def calibration_version() -> str:
    """Short hash of calibration.json (or 'default'), stored with every result so scores can be re-run later."""
    import hashlib
    return hashlib.sha1(CALIB_PATH.read_bytes()).hexdigest()[:8] if CALIB_PATH.exists() else "default"
