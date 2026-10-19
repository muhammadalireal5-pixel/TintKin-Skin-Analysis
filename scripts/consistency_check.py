"""Score images under brightness/tint/blur/grain changes to see how stable the scores are.

Usage: python -I scripts/consistency_check.py img1.jpg [img2.png ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skinai.analyze import analyze  # noqa: E402


def variants(img: np.ndarray):
    f = img.astype(np.float32)
    yield "orig", img
    yield "bright+30%", np.clip(f * 1.3, 0, 255).astype(np.uint8)
    yield "dark-30%", np.clip(f * 0.7, 0, 255).astype(np.uint8)
    yield "smoothed(blur)", cv2.GaussianBlur(img, (0, 0), 4)
    yield "noisy(+grain)", np.clip(f + np.random.RandomState(0).normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    yield "redtint", np.clip(f * np.array([0.85, 0.9, 1.15]), 0, 255).astype(np.uint8)


def main() -> None:
    for p in sys.argv[1:]:
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        print("##", Path(p).name)
        for name, v in variants(img):
            r = analyze(cv2.imencode(".jpg", v)[1].tobytes())
            print(f"{name:16}", json.dumps(r.get("scores", r.get("issues"))))


if __name__ == "__main__":
    main()
