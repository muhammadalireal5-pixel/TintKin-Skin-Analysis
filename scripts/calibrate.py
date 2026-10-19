"""Fit raw-measure -> score ranges (p5/p95) over reference faces; writes models/calibration.json.

Uses the FairFace 1.25-padding images if present (larger faces), else the 0.25 validation set.
Usage: python scripts/calibrate.py [--n 1500]
       python scripts/calibrate.py --dir test [--out models/calibration_preview.json]   (real photos instead of FairFace)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skinai import face as F, metrics as M, scoring as S  # noqa: E402

KEYS = ["redness", "texture", "wrinkle", "pores", "blemish", "dark_spots", "radiance_unevenness",
        "tone_L", "tone_chroma", "wrinkle_cheek", "jaw_sag"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--dir", type=Path, default=None, help="folder of real photos (jpg/png/webp) to calibrate on")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "calibration.json")
    args = ap.parse_args()
    if args.dir:
        sources = sorted(p for e in ("*.jpg", "*.jpeg", "*.png", "*.webp") for p in args.dir.glob(e))
        images = (F.load_image(str(p)) for p in sources)
    else:
        files = sorted(glob.glob(str(ROOT / "data/fairface/1.25/validation-*.parquet"))) \
            or sorted(glob.glob(str(ROOT / "data/fairface/0.25/validation-*.parquet")))
        df = pd.read_parquet(files[0]).sample(args.n, random_state=0)
        images = (cv2.resize(cv2.imdecode(np.frombuffer(d["bytes"], np.uint8), cv2.IMREAD_COLOR), (768, 768),
                             interpolation=cv2.INTER_CUBIC) for d in df["image"])
    rows = []
    for img in images:
        face = F.detect_face(img)
        if face is None:
            continue
        try:
            rows.append(M.compute_raw(face))
        except Exception:  # noqa: BLE001
            continue
    print(f"calibrated on {len(rows)} faces")
    calib = {}
    for k in KEYS:
        v = np.array([r[k] for r in rows], float)
        lo, hi = np.percentile(v, [5, 95])
        calib[k] = [float(lo), float(hi)]
        print(f"{k:22s} p5={lo:.4f} p95={hi:.4f}")
    # skin-age centre: median skin health (mean of the four TintKin metrics) of the reference set, so an
    # average face gets no skin-age shift
    h = [S.health(S.score_raw(r, {**S.DEFAULT_CALIB, **calib})) for r in rows]
    calib[S.HEALTH_CENTER_KEY] = [float(np.median(h)), float(np.median(h))]
    print(f"health center (median)  {calib[S.HEALTH_CENTER_KEY][0]:.1f}")
    args.out.write_text(json.dumps(calib, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
