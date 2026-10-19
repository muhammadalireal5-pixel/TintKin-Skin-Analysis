"""CPU age inference with the exported ONNX model (FairFace-trained MobileNetV3)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "age.onnx"
MIDPOINTS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], np.float32)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_sess = None


def _session() -> ort.InferenceSession:
    global _sess
    if _sess is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _sess = ort.InferenceSession(str(MODEL_PATH), opts, providers=["CPUExecutionProvider"])
    return _sess


def predict_age(face_crop_bgr: np.ndarray) -> float:
    """face_crop_bgr: square face crop (see Face.crop_face). Returns age in years."""
    rgb = cv2.cvtColor(cv2.resize(face_crop_bgr, (224, 224), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
    logits = _session().run(None, {"image": x})[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return float(p @ MIDPOINTS)
