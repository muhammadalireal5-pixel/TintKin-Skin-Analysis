"""Face detection, landmarks and skin-region masks (MediaPipe FaceMesh, CPU)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
_landmarker = None
_segmenter = None
SEG_PATH = Path(__file__).resolve().parents[1] / "models" / "selfie_multiclass_256x256.tflite"
SEG_FACE_SKIN = 3  # selfie_multiclass classes: bg, hair, body-skin, face-skin, clothes, others


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        _landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE, num_faces=2,
            min_face_detection_confidence=0.5,
        ))
    return _landmarker


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        _segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(SEG_PATH)),
            output_category_mask=True,
        ))
    return _segmenter


def face_skin_segmentation(img: np.ndarray) -> np.ndarray | None:
    """uint8 mask (255 = face skin) from the selfie multiclass segmenter; None if unavailable."""
    if not SEG_PATH.exists():
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = _get_segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    cm = res.category_mask.numpy_view().reshape(img.shape[:2])
    return ((cm == SEG_FACE_SKIN) * 255).astype(np.uint8)


# FaceMesh landmark index groups
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14,
        87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312,
        13, 82, 81, 80, 191, 78]
FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 21, 54, 103, 67, 109, 107, 336, 9]
LEFT_CHEEK = [234, 93, 132, 58, 172, 187, 205, 50, 116, 123, 147, 213]
RIGHT_CHEEK = [454, 323, 361, 288, 397, 411, 425, 280, 345, 352, 376, 433]
LEFT_UNDER_EYE = [111, 117, 118, 119, 120, 121, 128, 245, 233, 232, 231, 230, 229, 228]
RIGHT_UNDER_EYE = [340, 346, 347, 348, 349, 350, 357, 465, 453, 452, 451, 450, 449, 448]
JAW = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]


@dataclass
class Face:
    image: np.ndarray            # BGR, possibly resized
    landmarks: np.ndarray        # (478, 2) pixel coords
    n_faces: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    seg: np.ndarray | None = None    # face-skin segmentation (255 = skin), same size as image

    @property
    def face_width(self) -> int:
        return max(self.bbox[2] - self.bbox[0], 1)

    def poly_mask(self, idx: list[int]) -> np.ndarray:
        m = np.zeros(self.image.shape[:2], np.uint8)
        pts = self.landmarks[idx].astype(np.int32)
        cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
        return m

    def skin_mask(self) -> np.ndarray:
        """Face oval minus eyes, brows, lips, with a colour prior to drop hair/shadow."""
        fw = self.face_width
        m = self.poly_mask(FACE_OVAL)
        d = max(3, int(0.02 * fw) | 1)  # kernels scale with the face, not the image
        for grp in (LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS):
            hull = self.poly_mask(grp)
            hull = cv2.dilate(hull, np.ones((d, d), np.uint8))
            m[hull > 0] = 0
        # erode so boundary pixels (hair, background) don't leak in
        k = max(3, int(0.03 * fw) | 1)
        m = cv2.erode(m, np.ones((k, k), np.uint8))
        # colour prior in YCrCb (keeps skin-like pixels, tolerant across tones)
        ycc = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
        prior = cv2.inRange(ycc, (0, 130, 75), (255, 185, 140))
        prior = cv2.morphologyEx(prior, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        out = cv2.bitwise_and(m, prior)
        # fall back to geometric mask if prior removed too much (unusual lighting)
        out = out if out.sum() > 0.5 * m.sum() else m
        if self.seg is not None:
            seg_out = cv2.bitwise_and(out, self.seg)
            out = seg_out if seg_out.sum() > 0.5 * out.sum() else out
        return self._drop_hair(out)

    def _drop_hair(self, m: np.ndarray) -> np.ndarray:
        """Remove beard, stubble and hair strands without eating shadowed or dark skin.

        Below the nose base (beard/moustache zone) a pixel is dropped when it is much darker than
        the face's median skin AND textured, or extremely dark. Smooth shadows (jaw side, under the
        lip) are kept. Elsewhere only pixels far darker than their own neighbourhood go (hairline strands).
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sel = m > 0
        if sel.sum() < 500:
            return m
        fw = self.face_width
        med = float(np.median(gray[sel]))
        s = max(3, int(0.02 * fw) | 1)
        dark_blur = cv2.GaussianBlur(gray, (0, 0), s)
        busy = cv2.GaussianBlur(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.0)), (0, 0), s)
        busy_med = float(np.median(busy[sel]))

        zone = np.zeros(m.shape, bool)
        zone[int(self.landmarks[2, 1]):, :] = True

        mf = sel.astype(np.float32)
        sig = 0.12 * fw
        local = cv2.GaussianBlur(gray * mf, (0, 0), sig) / np.maximum(cv2.GaussianBlur(mf, (0, 0), sig), 1e-4)

        beard = zone & (((dark_blur < 0.72 * med) & (busy > 1.5 * busy_med)) | (dark_blur < 0.5 * med)
                        | ((busy > 3.0 * busy_med) & (gray < 0.9 * med)))
        strands = ~zone & (dark_blur < 0.6 * local)
        dark = ((beard | strands) & sel).astype(np.uint8) * 255
        dark = cv2.dilate(dark, np.ones((2 * s + 1, 2 * s + 1), np.uint8))
        out = cv2.bitwise_and(m, cv2.bitwise_not(dark))
        return out if out.sum() > 0.35 * m.sum() else m

    def region(self, idx: list[int]) -> np.ndarray:
        return cv2.bitwise_and(self.poly_mask(idx), self.skin_mask_cached())

    def skin_mask_cached(self) -> np.ndarray:
        if not hasattr(self, "_skin"):
            self._skin = self.skin_mask()
        return self._skin

    def normalized(self, face_width: int = 512) -> "Face":
        """Rescale so the face is a fixed width: pixel metrics become scale-invariant."""
        x0, _, x1, _ = self.bbox
        s = face_width / max(x1 - x0, 1)
        img = cv2.resize(self.image, None, fx=s, fy=s,
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        lm = self.landmarks * s
        b = tuple(int(v * s) for v in self.bbox)
        seg = None if self.seg is None else cv2.resize(self.seg, (img.shape[1], img.shape[0]),
                                                       interpolation=cv2.INTER_NEAREST)
        return Face(img, lm, self.n_faces, b, seg)

    def beard_free_image(self) -> np.ndarray:
        """Copy of the image with beard/stubble painted over (Telea inpaint), for the age model only.

        Region = face oval below the nose base, minus lips and minus the clean skin mask. Facial hair
        otherwise makes the age model read young faces as older."""
        skin = self.skin_mask_cached() > 0
        lips = cv2.dilate(self.poly_mask(LIPS), np.ones((15, 15), np.uint8)) > 0
        zone = np.zeros(skin.shape, bool)
        zone[int(self.landmarks[2, 1]):, :] = True
        reg = (self.poly_mask(FACE_OVAL) > 0) & zone & ~skin & ~lips
        reg = (cv2.dilate(reg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~lips
        return cv2.inpaint(self.image, reg.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)

    def crop_face(self, size: int = 224, margin: float = 0.15, beard_free: bool = False) -> np.ndarray:
        image = self.beard_free_image() if beard_free else self.image
        x0, y0, x1, y1 = self.bbox
        w, h = x1 - x0, y1 - y0
        s = int(max(w, h) * (1 + margin))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        xa, ya = max(0, cx - s // 2), max(0, cy - s // 2)
        xb, yb = min(image.shape[1], cx + s // 2), min(image.shape[0], cy + s // 2)
        return cv2.resize(image[ya:yb, xa:xb], (size, size), interpolation=cv2.INTER_AREA)


def load_image(data: bytes | str, max_side: int = 2048) -> np.ndarray:
    if isinstance(data, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def detect_face(img: np.ndarray) -> Face | None:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = _get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    h, w = img.shape[:2]
    lm = res.face_landmarks[0][:478]
    pts = np.array([[p.x * w, p.y * h] for p in lm], np.float32)
    x0, y0 = pts.min(0).astype(int)
    x1, y1 = pts.max(0).astype(int)
    bbox = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
    return Face(img, pts, len(res.face_landmarks), bbox, face_skin_segmentation(img))
