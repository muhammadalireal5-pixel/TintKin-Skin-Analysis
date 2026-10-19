"""Shape analyze() results like the YouCam skin-analysis reply that TintKin already parses.

TintKin's extractScoreInfo reads data.output[] items {type, ui_score, raw_score, mask_urls}, an "all" item
({score}) and a "skin_age" item. Masks can't be URLs from a stateless service, so each item carries
`mask_png_b64` (8-bit grayscale PNG, same pixel space as `image_size`) for the caller to upload.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

# TintKin metric key (our score name) -> YouCam action name used in the app
ACTIONS = {"wrinkles": "wrinkle", "firmness": "firmness", "spots": "age_spot", "radiance": "radiance"}
RAW_FOR = {"wrinkles": "wrinkle", "firmness": "wrinkle_cheek", "spots": "dark_spots", "radiance": "radiance_unevenness"}


def _png_b64(m: np.ndarray, hw: tuple[int, int]) -> str:
    m8 = (np.clip(cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR), 0, 1) * 255).astype(np.uint8)
    return base64.b64encode(cv2.imencode(".png", m8)[1].tobytes()).decode()


def to_youcam(res: dict, with_masks: bool = True) -> dict:
    maps = res.pop("_maps", None)
    hw = res.pop("_image_hw", None)
    head = {k: res[k] for k in ("ok", "issues", "quality", "confidence", "comparable",
                                "pipeline_version", "calibration_version") if k in res}
    if "scores" not in res:
        return {**head, "output": []}
    sc, raw = res["scores"], res.get("raw", {})
    output = []
    for key, action in ACTIONS.items():
        item = {"type": action, "ui_score": sc.get(key), "raw_score": raw.get(RAW_FOR[key])}
        if with_masks and maps is not None and hw is not None:
            item["mask_png_b64"] = _png_b64(maps[action if action != "wrinkle" else "wrinkle"], hw)
        output.append(item)
    output.append({"type": "all", "score": sc.get("overall")})
    output.append({"type": "skin_age", "score": res["skin_age"]})
    return {**head, "output": output, "face_age": res["face_age"],
            "extras": {k: sc[k] for k in ("smoothness", "pores", "redness", "clarity") if k in sc},
            **({"image_size": [hw[1], hw[0]]} if hw is not None else {})}
