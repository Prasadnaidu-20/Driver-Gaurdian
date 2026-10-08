"""Face crop shared by inference (emotion, Phase 4; ML model, Phase 8) and training (Phase 7).

Training imports this module directly, so it must not depend on app.config: size and
margin are explicit arguments (take them from `face.crop_size` / `face.crop_margin`).
"""

from __future__ import annotations

import cv2
import numpy as np


def crop_face(frame_bgr: np.ndarray, landmarks: np.ndarray, size: int, margin: float) -> np.ndarray:
    """Square face crop resized to (size, size, 3) **RGB** uint8.

    The square is centred on the landmark bounding box with side max(w, h) · (1 + margin).
    Parts of the square outside the frame are padded with black, so a face at the frame
    edge stays centred and is not stretched.

    Args:
        frame_bgr: full frame, BGR uint8 (OpenCV).
        landmarks: (N, 2+) normalized landmark coordinates (0-1).
        size: output side length in pixels.
        margin: extra margin as a fraction of the bbox side (0.25 = 25 %).
    """
    h, w = frame_bgr.shape[:2]
    xy = landmarks[:, :2] * np.array([w, h], dtype=np.float64)
    (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0, 1.0) * (1.0 + margin)

    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    side_px = max(int(round(side)), 1)
    right, bottom = left + side_px, top + side_px

    # Pad so the full square lies inside the (padded) frame, then slice.
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, right - w), max(0, bottom - h)
    if pad_l or pad_t or pad_r or pad_b:
        frame_bgr = cv2.copyMakeBorder(frame_bgr, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = frame_bgr[top + pad_t:bottom + pad_t, left + pad_l:right + pad_l]

    interp = cv2.INTER_AREA if side_px > size else cv2.INTER_LINEAR
    crop = cv2.resize(crop, (size, size), interpolation=interp)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
