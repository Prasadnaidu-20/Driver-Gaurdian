"""Per-frame geometric facial features from MediaPipe Face Landmarker output.

All functions are pure NumPy, so they are unit-testable without MediaPipe.

Conventions
-----------
* Landmarks are MediaPipe's 478-point mesh in normalized image coordinates (x right,
  y down, 0-1). Distances are computed in **pixels** (x*width, y*height) so ratios
  are not distorted by the frame's aspect ratio.
* "Left"/"right" for eyes mean the **driver's** anatomical side. In an unmirrored
  camera image the driver's right eye appears on the image left.
* Landmark indices below are fixed mesh topology (not tunable), so they live here as
  named constants rather than in config.

Head pose sign conventions (degrees, 0 = facing the camera)
-----------------------------------------------------------
MediaPipe's facial transformation matrix maps the canonical face model into camera
space: x = image right, y = up, z = toward the viewer (camera looks down -z). We
decompose its rotation as R = Ry(a) · Rx(b) · Rz(c) and report:
* yaw   = YAW_SIGN   * a : + = driver turns head to **their left** (nose toward image right)
* pitch = PITCH_SIGN * b : + = head tilts **up**, - = looking down
* roll  = ROLL_SIGN  * c : + = head tilts toward the driver's **left** shoulder
(The raw roll sign was checked against in-plane rotated images; flip a *_SIGN constant
if a live test shows a sign inverted.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# ---- landmark indices (MediaPipe face mesh topology) ----
# 6-point EAR order: p1 outer/inner corner, p2,p3 upper lid, p4 other corner, p5,p6 lower lid.
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)
# 8-point MAR order: p1 left corner, p2..p4 upper inner lip, p5 right corner, p6..p8 lower inner lip.
MOUTH_MAR = (61, 81, 13, 311, 291, 402, 14, 178)

RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
# Eye corners ordered image-left → image-right, and upper/lower lid midpoints.
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_LIDS = (159, 145)
LEFT_EYE_LIDS = (386, 374)

# Contours sent to the frontend overlay.
OVERLAY_GROUPS: dict[str, tuple[int, ...]] = {
    "right_eye": (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    "left_eye": (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
    "mouth": (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185),
    "right_iris": (468, 469, 470, 471, 472),
    "left_iris": (473, 474, 475, 476, 477),
}

# ---- head pose sign conventions (see module docstring) ----
YAW_SIGN = 1.0
PITCH_SIGN = -1.0  # Rx(+b) points the nose down in a y-up frame, so negate for "+ = up"
ROLL_SIGN = -1.0   # raw +c tilts the head toward the driver's right shoulder

_EPS = 1e-9

FEATURE_KEYS = (
    "ear_left", "ear_right", "ear_mean", "mar",
    "yaw", "pitch", "roll",
    "gaze_h", "gaze_v",
    "eye_blink_left", "eye_blink_right", "jaw_open",
)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized (N,2+) landmarks to (N,2) pixel coordinates."""
    return landmarks[:, :2] * np.array([width, height], dtype=np.float64)


def eye_aspect_ratio(points: np.ndarray) -> float:
    """Standard 6-point EAR: (|p2-p6| + |p3-p5|) / (2|p1-p4|). `points` is (6,2) in pixel space."""
    p1, p2, p3, p4, p5, p6 = points
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + _EPS)


def mouth_aspect_ratio(points: np.ndarray) -> float:
    """8-point MAR: (|p2-p8| + |p3-p7| + |p4-p6|) / (2|p1-p5|). `points` is (8,2) in pixel space."""
    p1, p2, p3, p4, p5, p6, p7, p8 = points
    return (_dist(p2, p8) + _dist(p3, p7) + _dist(p4, p6)) / (2.0 * _dist(p1, p5) + _EPS)


def rotation_to_euler(matrix: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a 3×3 or 4×4 transform, with the signs documented above.

    Decomposition R = Ry(a)·Rx(b)·Rz(c) gives
    b = asin(-R[1,2]), a = atan2(R[0,2], R[2,2]), c = atan2(R[1,0], R[1,1]).
    """
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    a = math.atan2(r[0, 2], r[2, 2])
    b = math.asin(float(np.clip(-r[1, 2], -1.0, 1.0)))
    c = math.atan2(r[1, 0], r[1, 1])
    return (
        YAW_SIGN * math.degrees(a),
        PITCH_SIGN * math.degrees(b),
        ROLL_SIGN * math.degrees(c),
    )


def _ratio_along(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Position of `point` projected onto the segment start→end (0 at start, 1 at end)."""
    axis = end - start
    return float(np.dot(point - start, axis) / (np.dot(axis, axis) + _EPS))


def gaze_ratios(px: np.ndarray) -> tuple[float, float]:
    """Iris position within the eyes, averaged over both eyes. `px` is (478,2) pixel landmarks.

    Horizontal: 0 = iris at the image-left eye corner, 1 = image-right corner, ≈0.5 centred.
    Vertical:   0 = iris at the upper lid, 1 = lower lid, ≈0.5 centred.
    """
    horiz, vert = [], []
    for iris, corners, lids in (
        (RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS),
        (LEFT_IRIS_CENTER, LEFT_EYE_CORNERS, LEFT_EYE_LIDS),
    ):
        horiz.append(_ratio_along(px[iris], px[corners[0]], px[corners[1]]))
        vert.append(_ratio_along(px[iris], px[lids[0]], px[lids[1]]))
    return float(np.mean(horiz)), float(np.mean(vert))


def face_bbox(landmarks: np.ndarray) -> list[float]:
    """Normalized [x, y, w, h] bounding box of all landmarks, clipped to the frame."""
    xy = np.clip(landmarks[:, :2], 0.0, 1.0)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def mouth_center(landmarks: np.ndarray) -> tuple[float, float]:
    """Normalized (x, y) centre of the inner-lip points (used by the drinking rule)."""
    x, y = landmarks[list(MOUTH_MAR), :2].mean(axis=0)
    return float(x), float(y)


def extract_features(
    landmarks: np.ndarray,
    blendshapes: dict[str, float],
    transform: np.ndarray | None,
    width: int,
    height: int,
) -> dict[str, float]:
    """All per-frame features (keys in FEATURE_KEYS). Head pose is NaN-free: 0 if no matrix."""
    px = to_pixels(landmarks, width, height)
    ear_right = eye_aspect_ratio(px[list(RIGHT_EYE_EAR)])
    ear_left = eye_aspect_ratio(px[list(LEFT_EYE_EAR)])
    yaw, pitch, roll = rotation_to_euler(transform) if transform is not None else (0.0, 0.0, 0.0)
    gaze_h, gaze_v = gaze_ratios(px)
    return {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "ear_mean": (ear_left + ear_right) / 2.0,
        "mar": mouth_aspect_ratio(px[list(MOUTH_MAR)]),
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "gaze_h": gaze_h,
        "gaze_v": gaze_v,
        "eye_blink_left": float(blendshapes.get("eyeBlinkLeft", 0.0)),
        "eye_blink_right": float(blendshapes.get("eyeBlinkRight", 0.0)),
        "jaw_open": float(blendshapes.get("jawOpen", 0.0)),
    }


def overlay_landmarks(landmarks: np.ndarray, groups: dict[str, Sequence[int]] = OVERLAY_GROUPS) -> dict[str, list[list[float]]]:
    """Normalized [x, y] points per overlay group, rounded to keep the JSON small."""
    return {
        name: [[round(float(landmarks[i, 0]), 4), round(float(landmarks[i, 1]), 4)] for i in idx]
        for name, idx in groups.items()
    }
