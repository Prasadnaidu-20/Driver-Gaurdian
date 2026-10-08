import math

import numpy as np
import pytest

from app.pipeline.features import (
    FEATURE_KEYS,
    LEFT_EYE_CORNERS,
    LEFT_EYE_LIDS,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_LIDS,
    RIGHT_IRIS_CENTER,
    extract_features,
    eye_aspect_ratio,
    face_bbox,
    gaze_ratios,
    mouth_aspect_ratio,
    rotation_to_euler,
)


def eye_points(width: float, opening: float) -> np.ndarray:
    """p1..p6 for an eye `width` wide whose lids are `opening` apart."""
    half = opening / 2
    return np.array([
        [0, 0], [width / 3, -half], [2 * width / 3, -half],
        [width, 0], [2 * width / 3, half], [width / 3, half],
    ], dtype=float)


def mouth_points(width: float, opening: float) -> np.ndarray:
    """p1..p8: corners at x=0 and x=width, three upper/lower pairs `opening` apart."""
    half = opening / 2
    xs = (width / 4, width / 2, 3 * width / 4)
    upper = [[x, -half] for x in xs]
    lower = [[x, half] for x in reversed(xs)]  # p6, p7, p8 run back from the right corner
    return np.array([[0, 0], *upper, [width, 0], *lower], dtype=float)


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


# ---- EAR / MAR ----

def test_ear_known_value() -> None:
    # verticals both = 9, horizontal = 30 -> (9 + 9) / (2 * 30) = 0.3
    assert eye_aspect_ratio(eye_points(30, 9)) == pytest.approx(0.3)


def test_ear_closed_eye_is_near_zero() -> None:
    assert eye_aspect_ratio(eye_points(30, 0)) == pytest.approx(0.0, abs=1e-9)
    assert eye_aspect_ratio(eye_points(30, 1)) < eye_aspect_ratio(eye_points(30, 9))


def test_ear_is_scale_and_translation_invariant() -> None:
    pts = eye_points(30, 9)
    assert eye_aspect_ratio(pts * 3.7 + 100) == pytest.approx(eye_aspect_ratio(pts))


def test_mar_known_value_and_ordering() -> None:
    # three verticals = 20, horizontal = 50 -> 60 / 100 = 0.6
    assert mouth_aspect_ratio(mouth_points(50, 20)) == pytest.approx(0.6)
    assert mouth_aspect_ratio(mouth_points(50, 0)) == pytest.approx(0.0, abs=1e-9)
    assert mouth_aspect_ratio(mouth_points(50, 5)) < mouth_aspect_ratio(mouth_points(50, 20))


def test_degenerate_points_do_not_divide_by_zero() -> None:
    assert math.isfinite(eye_aspect_ratio(np.zeros((6, 2))))
    assert math.isfinite(mouth_aspect_ratio(np.zeros((8, 2))))


# ---- rotation -> Euler ----

def test_identity_is_zero_pose() -> None:
    assert rotation_to_euler(np.eye(4)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_pure_yaw() -> None:
    # Ry(+20): nose (+z) swings toward +x (image right) = driver turns to their left -> yaw +20
    assert rotation_to_euler(rot_y(20)) == pytest.approx((20.0, 0.0, 0.0), abs=1e-9)


def test_pure_pitch_up_is_positive() -> None:
    # Rx(-15) in a y-up frame tilts the nose up -> pitch +15
    assert rotation_to_euler(rot_x(-15)) == pytest.approx((0.0, 15.0, 0.0), abs=1e-9)
    assert rotation_to_euler(rot_x(25))[1] == pytest.approx(-25.0)  # looking down is negative


def test_pure_roll() -> None:
    # Rz(+10) tilts the head toward the driver's right shoulder -> roll -10
    assert rotation_to_euler(rot_z(10)) == pytest.approx((0.0, 0.0, -10.0), abs=1e-9)


@pytest.mark.parametrize("a,b,c", [(30, -20, 10), (-45, 15, -5), (5, 40, 25)])
def test_combined_rotation_round_trip(a: float, b: float, c: float) -> None:
    m = np.eye(4)
    m[:3, :3] = rot_y(a) @ rot_x(b) @ rot_z(c)
    m[:3, 3] = [1.0, 20.0, -60.0]  # translation must be ignored
    assert rotation_to_euler(m) == pytest.approx((a, -b, -c), abs=1e-9)


# ---- gaze / bbox / extract ----

def synthetic_face(iris_shift_x: float = 0.0) -> np.ndarray:
    """478 pixel-space points with two simple eyes; irises centred plus an optional x shift."""
    px = np.full((478, 2), 50.0)
    for corners, lids, iris, x0 in (
        (RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, RIGHT_IRIS_CENTER, 20.0),
        (LEFT_EYE_CORNERS, LEFT_EYE_LIDS, LEFT_IRIS_CENTER, 60.0),
    ):
        px[corners[0]] = [x0, 40]
        px[corners[1]] = [x0 + 20, 40]
        px[lids[0]] = [x0 + 10, 35]
        px[lids[1]] = [x0 + 10, 45]
        px[iris] = [x0 + 10 + iris_shift_x, 40]
    return px


def test_gaze_centred_is_half() -> None:
    assert gaze_ratios(synthetic_face()) == pytest.approx((0.5, 0.5))


def test_gaze_follows_iris_horizontally() -> None:
    assert gaze_ratios(synthetic_face(iris_shift_x=5))[0] == pytest.approx(0.75)
    assert gaze_ratios(synthetic_face(iris_shift_x=-5))[0] == pytest.approx(0.25)


def test_face_bbox_is_clipped() -> None:
    lm = np.array([[-0.1, 0.2, 0.0], [0.5, 1.3, 0.0], [0.3, 0.5, 0.0]])
    assert face_bbox(lm) == pytest.approx([0.0, 0.2, 0.5, 0.8])


def test_extract_features_has_all_keys() -> None:
    lm = np.random.default_rng(0).uniform(0.3, 0.7, size=(478, 3))
    feats = extract_features(lm, {"eyeBlinkLeft": 0.9, "jawOpen": 0.2}, np.eye(4), 640, 480)
    assert set(feats) == set(FEATURE_KEYS)
    assert feats["eye_blink_left"] == 0.9
    assert feats["eye_blink_right"] == 0.0
    assert feats["ear_mean"] == pytest.approx((feats["ear_left"] + feats["ear_right"]) / 2)
    assert all(math.isfinite(v) for v in feats.values())
