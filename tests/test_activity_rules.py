"""Phone-use / drinking rules on synthetic detections (YOLO runs every 5th frame at 30 fps)."""

import pytest

from app.config import load_settings
from app.estimators.objects import ActivityRules, Detection

FRAME = (640, 480)
STEP_MS = 5 * 1000 / 30           # one YOLO run every 5 frames at 30 fps
FACE = [0.40, 0.20, 0.20, 0.30]   # normalized face box: 128×144 px, centred horizontally
MOUTH = (0.50, 0.44)


def det(kind: str, x: float, y: float, w: float = 0.08, h: float = 0.10) -> Detection:
    label = "cell phone" if kind == "phone" else "cup"
    return Detection(label=label, kind=kind, conf=0.8, bbox=[x, y, w, h])


PHONE_AT_EAR = det("phone", 0.30, 0.25)          # just left of the face
PHONE_TEXTING = det("phone", 0.45, 0.70)         # below the face, chest height
PHONE_FAR = det("phone", 0.0, 0.0, 0.05, 0.05)   # top-left corner, away from the driver
CUP_AT_MOUTH = det("drink", 0.46, 0.40)
CUP_ON_TABLE = det("drink", 0.85, 0.85)


@pytest.fixture
def rules() -> ActivityRules:
    return ActivityRules(load_settings().objects)


def run(rules, seconds, detections, face=FACE, mouth=MOUTH, t0=0.0):
    out, t = None, t0
    for _ in range(int(seconds * 1000 / STEP_MS)):
        out = rules.update(detections, face, mouth, FRAME, t)
        t += STEP_MS
    return out, t


@pytest.mark.parametrize("phone", [PHONE_AT_EAR, PHONE_TEXTING])
def test_phone_near_driver_becomes_phone_use(rules, phone) -> None:
    out, t = run(rules, 0.8, [phone])
    assert not out.phone_use  # below phone_min_s
    out, _ = run(rules, 1.0, [phone], t0=t)
    assert out.phone_use
    assert out.phone_s >= 1.0
    assert out.reasons[0].startswith("phone use")


def test_phone_far_from_face_is_ignored(rules) -> None:
    out, _ = run(rules, 3, [PHONE_FAR])
    assert not out.phone_use


def test_phone_without_face_counts_anywhere(rules) -> None:
    out, _ = run(rules, 2, [PHONE_FAR], face=None, mouth=None)
    assert out.phone_use


def test_short_misses_do_not_split_phone_episode(rules) -> None:
    _, t = run(rules, 0.7, [PHONE_AT_EAR])
    _, t = run(rules, 0.5, [], t0=t)  # YOLO misses for 0.5 s (< max_gap_s)
    out, _ = run(rules, 0.5, [PHONE_AT_EAR], t0=t)
    assert out.phone_use and out.phone_s >= 1.5


def test_phone_use_ends_after_gap(rules) -> None:
    _, t = run(rules, 2, [PHONE_AT_EAR])
    out, _ = run(rules, 2, [], t0=t)
    assert not out.phone_use and out.phone_s == 0.0


def test_cup_at_mouth_is_drinking(rules) -> None:
    out, _ = run(rules, 1, [CUP_AT_MOUTH])
    assert out.drinking
    assert not out.phone_use


def test_cup_away_from_mouth_is_not_drinking(rules) -> None:
    out, _ = run(rules, 2, [CUP_ON_TABLE])
    assert not out.drinking


def test_drinking_uses_last_face_while_face_hidden(rules) -> None:
    _, t = run(rules, 0.2, [CUP_AT_MOUTH])
    out, t = run(rules, 0.6, [CUP_AT_MOUTH], face=None, mouth=None, t0=t)  # cup hides the face
    assert out.drinking
    out, _ = run(rules, 2.0, [CUP_AT_MOUTH], face=None, mouth=None, t0=t)  # beyond face_hold_s
    assert not out.drinking


GLASS_READ_AS_PHONE = det("phone", 0.46, 0.38)  # hand-held glass at the mouth, misread by YOLO


def test_phone_box_covering_mouth_counts_as_drinking(rules) -> None:
    out, _ = run(rules, 1, [GLASS_READ_AS_PHONE])
    assert out.drinking
    assert not out.phone_use
    assert rules.detections[0].kind == "drink"
    assert "cell phone at mouth" in rules.detections[0].label


def test_glass_read_as_phone_while_face_hidden(rules) -> None:
    _, t = run(rules, 0.2, [GLASS_READ_AS_PHONE])
    out, _ = run(rules, 0.6, [GLASS_READ_AS_PHONE], face=None, mouth=None, t0=t)  # held face is used
    assert out.drinking and not out.phone_use


def test_phone_at_ear_is_not_relabelled(rules) -> None:
    run(rules, 2, [PHONE_AT_EAR])
    assert rules.detections[0].kind == "phone"


def test_phone_at_mouth_rule_can_be_disabled() -> None:
    cfg = load_settings().objects.model_copy(update={"phone_at_mouth_is_drink": False})
    out, _ = run(ActivityRules(cfg), 2, [GLASS_READ_AS_PHONE])
    assert out.phone_use and not out.drinking


def test_backward_timestamp_resets(rules) -> None:
    out, _ = run(rules, 2, [PHONE_AT_EAR], t0=10_000)
    assert out.phone_use
    out, _ = run(rules, 0.3, [PHONE_AT_EAR], t0=0)
    assert not out.phone_use
