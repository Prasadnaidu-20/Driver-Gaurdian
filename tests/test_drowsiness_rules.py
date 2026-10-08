"""Synthetic 30 fps feature sequences through the rule-based drowsiness estimator."""

import random

import pytest

from app.config import load_settings
from app.estimators.drowsiness_rules import DrowsinessRules
from app.pipeline.calibration import Baseline

FPS = 30
DT = 1000 / FPS
OPEN_EAR = 0.30
BASELINE = Baseline(ear_open=OPEN_EAR, mar_closed=0.04, yaw=0, pitch=0, gaze_h=0.5, gaze_v=0.5, calibrated=True)


def feat(ear=OPEN_EAR, mar=0.04, blink=0.1, pitch=0.0) -> dict[str, float]:
    return {
        "ear_left": ear, "ear_right": ear, "ear_mean": ear, "mar": mar,
        "yaw": 0.0, "pitch": pitch, "roll": 0.0, "gaze_h": 0.5, "gaze_v": 0.5,
        "eye_blink_left": blink, "eye_blink_right": blink, "jaw_open": 0.0,
    }


CLOSED = feat(ear=0.08, blink=0.8)


class Clock:
    def __init__(self, est: DrowsinessRules) -> None:
        self.est = est
        self.t = 0.0
        self.outputs = []

    def run(self, seconds: float, f) -> list:
        out = []
        for _ in range(round(seconds * FPS)):
            out.append(self.est.update(f, self.t))
            self.t += DT
        self.outputs += out
        return out


@pytest.fixture
def est() -> DrowsinessRules:
    return DrowsinessRules(load_settings().drowsiness, BASELINE)


def labels(outputs) -> set[str]:
    return {o.label for o in outputs}


def test_natural_blinks_stay_alert(est: DrowsinessRules) -> None:
    clock = Clock(est)
    rng = random.Random(0)
    while clock.t < 60_000:
        clock.run(rng.uniform(0.15, 0.30), CLOSED)   # blink 150-300 ms
        clock.run(rng.uniform(3.0, 5.0), feat())     # every 3-5 s
    assert labels(clock.outputs) == {"alert"}
    assert max(o.score for o in clock.outputs) < 0.1
    last = clock.outputs[-1]
    assert last.details["blinks"] >= 10
    assert 100 <= last.details["mean_blink_ms"] <= 300


def test_two_second_closure_triggers_then_recovers_after_exit_delay(est: DrowsinessRules) -> None:
    cfg = load_settings().drowsiness
    clock = Clock(est)
    clock.run(10, feat())
    closure = clock.run(2.0, CLOSED)
    assert closure[-1].label == "drowsy"
    assert any(r.startswith("eyes closed") for r in closure[-1].reasons)
    assert all(o.label == "alert" for o in closure[: int(0.9 * FPS)])  # nothing during the first 0.9 s
    after = clock.run(5, feat())
    # Still elevated during the exit delay, back to alert after it.
    assert after[int(1.0 * FPS)].label != "alert"
    first_alert = next(i for i, o in enumerate(after) if o.label == "alert")
    assert first_alert / FPS >= cfg.min_exit_s
    assert after[-1].label == "alert"
    assert after[-1].details["long_closures"] == 1


def test_one_second_closure_is_not_drowsy(est: DrowsinessRules) -> None:
    clock = Clock(est)
    clock.run(10, feat())
    out = clock.run(0.9, CLOSED) + clock.run(3, feat())
    assert labels(out) == {"alert"}


def test_yawns_counted_and_raise_score(est: DrowsinessRules) -> None:
    clock = Clock(est)
    clock.run(5, feat())
    base_score = clock.outputs[-1].score
    for _ in range(3):
        clock.run(2.0, feat(mar=0.8))
        clock.run(8, feat())
    last = clock.outputs[-1]
    assert last.details["yawns"] == 3
    assert last.score > base_score + 0.3
    assert last.label == "slightly_drowsy"
    assert any("3 yawns" in r for r in last.reasons)


def test_talking_is_not_a_yawn(est: DrowsinessRules) -> None:
    clock = Clock(est)
    for _ in range(10):
        clock.run(1.0, feat(mar=0.8))  # 1 s mouth openings (shorter than yawn_min_s)
        clock.run(0.5, feat())
    assert clock.outputs[-1].details["yawns"] == 0


def test_high_perclos_is_drowsy(est: DrowsinessRules) -> None:
    clock = Clock(est)
    # Slow, long blinks (0.8 s every 1.6 s): no single long closure, but PERCLOS ~50%.
    for _ in range(40):
        clock.run(0.8, CLOSED)
        clock.run(0.8, feat())
    last = clock.outputs[-1]
    assert last.details["perclos"] > 0.4
    assert last.label == "drowsy"
    assert any(r.startswith("PERCLOS") for r in last.reasons)


def test_calibration_changes_ear_threshold() -> None:
    cfg = load_settings().drowsiness
    narrow_eyes = Baseline(ear_open=0.20, mar_closed=0.04, yaw=0, pitch=0, gaze_h=0.5, gaze_v=0.5, calibrated=True)
    f = feat(ear=0.17, blink=0.1)  # a narrow-eyed driver with eyes open
    assert DrowsinessRules(cfg, BASELINE).eye_closed(f)           # 0.17 < 0.7 × 0.30
    assert not DrowsinessRules(cfg, narrow_eyes).eye_closed(f)    # 0.17 ≥ 0.7 × 0.20


def test_looking_down_uses_only_blendshape(est: DrowsinessRules) -> None:
    clock = Clock(est)
    clock.run(5, feat())
    out = clock.run(4, feat(ear=0.15, blink=0.3, pitch=-30))  # head down: EAR drops, eyes open
    assert labels(out) == {"alert"}
    assert est.eye_closed(feat(ear=0.05, blink=0.9, pitch=-30))  # real closure still caught by blendshape


def test_no_face_reports_unknown_and_keeps_going(est: DrowsinessRules) -> None:
    clock = Clock(est)
    clock.run(2, feat())
    out = clock.run(1, None)
    assert labels(out) == {"unknown"}
    assert out[-1].reasons == ["no face"]
    assert clock.run(1, feat())[-1].label == "alert"


def test_duplicate_timestamp_returns_previous(est: DrowsinessRules) -> None:
    a = est.update(feat(), 100)
    assert est.update(CLOSED, 100) is a


def test_reset_clears_state(est: DrowsinessRules) -> None:
    clock = Clock(est)
    clock.run(2.0, CLOSED)
    est.reset()
    clock.t = 0
    assert clock.run(0.1, feat())[-1].label == "alert"
