"""Synthetic 30 fps feature sequences through the rule-based distraction estimator."""

import pytest

from app.config import load_settings
from app.estimators.distraction_rules import DistractionRules
from app.pipeline.calibration import Baseline

FPS = 30
DT = 1000 / FPS
BASELINE = Baseline(ear_open=0.30, mar_closed=0.04, yaw=0, pitch=0, gaze_h=0.5, gaze_v=0.5, calibrated=True)


def feat(yaw=0.0, pitch=0.0, gaze_h=0.5, gaze_v=0.5, ear=0.30) -> dict[str, float]:
    return {
        "ear_left": ear, "ear_right": ear, "ear_mean": ear, "mar": 0.04,
        "yaw": yaw, "pitch": pitch, "roll": 0.0, "gaze_h": gaze_h, "gaze_v": gaze_v,
        "eye_blink_left": 0.1, "eye_blink_right": 0.1, "jaw_open": 0.0,
    }


class Clock:
    def __init__(self, est: DistractionRules) -> None:
        self.est = est
        self.t = 0.0

    def run(self, seconds: float, f) -> list:
        out = []
        for _ in range(round(seconds * FPS)):
            out.append(self.est.update(f, self.t))
            self.t += DT
        return out


@pytest.fixture
def est() -> DistractionRules:
    return DistractionRules(load_settings().distraction, BASELINE)


def labels(outputs) -> set[str]:
    return {o.label for o in outputs}


def test_one_second_mirror_glance_is_ignored(est: DistractionRules) -> None:
    clock = Clock(est)
    out = clock.run(5, feat()) + clock.run(1.0, feat(yaw=45)) + clock.run(5, feat())
    assert labels(out) == {"attentive"}
    assert max(o.score for o in out) == 0.0


def test_glance_at_stream_start_is_ignored(est: DistractionRules) -> None:
    clock = Clock(est)
    out = clock.run(1.5, feat(yaw=-45)) + clock.run(3, feat())
    assert labels(out) == {"attentive"}


def test_four_second_look_down_is_distracted(est: DistractionRules) -> None:
    clock = Clock(est)
    clock.run(5, feat())
    out = clock.run(4.0, feat(pitch=-35))
    assert out[int(1.9 * FPS)].label == "attentive"
    assert out[-1].label == "distracted"
    assert any("looking down" in r for r in out[-1].reasons)
    after = clock.run(5, feat())
    assert after[-1].label == "attentive"


def test_frequent_short_glances_raise_fraction(est: DistractionRules) -> None:
    clock = Clock(est)
    out = []
    for _ in range(5):  # 1.5 s away, 1 s back: no glance > 2 s, but 60% off road
        out += clock.run(1.5, feat(yaw=45)) + clock.run(1.0, feat())
    assert max(o.details["off_road_s"] for o in out) < 2.0  # glances stay separate episodes
    assert out[-1].details["glance_fraction"] > 0.5
    assert out[-1].label in ("looking_away", "distracted")
    assert any("of last 10 s" in r for r in out[-1].reasons)


def test_camera_offset_absorbed_by_calibration() -> None:
    cfg = load_settings().distraction
    side_camera = Baseline(ear_open=0.30, mar_closed=0.04, yaw=25, pitch=-10, gaze_h=0.6, gaze_v=0.5, calibrated=True)
    road = feat(yaw=25, pitch=-10, gaze_h=0.6)  # looking at the road through an off-axis camera
    assert DistractionRules(cfg, side_camera).off_road(road) == []
    assert DistractionRules(cfg, BASELINE).off_road(feat(yaw=35, pitch=-10, gaze_h=0.6))  # uncorrected → off road
    clock = Clock(DistractionRules(cfg, side_camera))
    assert labels(clock.run(6, road)) == {"attentive"}


def test_off_road_cone_and_gaze() -> None:
    est = DistractionRules(load_settings().distraction, BASELINE)
    assert est.off_road(feat(yaw=29)) == []
    assert "head turned left" in est.off_road(feat(yaw=31))[0]
    assert "head turned right" in est.off_road(feat(yaw=-31))[0]
    assert est.off_road(feat(pitch=-19)) == []
    assert "looking down" in est.off_road(feat(pitch=-21))[0]
    assert est.off_road(feat(gaze_h=0.8)) == ["gaze off road"]
    assert est.off_road(feat(gaze_h=0.8, ear=0.1)) == []  # eyes closed: gaze ignored


def test_no_face_reports_unknown(est: DistractionRules) -> None:
    clock = Clock(est)
    clock.run(1, feat())
    assert labels(clock.run(1, None)) == {"unknown"}
