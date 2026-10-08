import pytest

from app.config import load_settings
from app.pipeline.calibration import Baseline, Calibrator

FPS = 30
DT = 1000 / FPS


def features(ear=0.30, mar=0.04, yaw=12.0, pitch=-8.0, gaze_h=0.55, gaze_v=0.45) -> dict[str, float]:
    return {"ear_mean": ear, "mar": mar, "yaw": yaw, "pitch": pitch, "gaze_h": gaze_h, "gaze_v": gaze_v}


@pytest.fixture
def cal() -> Calibrator:
    return Calibrator(load_settings().calibration)


def run(cal: Calibrator, seconds: float, frame_fn, t0: float = 0.0):
    result = None
    n = int(seconds * FPS) + 1
    for i in range(n):
        out = cal.feed(frame_fn(i), t0 + i * DT)
        if out is not None:
            result = out
    return result


def test_uncalibrated_defaults(cal: Calibrator) -> None:
    assert cal.state == "uncalibrated"
    assert not cal.baseline.calibrated
    assert cal.baseline.ear_open == load_settings().calibration.defaults.ear_open


def test_calibration_medians_with_blinks(cal: Calibrator) -> None:
    cal.start()
    assert cal.state == "calibrating"

    def frame(i: int):
        if i % 100 < 6:  # a blink every ~3.3 s
            return features(ear=0.05)
        return features(ear=0.30 + (0.01 if i % 2 else -0.01))

    baseline = run(cal, 10.0, frame, t0=5000)
    assert baseline is not None and baseline.calibrated
    assert baseline.ear_open == pytest.approx(0.30, abs=0.011)
    assert (baseline.yaw, baseline.pitch) == (12.0, -8.0)
    assert (baseline.gaze_h, baseline.gaze_v) == (0.55, 0.45)
    assert cal.state == "calibrated" and cal.progress == 1.0


def test_calibration_completes_by_frame_time(cal: Calibrator) -> None:
    cal.start()
    run(cal, 5.0, lambda i: features())
    assert cal.state == "calibrating"
    assert cal.progress == pytest.approx(0.5, abs=0.01)
    assert cal.remaining_s == pytest.approx(5.0, abs=0.05)


def test_calibration_fails_without_face(cal: Calibrator) -> None:
    cal.start()
    assert run(cal, 10.0, lambda i: features() if i % 3 == 0 else None) is None
    assert cal.state == "uncalibrated"
    assert "face visible" in cal.message


def test_calibration_fails_with_closed_eyes_and_keeps_previous(cal: Calibrator) -> None:
    previous = Baseline.from_dict(features() | {"ear_open": 0.31, "mar_closed": 0.03})
    cal.set_baseline(previous)
    cal.start()
    assert run(cal, 10.0, lambda i: features(ear=0.08)) is None
    assert "eyes looked closed" in cal.message
    assert cal.state == "calibrated" and cal.baseline == previous


def test_calibration_fails_with_open_mouth(cal: Calibrator) -> None:
    cal.start()
    assert run(cal, 10.0, lambda i: features(mar=0.6)) is None
    assert "mouth looked open" in cal.message


def test_restart_on_seek(cal: Calibrator) -> None:
    cal.start()
    run(cal, 6.0, lambda i: features(), t0=60_000)
    cal.restart_if_running()
    assert cal.progress == 0.0
    run(cal, 6.0, lambda i: features(), t0=0)
    assert cal.state == "calibrating"  # needs the full duration again


def test_baseline_from_dict_validation() -> None:
    b = Baseline.from_dict({"ear_open": 0.3, "mar_closed": 0.05, "yaw": 1, "pitch": 2, "gaze_h": 0.5, "gaze_v": 0.5})
    assert b.calibrated and b.to_dict()["yaw"] == 1.0
    with pytest.raises(ValueError):
        Baseline.from_dict({"ear_open": 0.3})
    with pytest.raises(ValueError):
        Baseline.from_dict({"ear_open": "x", "mar_closed": 0, "yaw": 0, "pitch": 0, "gaze_h": 0, "gaze_v": 0})
    with pytest.raises(ValueError):
        Baseline.from_dict({"ear_open": float("nan"), "mar_closed": 0, "yaw": 0, "pitch": 0, "gaze_h": 0, "gaze_v": 0})
