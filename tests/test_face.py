import numpy as np
import pytest

from app.config import Settings, get_settings
from app.pipeline.face import MonotonicClock
from app.pipeline.frame_processor import FrameProcessor


def test_clock_increasing_passthrough() -> None:
    clock = MonotonicClock(reset_ms=1000)
    assert clock.next(10.4) == (10, False)
    assert clock.next(43.9) == (43, False)


def test_clock_nudges_equal_and_small_backward() -> None:
    clock = MonotonicClock(reset_ms=1000)
    clock.next(100.0)
    assert clock.next(100.0) == (101, False)  # duplicate video currentTime
    assert clock.next(100.2) == (102, False)  # same integer ms
    assert clock.next(50.0) == (103, False)   # small backward jump
    assert clock.next(200.0) == (200, False)


def test_clock_large_backward_requests_reset() -> None:
    clock = MonotonicClock(reset_ms=1000)
    clock.next(5000.0)
    assert clock.next(0.0) == (0, True)  # seek to start / replay
    assert clock.next(33.0) == (33, False)


def settings_with_model(path: str) -> Settings:
    s = get_settings()
    return s.model_copy(update={"face": s.face.model_copy(update={"model_path": path})})


def test_missing_model_is_unavailable_not_a_crash() -> None:
    processor = FrameProcessor(settings_with_model("models_store/does_not_exist.task"))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = processor.process(frame, 10.0, 1)
    assert result.status == "ok"
    assert result.face.available is False
    assert result.face.detected is False
    processor.close()


requires_model = pytest.mark.skipif(
    not get_settings().face.resolved_model_path().is_file(),
    reason="face_landmarker.task not downloaded",
)


@requires_model
def test_blank_frames_report_no_face_and_survive_timestamp_jumps() -> None:
    processor = FrameProcessor(get_settings())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for ts in (0.0, 33.0, 33.0, 10.0, 5000.0, 0.0, 33.0):  # duplicate, small and large backward jumps
        result = processor.process(frame, ts, 1)
        assert result.status == "no_face"
        assert result.face.available is True
        assert result.face.detected is False
        assert result.timestamp_ms == ts
    processor.close()
