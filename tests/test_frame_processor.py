"""FrameProcessor wiring for Phase 4: frame-skip of heavy models, activities → distraction, unavailable models."""

import numpy as np
import pytest

from app.config import Settings, get_settings
from app.estimators.emotion import EmotionEstimator
from app.estimators.objects import Detection
from app.pipeline.face import FaceDetection
from app.pipeline.frame_processor import FrameProcessor

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
DT = 1000 / 30


class FakeFace:
    available = True

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(0.4, 0.6, 478), rng.uniform(0.2, 0.5, 478), np.zeros(478)])
        self.detection = FaceDetection(landmarks=pts, blendshapes={}, transform=None)
        self.present = True

    def detect(self, frame, timestamp_ms):
        return self.detection if self.present else None

    def close(self) -> None:
        pass


class FakeObjects:
    available = True

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [Detection(label="cell phone", kind="phone", conf=0.9, bbox=[0.3, 0.3, 0.08, 0.1])]


class FakeEmotionModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, crop):
        assert crop.shape == (224, 224, 3)
        self.calls += 1
        return {c: (1.0 if c == "happy" else 0.0) for c in get_settings().emotion.classes}


def make_processor(settings: Settings | None = None):
    settings = settings or get_settings()
    proc = FrameProcessor(settings)
    proc.face.close()
    proc.face = FakeFace()
    proc.objects = FakeObjects()
    model = FakeEmotionModel()
    proc.estimators["emotion"] = EmotionEstimator(settings.emotion, model)
    return proc, model


def run(proc, n, t0=0.0):
    results = [proc.process(FRAME, t0 + i * DT, i) for i in range(n)]
    return results, t0 + n * DT


def test_heavy_models_run_every_n_frames() -> None:
    proc, model = make_processor()
    results, _ = run(proc, 15)
    s = get_settings()
    assert proc.objects.calls == len(range(0, 15, s.objects.every_n_frames))
    assert model.calls == len(range(0, 15, s.emotion.every_n_frames))
    last = results[-1]
    assert last.objects.available and last.objects.detections[0].label == "cell phone"
    assert last.emotion.label == "happy"
    assert {"landmarks_ms", "objects_ms", "emotion_ms", "pipeline_ms"} <= set(last.perf.components)


def test_phone_activity_reaches_distraction() -> None:
    proc, _ = make_processor()
    results, _ = run(proc, 90)  # 3 s with a phone beside the face
    last = results[-1]
    assert last.objects.activities.phone_use
    assert last.distraction.label in ("looking_away", "distracted")
    assert any(r.startswith("phone use") for r in last.distraction.reasons)


def test_seek_restarts_frame_counter_and_activities() -> None:
    proc, _ = make_processor()
    run(proc, 90, t0=60_000)
    calls = proc.objects.calls
    result = proc.process(FRAME, 0.0, 999)  # seek back to the start
    assert proc.objects.calls == calls + 1  # YOLO runs on the first frame after the seek
    assert not result.objects.activities.phone_use


def test_no_face_frame_still_runs_objects() -> None:
    proc, _ = make_processor()
    proc.face.present = False
    results, _ = run(proc, 45)
    last = results[-1]
    assert last.status == "no_face"
    assert last.objects.activities.phone_use  # phone anywhere counts without a face
    assert last.emotion.label == "unknown"
    assert last.distraction.label != "unknown"


def test_missing_models_are_unavailable_not_a_crash() -> None:
    s = get_settings()
    settings = s.model_copy(update={
        "objects": s.objects.model_copy(update={"model_path": "models_store/nope.pt"}),
        "emotion": s.emotion.model_copy(update={"model_path": "models_store/nope.onnx"}),
    })
    proc = FrameProcessor(settings)
    result = proc.process(FRAME, 0.0, 0)
    assert result.objects.available is False
    assert result.emotion is None
    proc.close()


@pytest.mark.skipif(not get_settings().objects.resolved_model_path().is_file(), reason="YOLO weights not downloaded")
def test_real_yolo_runs_on_blank_frame() -> None:
    proc = FrameProcessor(get_settings())
    assert proc.objects.available
    result = proc.process(FRAME, 0.0, 0)
    assert result.objects.available
    assert result.objects.detections == []
    proc.close()
