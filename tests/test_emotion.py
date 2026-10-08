"""Emotion: class mapping, preprocessing, EMA smoothing and skipped frames (fake model), plus a real-model smoke test."""

import numpy as np
import pytest

from app.config import get_settings, load_settings
from app.estimators.base import FrameContext
from app.estimators.emotion import EmotionEstimator, EmotionModel, map_probabilities, preprocess

CFG = load_settings().emotion
FPS = 30
DT = 1000 / FPS
CROP = np.zeros((224, 224, 3), dtype=np.uint8)
FEATURES = {"ear_mean": 0.3}  # emotion only checks that a face is present


def onehot(name: str) -> dict[str, float]:
    return {c: (1.0 if c == name else 0.0) for c in CFG.classes}


class FakeModel:
    def __init__(self) -> None:
        self.current = onehot("neutral")
        self.calls = 0

    def predict(self, crop):
        self.calls += 1
        return dict(self.current)


def test_map_probabilities_drops_contempt_and_renormalizes() -> None:
    # Model order: anger, contempt, disgust, fear, happy, neutral, sad, surprise (contempt = 0.5).
    logits = np.log(np.array([0.1, 0.5, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05]))
    p = map_probabilities(logits, CFG.model_classes, CFG.classes)
    assert list(p) == CFG.classes
    assert "contempt" not in p
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["anger"] == pytest.approx(0.2)  # 0.1 / 0.5 after dropping contempt
    assert p["happy"] == pytest.approx(0.1)


def test_preprocess_shape_and_normalization() -> None:
    crop = np.full((100, 100, 3), 255, dtype=np.uint8)
    x = preprocess(crop, 224)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert x[0, 0, 0, 0] == pytest.approx((1 - 0.485) / 0.229, rel=1e-4)  # R channel


def run(est, t0, seconds, every_n=1, features=FEATURES):
    out, t = [], t0
    for i in range(round(seconds * FPS)):
        crop = CROP if i % every_n == 0 else None
        out.append(est.update(features, t, FrameContext(face_crop=crop)))
        t += DT
    return out, t


def test_emotion_follows_change_within_two_seconds() -> None:
    model = FakeModel()
    est = EmotionEstimator(CFG, model)
    out, t = run(est, 0.0, 3)
    assert out[-1].label == "neutral"
    model.current = onehot("happy")
    out, _ = run(est, t, 2.0)
    assert out[int(0.3 * FPS)].label == "neutral"  # smoothed: not an instant flip
    assert out[-1].label == "happy"
    assert out[-1].probabilities["happy"] > 0.9
    assert sum(out[-1].probabilities.values()) == pytest.approx(1.0)


def test_skipped_frames_reuse_last_output() -> None:
    model = FakeModel()
    est = EmotionEstimator(CFG, model)
    out, _ = run(est, 0.0, 1, every_n=3)
    assert model.calls == 10
    assert out[1] is out[0] and out[2] is out[0]


def test_no_face_is_unknown_and_score_is_negative_mass() -> None:
    model = FakeModel()
    model.current = {**onehot("neutral"), "neutral": 0.4, "anger": 0.6}
    est = EmotionEstimator(CFG, model)
    out, t = run(est, 0.0, 0.5)
    assert out[-1].score == pytest.approx(0.6)
    assert out[-1].label == "anger"
    out, _ = run(est, t, 0.2, features=None)
    assert out[-1].label == "unknown"


def test_waiting_before_first_estimate() -> None:
    est = EmotionEstimator(CFG, FakeModel())
    assert est.update(FEATURES, 0.0, FrameContext(face_crop=None)).label == "unknown"


@pytest.mark.skipif(not get_settings().emotion.resolved_model_path().is_file(), reason="emotion model not downloaded")
def test_real_model_runs_on_cpu() -> None:
    model = EmotionModel(get_settings().emotion)
    assert model.available
    p = model.predict(np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8))
    assert set(p) == set(CFG.classes)
    assert sum(p.values()) == pytest.approx(1.0)


def test_missing_model_is_unavailable() -> None:
    model = EmotionModel(CFG.model_copy(update={"model_path": "models_store/nope.onnx"}))
    assert not model.available
    assert model.predict(CROP) is None
