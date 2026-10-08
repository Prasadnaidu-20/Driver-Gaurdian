"""Facial-expression (emotion) estimator: context only, it does not affect risk yet.

Model: HSEmotion EfficientNet-B0 trained on AffectNet (8 classes), as ONNX, run with
onnxruntime on CPU. Input is the `crop_face` output (RGB, square). The model returns logits.
They are converted with softmax, classes not in `emotion.classes` (contempt) are dropped and
the rest renormalized. Each class is then smoothed with a time-based EMA (~2 s), and the
smoothed vector is renormalized again.

Output: label = most likely class, confidence = its probability, probabilities = all classes,
score = summed probability of `negative_classes` (for the Phase 5 modifier; unused now).
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from app.config import EmotionSettings
from app.estimators.base import Estimator, FrameContext, TaskOutput, no_face_output
from app.pipeline.calibration import Baseline
from app.temporal.ema import EMA

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # HSEmotion uses ImageNet normalization
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(crop_rgb: np.ndarray, input_size: int) -> np.ndarray:
    """RGB uint8 crop → (1, 3, S, S) float32, ImageNet-normalized."""
    if crop_rgb.shape[:2] != (input_size, input_size):
        crop_rgb = cv2.resize(crop_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    x = (crop_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def map_probabilities(logits: np.ndarray, model_classes: list[str], classes: list[str]) -> dict[str, float]:
    """Softmax over the model's classes, keep `classes` (in that order) and renormalize."""
    z = np.asarray(logits, dtype=np.float64).ravel()
    p = np.exp(z - z.max())
    p /= p.sum()
    by_name = dict(zip(model_classes, p))
    kept = np.array([by_name[c] for c in classes])
    total = kept.sum()
    kept = kept / total if total > 0 else np.full(len(classes), 1.0 / len(classes))
    return {c: float(v) for c, v in zip(classes, kept)}


@lru_cache(maxsize=2)
def _load_session(path: str) -> Any:
    """Load the ONNX session once per process. Raises on failure (not cached)."""
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


class EmotionModel:
    """ONNX facial-expression model. `predict` returns None on error; never raises."""

    _lock = threading.Lock()

    def __init__(self, cfg: EmotionSettings) -> None:
        self.cfg = cfg
        self.available = False
        path = cfg.resolved_model_path()
        if not path.is_file():
            logger.error("Emotion model not found at %s (run scripts/download_models.py): emotion unavailable", path)
            return
        try:
            self._session = _load_session(str(path))
            self._input = self._session.get_inputs()[0].name
            n_out = self._session.get_outputs()[0].shape[-1]
        except Exception as exc:
            logger.error("Could not load emotion model from %s: %s — emotion unavailable", path, exc)
            return
        if isinstance(n_out, int) and n_out != len(cfg.model_classes):
            logger.error("Emotion model has %d outputs but %d model_classes are configured: emotion unavailable",
                         n_out, len(cfg.model_classes))
            return
        self.available = True

    def predict(self, crop_rgb: np.ndarray) -> dict[str, float] | None:
        """Per-class probabilities (configured `classes`) for one face crop."""
        if not self.available:
            return None
        try:
            with self._lock:
                logits = self._session.run(None, {self._input: preprocess(crop_rgb, self.cfg.input_size)})[0][0]
        except Exception:
            logger.exception("Emotion inference failed")
            return None
        return map_probabilities(logits, self.cfg.model_classes, self.cfg.classes)


class EmotionEstimator(Estimator):
    """Smoothed emotion from face crops. The model runs only on frames whose context carries a crop."""

    def __init__(self, cfg: EmotionSettings, model: Any) -> None:
        self.cfg = cfg
        self.model = model  # anything with predict(crop_rgb) -> dict | None (EmotionModel, or a fake in tests)
        self.reset()

    def set_baseline(self, baseline: Baseline) -> None:
        pass  # emotion is not relative to the driver baseline

    def reset(self) -> None:
        self._ema = {c: EMA(self.cfg.ema_tau_s) for c in self.cfg.classes}
        self._last_face_output: TaskOutput | None = None

    def update(self, features: dict[str, float] | None, timestamp_ms: float,
               context: FrameContext | None = None) -> TaskOutput:
        if features is None:
            return no_face_output()  # the EMA is kept, so a short dropout doesn't restart smoothing
        crop = context.face_crop if context is not None else None
        probs = self.model.predict(crop) if crop is not None else None
        if probs is None:
            # Skipped frame (or a failed inference): reuse the last estimate.
            return self._last_face_output or TaskOutput(reasons=["waiting for first emotion estimate"])

        smoothed = {c: self._ema[c].update(probs[c], timestamp_ms) for c in self.cfg.classes}
        total = sum(smoothed.values()) or 1.0
        smoothed = {c: v / total for c, v in smoothed.items()}
        label = max(smoothed, key=smoothed.__getitem__)
        self._last_face_output = TaskOutput(
            score=sum(smoothed[c] for c in self.cfg.negative_classes),
            label=label,
            confidence=smoothed[label],
            probabilities=smoothed,
            details={"raw_top_p": max(probs.values())},
        )
        return self._last_face_output
