"""Orchestrates one frame end to end.

This is the single pipeline used by the live WebSocket and (Phase 6) the offline
evaluation runner. It knows nothing about transport: input is a decoded BGR image
plus the frame timestamp/id, output is a `FrameResult`. Calibration is controlled through
`start_calibration` / `set_baseline` / `clear_calibration`.

Per frame: landmarks → objects (YOLO every `objects.every_n_frames`) → activities →
estimators. Emotion gets a face crop every `emotion.every_n_frames` frames; in between, the
heavy models' last results are reused. The frame counter restarts after a seek, so the
models run on the first frame after it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from app.config import Settings
from app.estimators import build_estimators
from app.estimators.base import Activities, FrameContext
from app.estimators.objects import ActivityRules, Detection, ObjectDetector
from app.pipeline.calibration import Baseline, Calibrator
from app.pipeline.face import FaceAnalyzer, FaceDetection
from app.pipeline.features import extract_features, face_bbox, mouth_center, overlay_landmarks
from app.pipeline.preprocess import crop_face
from app.schemas import CalibrationBlock, FaceBlock, FrameResult, ObjectsBlock

logger = logging.getLogger(__name__)


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


class FrameProcessor:
    """Per-stream processor. One instance per session: owns the VIDEO-mode landmarker, calibration and temporal state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.face = FaceAnalyzer(settings.face) if settings.face.enabled else None
        self.objects = ObjectDetector(settings.objects) if settings.objects.enabled else None
        self.activity_rules = ActivityRules(settings.objects)
        self.calibrator = Calibrator(settings.calibration)
        self.estimators = build_estimators(settings, self.calibrator.baseline)
        self._last_ts: float | None = None
        # Last measured time per component (ms); heavy models keep their latest run's time.
        self._timings: dict[str, float] = {}
        self._reset_frame_state()

    def _reset_frame_state(self) -> None:
        self._frame_index = 0                       # frames since stream start / last seek
        self._detections: list[Detection] = []      # latest YOLO detections
        self._activities: Activities | None = None  # latest activities (None until YOLO has run)

    # ---- calibration control ----

    def start_calibration(self) -> None:
        """Start (or restart) calibration; it runs on the following frames."""
        self.calibrator.start()

    def set_baseline(self, data: dict[str, Any]) -> None:
        """Use a previously computed baseline (e.g. restored by the browser). Raises ValueError if invalid."""
        self._apply_baseline(Baseline.from_dict(data))

    def clear_calibration(self) -> None:
        """Go back to the config default baseline ("uncalibrated")."""
        self._apply_baseline(Baseline.defaults(self.settings.calibration))

    def _apply_baseline(self, baseline: Baseline) -> None:
        self.calibrator.set_baseline(baseline)
        for estimator in self.estimators.values():
            estimator.set_baseline(baseline)

    # ---- per frame ----

    def _check_time(self, timestamp_ms: float) -> None:
        """A large backward jump (video seek / replay) invalidates all temporal state."""
        if self._last_ts is not None and self._last_ts - timestamp_ms > self.settings.face.timestamp_reset_ms:
            logger.info("Timestamp jumped back %.0f ms: resetting temporal state", self._last_ts - timestamp_ms)
            for estimator in self.estimators.values():
                estimator.reset()
            self.activity_rules.reset()
            self._reset_frame_state()
            self.calibrator.restart_if_running()
        self._last_ts = timestamp_ms

    def _calibration_block(self) -> CalibrationBlock:
        c = self.calibrator
        return CalibrationBlock(
            state=c.state, progress=c.progress, remaining_s=c.remaining_s,
            message=c.message, baseline=c.baseline.to_dict(),
        )

    def _due(self, every_n: int) -> bool:
        """True on frames where a heavy model should run."""
        return self._frame_index % max(1, every_n) == 0

    def _run_objects(self, frame: np.ndarray, detection: FaceDetection | None, timestamp_ms: float) -> ObjectsBlock:
        """YOLO every N frames. Activity rules are fed only on frames where YOLO ran."""
        if self.objects is None or not self.objects.available:
            return ObjectsBlock(available=False)
        if self._due(self.settings.objects.every_n_frames):
            start = time.perf_counter()
            self._detections = self.objects.detect(frame)
            self._timings["objects_ms"] = _ms_since(start)
            h, w = frame.shape[:2]
            fbox = face_bbox(detection.landmarks) if detection is not None else None
            mouth = mouth_center(detection.landmarks) if detection is not None else None
            self._activities = self.activity_rules.update(self._detections, fbox, mouth, (w, h), timestamp_ms)
            self._detections = self.activity_rules.detections  # phone-at-mouth relabelled as drink
        return ObjectsBlock(available=True, detections=self._detections,
                            activities=self._activities or Activities())

    def _emotion_crop(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray | None:
        """Face crop for emotion on frames where it is due, else None (the estimator reuses its last result)."""
        if "emotion" not in self.estimators or not self._due(self.settings.emotion.every_n_frames):
            return None
        face_cfg = self.settings.face
        return crop_face(frame, detection.landmarks, face_cfg.crop_size, face_cfg.crop_margin)

    def _run_tasks(self, result: FrameResult, features: dict[str, float] | None, timestamp_ms: float,
                   context: FrameContext) -> None:
        """Calibration + estimators for one frame (features=None means no face)."""
        new_baseline = self.calibrator.feed(features, timestamp_ms)
        if new_baseline is not None:
            self._apply_baseline(new_baseline)
        for name, estimator in self.estimators.items():
            start = time.perf_counter()
            setattr(result, name, estimator.update(features, timestamp_ms, context))
            if name == "emotion" and context.face_crop is not None:
                self._timings["emotion_ms"] += _ms_since(start)  # crop time was recorded by the caller
        result.calibration = self._calibration_block()

    def process(self, frame: np.ndarray | None, timestamp_ms: float, frame_id: int) -> FrameResult:
        """Process one frame. `frame` is BGR (OpenCV), or None if it could not be decoded."""
        if frame is None:
            return FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status="decode_error",
                               calibration=self._calibration_block())
        total_start = time.perf_counter()
        self._check_time(timestamp_ms)

        detection: FaceDetection | None = None
        face_ok = self.face is not None and self.face.available
        if face_ok:
            start = time.perf_counter()
            detection = self.face.detect(frame, timestamp_ms)
            self._timings["landmarks_ms"] = _ms_since(start)
            face_ok = self.face.available  # may have become unavailable while recreating after a seek

        objects_block = self._run_objects(frame, detection, timestamp_ms)

        if not face_ok:
            # Face analysis unavailable or disabled: no estimators, but objects still run.
            result = FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status="ok",
                                 objects=objects_block, calibration=self._calibration_block())
        elif detection is None:
            result = FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status="no_face",
                                 face=FaceBlock(available=True, detected=False), objects=objects_block)
            self._run_tasks(result, None, timestamp_ms, FrameContext(activities=self._activities))
        else:
            h, w = frame.shape[:2]
            features = extract_features(detection.landmarks, detection.blendshapes, detection.transform, w, h)
            block = FaceBlock(
                available=True,
                detected=True,
                bbox=face_bbox(detection.landmarks),
                features=features,
                landmarks=overlay_landmarks(detection.landmarks) if self.settings.face.send_landmarks else None,
            )
            result = FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status="ok", face=block,
                                 objects=objects_block)
            start = time.perf_counter()
            crop = self._emotion_crop(frame, detection)
            if crop is not None:
                self._timings["emotion_ms"] = _ms_since(start)
            self._run_tasks(result, features, timestamp_ms, FrameContext(face_crop=crop, activities=self._activities))

        self._frame_index += 1
        result.perf.components = {**self._timings, "pipeline_ms": _ms_since(total_start)}
        return result

    def close(self) -> None:
        """Release native resources at the end of a stream."""
        if self.face is not None:
            self.face.close()
