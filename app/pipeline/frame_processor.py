"""Orchestrates one frame end to end.

This is the single pipeline used by the live WebSocket and (Phase 6) the offline
evaluation runner. It knows nothing about transport: input is a decoded BGR image
plus the frame timestamp/id, output is a `FrameResult`. Calibration is controlled through
`start_calibration` / `set_baseline` / `clear_calibration`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.config import Settings
from app.estimators import build_estimators
from app.pipeline.calibration import Baseline, Calibrator
from app.pipeline.face import FaceAnalyzer
from app.pipeline.features import extract_features, face_bbox, overlay_landmarks
from app.schemas import CalibrationBlock, FaceBlock, FrameResult

logger = logging.getLogger(__name__)


class FrameProcessor:
    """Per-stream processor. One instance per session: owns the VIDEO-mode landmarker, calibration and temporal state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.face = FaceAnalyzer(settings.face) if settings.face.enabled else None
        self.calibrator = Calibrator(settings.calibration)
        self.estimators = build_estimators(settings, self.calibrator.baseline)
        self._last_ts: float | None = None

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
            self.calibrator.restart_if_running()
        self._last_ts = timestamp_ms

    def _calibration_block(self) -> CalibrationBlock:
        c = self.calibrator
        return CalibrationBlock(
            state=c.state, progress=c.progress, remaining_s=c.remaining_s,
            message=c.message, baseline=c.baseline.to_dict(),
        )

    def _run_tasks(self, result: FrameResult, features: dict[str, float] | None, timestamp_ms: float) -> None:
        """Calibration + estimators for one frame (features=None means no face)."""
        new_baseline = self.calibrator.feed(features, timestamp_ms)
        if new_baseline is not None:
            self._apply_baseline(new_baseline)
        for name, estimator in self.estimators.items():
            setattr(result, name, estimator.update(features, timestamp_ms))
        result.calibration = self._calibration_block()

    def process(self, frame: np.ndarray | None, timestamp_ms: float, frame_id: int) -> FrameResult:
        """Process one frame. `frame` is BGR (OpenCV), or None if it could not be decoded."""
        if frame is None:
            return FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status="decode_error",
                               calibration=self._calibration_block())
        if self.face is None or not self.face.available:
            return FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status="ok",
                               calibration=self._calibration_block())

        self._check_time(timestamp_ms)
        detection = self.face.detect(frame, timestamp_ms)
        if detection is None:
            # The analyzer may have become unavailable while recreating after a seek.
            status = "no_face" if self.face.available else "ok"
            block = FaceBlock(available=self.face.available, detected=False)
            result = FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status=status, face=block)
            self._run_tasks(result, None, timestamp_ms)
            return result

        h, w = frame.shape[:2]
        features = extract_features(detection.landmarks, detection.blendshapes, detection.transform, w, h)
        block = FaceBlock(
            available=True,
            detected=True,
            bbox=face_bbox(detection.landmarks),
            features=features,
            landmarks=overlay_landmarks(detection.landmarks) if self.settings.face.send_landmarks else None,
        )
        result = FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status="ok", face=block)
        self._run_tasks(result, features, timestamp_ms)
        return result

    def close(self) -> None:
        """Release native resources at the end of a stream."""
        if self.face is not None:
            self.face.close()
