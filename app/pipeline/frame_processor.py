"""Orchestrates one frame end to end.

This is the single pipeline used by the live WebSocket and (Phase 6) the offline
evaluation runner. It knows nothing about transport: input is a decoded BGR image
plus the frame timestamp/id, output is a `FrameResult`.
"""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.pipeline.face import FaceAnalyzer
from app.pipeline.features import extract_features, face_bbox, overlay_landmarks
from app.schemas import FaceBlock, FrameResult


class FrameProcessor:
    """Per-stream processor. One instance per session (owns the VIDEO-mode landmarker and later temporal state)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.face = FaceAnalyzer(settings.face) if settings.face.enabled else None

    def process(self, frame: np.ndarray | None, timestamp_ms: float, frame_id: int) -> FrameResult:
        """Process one frame. `frame` is BGR (OpenCV), or None if it could not be decoded."""
        if frame is None:
            return FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status="decode_error")
        if self.face is None or not self.face.available:
            return FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status="ok")

        detection = self.face.detect(frame, timestamp_ms)
        if detection is None:
            # The analyzer may have become unavailable while recreating after a seek.
            status = "no_face" if self.face.available else "ok"
            block = FaceBlock(available=self.face.available, detected=False)
            return FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status=status, face=block)

        h, w = frame.shape[:2]
        block = FaceBlock(
            available=True,
            detected=True,
            bbox=face_bbox(detection.landmarks),
            features=extract_features(detection.landmarks, detection.blendshapes, detection.transform, w, h),
            landmarks=overlay_landmarks(detection.landmarks) if self.settings.face.send_landmarks else None,
        )
        return FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status="ok", face=block)

    def close(self) -> None:
        """Release native resources at the end of a stream."""
        if self.face is not None:
            self.face.close()
