"""MediaPipe Face Landmarker wrapper (VIDEO running mode).

VIDEO mode tracks landmarks across frames and requires strictly increasing integer
millisecond timestamps. Client timestamps can repeat (rAF faster than the video's frame
rate) or jump backwards (seek / replay), so `MonotonicClock` maps them to valid
MediaPipe timestamps; a large backward jump recreates the landmarker so stale tracking
state is dropped. The original client timestamp is still used everywhere else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import FaceSettings

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """One detected face."""

    landmarks: np.ndarray            # (478, 3) normalized x, y (0-1), relative z
    blendshapes: dict[str, float]    # category name -> score (0-1)
    transform: np.ndarray | None     # 4×4 facial transformation matrix, if available


class MonotonicClock:
    """Maps client timestamps (ms) to strictly increasing integer MediaPipe timestamps."""

    def __init__(self, reset_ms: float) -> None:
        self.reset_ms = reset_ms
        self.last: int | None = None

    def next(self, timestamp_ms: float) -> tuple[int, bool]:
        """Return (mediapipe_ts, reset). `reset` is True when the caller must recreate the landmarker.

        * ts > last                    → ts
        * last - ts <= reset_ms        → last + 1 (duplicate or slightly out-of-order frame)
        * last - ts >  reset_ms        → ts, reset=True (seek / replay: start a fresh timeline)
        """
        ts = int(timestamp_ms)
        if self.last is None or ts > self.last:
            self.last = ts
            return ts, False
        if self.last - ts <= self.reset_ms:
            self.last += 1
            return self.last, False
        self.last = ts
        return ts, True


class FaceAnalyzer:
    """Face Landmarker in VIDEO mode, one instance per stream. Never raises from `detect`."""

    def __init__(self, settings: FaceSettings) -> None:
        self.settings = settings
        self.clock = MonotonicClock(settings.timestamp_reset_ms)
        self._landmarker = None
        self.available = False
        self._create()

    def _create(self) -> None:
        """(Re)create the landmarker; on failure mark the analyzer unavailable."""
        self.close()
        path: Path = self.settings.resolved_model_path()
        try:
            from mediapipe.tasks.python import BaseOptions, vision

            if not path.is_file():
                raise FileNotFoundError(f"{path} not found — run `python scripts/download_models.py`")
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=self.settings.num_faces,
                min_face_detection_confidence=self.settings.min_detection_confidence,
                min_face_presence_confidence=self.settings.min_presence_confidence,
                min_tracking_confidence=self.settings.min_tracking_confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self.available = True
        except Exception as exc:
            logger.error("Face Landmarker unavailable: %s", exc)
            self._landmarker = None
            self.available = False

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: float) -> FaceDetection | None:
        """Detect the driver's face. Returns None if no face (or if the analyzer is unavailable)."""
        if not self.available:
            return None
        mp_ts, reset = self.clock.next(timestamp_ms)
        if reset:
            logger.info("Timestamp went back to %d ms: recreating Face Landmarker", mp_ts)
            self._create()
            if not self.available:
                return None
        try:
            import mediapipe as mp

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), mp_ts)
        except Exception as exc:
            # Keep the stream alive: a failed frame is reported as no face.
            logger.warning("Face Landmarker failed on frame at %.1f ms: %s", timestamp_ms, exc)
            return None
        if not result.face_landmarks:
            return None
        landmarks = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float64)
        blendshapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]} if result.face_blendshapes else {}
        transform = np.asarray(result.facial_transformation_matrixes[0]) if result.facial_transformation_matrixes else None
        return FaceDetection(landmarks=landmarks, blendshapes=blendshapes, transform=transform)

    def close(self) -> None:
        """Release the native landmarker."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as exc:
                logger.warning("Error closing Face Landmarker: %s", exc)
            self._landmarker = None
