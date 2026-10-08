"""Backend→frontend contract: the JSON message sent for every processed frame."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.estimators.base import Activities, TaskOutput
from app.estimators.objects import Detection
from app.pipeline.calibration import CalibrationState

# ok: frame processed (face found, or face analysis unavailable/disabled — see face.available)
# no_face: face analysis ran and found no face
FrameStatus = Literal["ok", "no_face", "bad_message", "decode_error"]


class PerfInfo(BaseModel):
    """Server-side performance numbers. Client round-trip latency is measured in the browser."""

    backend_ms: float = 0.0       # backend processing time for this frame (decode + pipeline)
    backend_p50_ms: float = 0.0   # rolling median backend processing time
    backend_p95_ms: float = 0.0   # rolling 95th-percentile backend processing time
    fps: float = 0.0              # observed stream/backend FPS (frames completed per second)
    # Last measured time (ms) of each pipeline component: landmarks_ms, objects_ms, emotion_ms
    # (heavy models run every N frames, so these are their latest run) and pipeline_ms (this frame).
    components: dict[str, float] = Field(default_factory=dict)


class FaceBlock(BaseModel):
    """Face analysis for one frame. Coordinates are normalized to the sent frame (0-1)."""

    available: bool = False                    # Face Landmarker loaded and enabled
    detected: bool = False                     # a face was found in this frame
    bbox: list[float] | None = None            # [x, y, w, h], normalized
    features: dict[str, float] = Field(default_factory=dict)  # see app.pipeline.features.FEATURE_KEYS
    landmarks: dict[str, list[list[float]]] | None = None     # overlay groups -> [[x, y], ...], normalized


class ObjectsBlock(BaseModel):
    """Object detection and activities. Detections are from the latest YOLO run (every N frames)."""

    available: bool = False                                    # YOLO loaded and enabled
    detections: list[Detection] = Field(default_factory=list)  # phone / cup / bottle boxes
    activities: Activities = Field(default_factory=Activities)


class CalibrationBlock(BaseModel):
    """Calibration status. `baseline` is the baseline currently in use (defaults if uncalibrated)."""

    state: CalibrationState = "uncalibrated"
    progress: float = 0.0       # 0-1 while calibrating
    remaining_s: float = 0.0    # frame-time seconds left while calibrating
    message: str = ""           # instruction, "calibrated", or a failure reason
    baseline: dict[str, float | bool] | None = None


class FrameResult(BaseModel):
    """Result for one frame. Phase 5 adds risk blocks here."""

    frame_id: int
    timestamp_ms: float  # client-supplied frame timestamp, echoed back
    status: FrameStatus
    perf: PerfInfo = Field(default_factory=PerfInfo)
    face: FaceBlock = Field(default_factory=FaceBlock)
    calibration: CalibrationBlock = Field(default_factory=CalibrationBlock)
    drowsiness: TaskOutput | None = None   # None when disabled or face analysis unavailable
    distraction: TaskOutput | None = None  # None when disabled or face analysis unavailable
    emotion: TaskOutput | None = None      # None when disabled or its model is unavailable
    objects: ObjectsBlock = Field(default_factory=ObjectsBlock)
