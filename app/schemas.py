"""Backend→frontend contract: the JSON message sent for every processed frame."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ok: frame processed (face found, or face analysis unavailable/disabled — see face.available)
# no_face: face analysis ran and found no face
FrameStatus = Literal["ok", "no_face", "bad_message", "decode_error"]


class PerfInfo(BaseModel):
    """Server-side performance numbers. Client round-trip latency is measured in the browser."""

    backend_ms: float = 0.0       # backend processing time for this frame (decode + pipeline)
    backend_p50_ms: float = 0.0   # rolling median backend processing time
    backend_p95_ms: float = 0.0   # rolling 95th-percentile backend processing time
    fps: float = 0.0              # observed stream/backend FPS (frames completed per second)


class FaceBlock(BaseModel):
    """Face analysis for one frame. Coordinates are normalized to the sent frame (0-1)."""

    available: bool = False                    # Face Landmarker loaded and enabled
    detected: bool = False                     # a face was found in this frame
    bbox: list[float] | None = None            # [x, y, w, h], normalized
    features: dict[str, float] = Field(default_factory=dict)  # see app.pipeline.features.FEATURE_KEYS
    landmarks: dict[str, list[list[float]]] | None = None     # overlay groups -> [[x, y], ...], normalized


class FrameResult(BaseModel):
    """Result for one frame. Later phases add task / risk blocks here."""

    frame_id: int
    timestamp_ms: float  # client-supplied frame timestamp, echoed back
    status: FrameStatus
    perf: PerfInfo = Field(default_factory=PerfInfo)
    face: FaceBlock = Field(default_factory=FaceBlock)
