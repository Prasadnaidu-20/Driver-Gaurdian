"""Backend→frontend contract: the JSON message sent for every processed frame."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

FrameStatus = Literal["ok", "bad_message", "decode_error"]


class PerfInfo(BaseModel):
    """Server-side performance numbers. Client round-trip latency is measured in the browser."""

    backend_ms: float = 0.0       # backend processing time for this frame (decode + pipeline)
    backend_p50_ms: float = 0.0   # rolling median backend processing time
    backend_p95_ms: float = 0.0   # rolling 95th-percentile backend processing time
    fps: float = 0.0              # observed stream/backend FPS (frames completed per second)


class FrameResult(BaseModel):
    """Result for one frame. Later phases add face / task / risk blocks here."""

    frame_id: int
    timestamp_ms: float  # client-supplied frame timestamp, echoed back
    status: FrameStatus
    perf: PerfInfo = PerfInfo()
