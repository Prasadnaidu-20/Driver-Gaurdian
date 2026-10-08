"""Orchestrates one frame end to end.

This is the single pipeline used by the live WebSocket and (Phase 6) the offline
evaluation runner. It knows nothing about transport: input is a decoded BGR image
plus the frame timestamp/id, output is a `FrameResult`.
"""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.schemas import FrameResult


class FrameProcessor:
    """Per-stream processor. One instance per session, because later phases keep temporal state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def process(self, frame: np.ndarray | None, timestamp_ms: float, frame_id: int) -> FrameResult:
        """Process one frame. `frame` is BGR (OpenCV), or None if it could not be decoded.

        Phase 1 stub: no analysis yet, only reports whether a frame was received.
        """
        status = "ok" if frame is not None else "decode_error"
        return FrameResult(frame_id=frame_id, timestamp_ms=timestamp_ms, status=status)
