"""/ws/stream WebSocket: receives frames, runs the pipeline, returns FrameResult JSON.

Wire format (client → server), one binary message per frame, little-endian:
    float64 timestamp_ms | uint32 frame_id | JPEG bytes
The client keeps exactly one frame in flight, so frames are handled strictly in order.
Text messages are JSON control commands (see `handle_command`); valid ones get no reply.
"""

from __future__ import annotations

import json
import logging
import struct
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.pipeline.frame_processor import FrameProcessor
from app.schemas import FrameResult, PerfInfo
from app.utils.perf import PerfTracker

logger = logging.getLogger(__name__)
router = APIRouter()

HEADER = struct.Struct("<dI")  # timestamp_ms (float64), frame_id (uint32)


class BadMessage(ValueError):
    """Raised when a binary message is too short to contain the header."""


def parse_frame_message(data: bytes) -> tuple[float, int, bytes]:
    """Split a frame message into (timestamp_ms, frame_id, jpeg_bytes)."""
    if len(data) < HEADER.size:
        raise BadMessage(f"message has {len(data)} bytes, header needs {HEADER.size}")
    timestamp_ms, frame_id = HEADER.unpack_from(data)
    return timestamp_ms, frame_id, data[HEADER.size:]


def decode_jpeg(jpeg: bytes) -> np.ndarray | None:
    """Decode JPEG bytes into a BGR image, or None if they are not a valid image."""
    if not jpeg:
        return None
    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


def handle_command(processor: FrameProcessor, text: str) -> bool:
    """Apply a JSON control command. Returns False if the text is not a valid command.

    Commands: {"type": "calibrate"}, {"type": "clear_calibration"},
              {"type": "set_baseline", "baseline": {ear_open, mar_closed, yaw, pitch, gaze_h, gaze_v}}
    """
    try:
        command = json.loads(text)
        kind = command.get("type") if isinstance(command, dict) else None
        if kind == "calibrate":
            processor.start_calibration()
        elif kind == "clear_calibration":
            processor.clear_calibration()
        elif kind == "set_baseline":
            processor.set_baseline(command.get("baseline") or {})
        else:
            raise ValueError(f"unknown command type {kind!r}")
    except ValueError as exc:  # includes json.JSONDecodeError
        logger.warning("Bad control message: %s", exc)
        return False
    logger.info("Control command: %s", kind)
    return True


def _handle(processor: FrameProcessor, data: bytes) -> FrameResult:
    """Parse, decode and process one message (runs in a worker thread)."""
    try:
        timestamp_ms, frame_id, jpeg = parse_frame_message(data)
    except BadMessage as exc:
        logger.warning("Bad frame message: %s", exc)
        return FrameResult(frame_id=0, timestamp_ms=0.0, status="bad_message")
    frame = decode_jpeg(jpeg)
    if frame is None:
        logger.warning("Frame %d: JPEG decode failed (%d bytes)", frame_id, len(jpeg))
    return processor.process(frame, timestamp_ms, frame_id)


@router.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    """One connection = one stream session with its own processor and perf tracker."""
    settings = get_settings()
    await websocket.accept()
    processor = FrameProcessor(settings)
    perf = PerfTracker(settings.stream.perf_window_frames)
    logger.info("Stream connected: %s", websocket.client)
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            text = message.get("text")
            # A valid control command gets no reply, so the client's one-frame-in-flight loop is unaffected.
            if text is not None and handle_command(processor, text):
                continue
            data = message.get("bytes") or b""  # an invalid text message is answered as a bad message
            start = time.perf_counter()
            # Awaited one message at a time: the thread only keeps the event loop free.
            result = await run_in_threadpool(_handle, processor, data)
            backend_ms = (time.perf_counter() - start) * 1000.0
            perf.add(backend_ms)
            snap = perf.snapshot()
            result.perf = PerfInfo(
                backend_ms=backend_ms,
                backend_p50_ms=snap["p50_ms"],
                backend_p95_ms=snap["p95_ms"],
                fps=snap["fps"],
                components=result.perf.components,
            )
            await websocket.send_json(result.model_dump())
    except WebSocketDisconnect:
        logger.info("Stream disconnected: %s (%d frames in window)", websocket.client, len(perf))
    finally:
        await run_in_threadpool(processor.close)
