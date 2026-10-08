import json
import struct

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.ws import BadMessage, parse_frame_message
from app.config import get_settings
from app.main import app
from app.pipeline.frame_processor import FrameProcessor


def make_message(timestamp_ms: float, frame_id: int, payload: bytes) -> bytes:
    return struct.pack("<dI", timestamp_ms, frame_id) + payload


def make_jpeg(width: int = 640, height: int = 480) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    return buf.tobytes()


def test_parse_frame_message_round_trip() -> None:
    ts, fid, jpeg = parse_frame_message(make_message(1234.5, 42, b"abc"))
    assert ts == 1234.5
    assert fid == 42
    assert jpeg == b"abc"


def test_parse_frame_message_max_uint32() -> None:
    _, fid, _ = parse_frame_message(make_message(0.0, 2**32 - 1, b""))
    assert fid == 2**32 - 1


def test_parse_frame_message_too_short() -> None:
    with pytest.raises(BadMessage):
        parse_frame_message(b"\x00" * 11)


def test_frame_processor_is_transport_independent() -> None:
    processor = FrameProcessor(get_settings())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert processor.process(frame, 10.0, 1).status in ("ok", "no_face")  # no_face if the model is present
    assert processor.process(None, 20.0, 2).status == "decode_error"
    processor.close()


def test_api_config_returns_stream_settings() -> None:
    cfg = TestClient(app).get("/api/config").json()
    assert cfg["frame_width"] == 640
    assert cfg["frame_height"] == 480
    assert cfg["jpeg_quality"] == 0.7
    assert cfg["perf_window_frames"] >= 2
    assert cfg["chart_window_s"] > 0


def test_ws_round_trip_ok() -> None:
    client = TestClient(app)
    jpeg = make_jpeg()
    with client.websocket_connect("/ws/stream") as ws:
        for fid in range(3):  # one frame in flight: send, wait for result, repeat
            ws.send_bytes(make_message(1000.0 + fid * 33.3, fid, jpeg))
            result = ws.receive_json()
            assert result["frame_id"] == fid
            assert result["timestamp_ms"] == pytest.approx(1000.0 + fid * 33.3)
            assert result["status"] in ("ok", "no_face")  # synthetic frame has no face
            assert {"available", "detected", "bbox", "features", "landmarks"} <= set(result["face"])
            assert result["perf"]["backend_ms"] > 0
            assert set(result["perf"]) == {"backend_ms", "backend_p50_ms", "backend_p95_ms", "fps"}
        assert result["perf"]["fps"] > 0


def test_ws_bad_input_keeps_socket_alive() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(make_message(5.0, 7, b"not a jpeg"))
        result = ws.receive_json()
        assert result["status"] == "decode_error"
        assert result["frame_id"] == 7

        ws.send_bytes(b"\x01\x02")
        assert ws.receive_json()["status"] == "bad_message"

        ws.send_text("hello")
        assert ws.receive_json()["status"] == "bad_message"

        ws.send_bytes(make_message(6.0, 8, make_jpeg()))
        result = ws.receive_json()
        assert result["status"] in ("ok", "no_face")
        assert result["frame_id"] == 8


def test_ws_result_has_calibration_and_task_blocks() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(make_message(0.0, 0, make_jpeg()))
        result = ws.receive_json()
    assert result["calibration"]["state"] == "uncalibrated"
    assert result["calibration"]["baseline"]["calibrated"] is False
    if result["face"]["available"]:  # model downloaded: blank frame → no face → unknown
        assert result["drowsiness"]["label"] == "unknown"
        assert result["distraction"]["label"] == "unknown"
    else:
        assert result["drowsiness"] is None and result["distraction"] is None


def test_ws_control_commands() -> None:
    client = TestClient(app)
    jpeg = make_jpeg()
    baseline = {"ear_open": 0.31, "mar_closed": 0.03, "yaw": 5.0, "pitch": -4.0, "gaze_h": 0.5, "gaze_v": 0.5}
    with client.websocket_connect("/ws/stream") as ws:
        # Valid commands get no reply: the next message received is the frame result.
        ws.send_text(json.dumps({"type": "set_baseline", "baseline": baseline}))
        ws.send_bytes(make_message(0.0, 1, jpeg))
        result = ws.receive_json()
        assert result["frame_id"] == 1
        assert result["calibration"]["state"] == "calibrated"
        assert result["calibration"]["baseline"]["yaw"] == 5.0

        ws.send_text(json.dumps({"type": "clear_calibration"}))
        ws.send_bytes(make_message(33.0, 2, jpeg))
        assert ws.receive_json()["calibration"]["state"] == "uncalibrated"

        ws.send_text(json.dumps({"type": "set_baseline", "baseline": {"ear_open": 0.3}}))
        assert ws.receive_json()["status"] == "bad_message"  # invalid baseline
        ws.send_text(json.dumps({"type": "dance"}))
        assert ws.receive_json()["status"] == "bad_message"


def test_ws_calibration_progress_on_frames() -> None:
    client = TestClient(app)
    jpeg = make_jpeg()
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(make_message(0.0, 0, jpeg))
        face_available = ws.receive_json()["face"]["available"]
        if not face_available:
            pytest.skip("face model not downloaded: calibration needs face analysis")
        ws.send_text(json.dumps({"type": "calibrate"}))
        for fid, ts in enumerate([100.0, 2100.0, 4100.0], start=1):
            ws.send_bytes(make_message(ts, fid, jpeg))
            cal = ws.receive_json()["calibration"]
            assert cal["state"] == "calibrating"
        assert cal["progress"] == pytest.approx(0.4)
        assert "Look at the road" in cal["message"]
        # Blank frames have no face, so after the full duration calibration fails cleanly.
        ws.send_bytes(make_message(10_200.0, 9, jpeg))
        cal = ws.receive_json()["calibration"]
        assert cal["state"] == "uncalibrated"
        assert cal["message"].startswith("Calibration failed")
