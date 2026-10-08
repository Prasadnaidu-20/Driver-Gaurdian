# DriveGuardian — Progress

## Phase 0 — Project setup
*(Back-filled during Phase 1: this file was missing although Phase 0 asked for it.)*

**Built:** folder structure from CLAUDE.md, `.gitignore`, `requirements.txt` (fastapi, uvicorn[standard], opencv-python, numpy, pydantic, pyyaml, psutil, pytest, httpx), `config/default.yaml` with all top-level sections, `app/config.py` (typed settings), `app/main.py` (`GET /health`, static `web/` at `/`), placeholder `web/index.html`, README stub, `tests/test_config.py`.

**Environment:** venv on Python 3.12.8 (inside MediaPipe's supported 3.9–3.12 range).

---

## Phase 1 — Live video loop (webcam/video → backend → dashboard)

### What was built
- **Wire protocol** (`/ws/stream`): one binary message per frame, **little-endian**: `float64 timestamp_ms | uint32 frame_id | JPEG bytes` (Python `struct "<dI"`, JS `DataView(..., true)`). The reply is `FrameResult` JSON.
- **Flow control:** exactly one frame in flight. The browser captures and sends the next frame only after the previous result arrives. The server handles messages strictly one at a time, with no buffering and no parallel processing. Decode and processing run in a worker thread (`run_in_threadpool`), but each call is awaited before the next message is read; the thread only keeps the event loop free for later heavy models.
- `app/schemas.py`: `FrameResult{frame_id, timestamp_ms, status, perf}` and `PerfInfo{backend_ms, backend_p50_ms, backend_p95_ms, fps}`. `status` is `ok | bad_message | decode_error`.
- `app/pipeline/frame_processor.py`: stub `FrameProcessor.process(frame_bgr | None, timestamp_ms, frame_id) -> FrameResult`. It has **no transport dependency** (no FastAPI or WebSocket imports), so Phase 2 replaces its body without touching the protocol. There is one instance per connection, ready for per-session temporal state.
- `app/api/ws.py`: `parse_frame_message`, `decode_jpeg` (OpenCV), and the WebSocket handler with a per-connection `PerfTracker`.
- `app/utils/perf.py`: `PerfTracker(window)` gives rolling FPS ((n−1)/time span) and latency percentiles over the last N frames. It uses `time.perf_counter()`.
- `app/main.py`: includes the WS router and adds `GET /api/config` (stream settings for the browser).
- `config/default.yaml`: `stream.perf_window_frames: 60`.
- Dashboard (`web/`): Start webcam / Load video file / Stop; `<video>` with a transparent overlay canvas (empty for now); Performance card; placeholder cards for Driver State, Drowsiness, Distraction, Emotion, Risk and Alerts. Frames are drawn to an offscreen canvas at the configured size (640×480) and JPEG-encoded at the configured quality (0.7). Webcam timestamps use `performance.now()`; video files use `video.currentTime * 1000`. A video file stops sending while paused or seeking, resumes on play/seeked, and closes the stream on `ended`.

### Three separate measurements
| Dashboard label | What it is | Where measured |
|---|---|---|
| **Stream/backend FPS** | Observed frames completed per second over the last 60 frames. This is *not* ML inference FPS; there is no model in Phase 1. With one frame in flight it is bounded by the full loop (capture + encode + network + backend). | server, `PerfTracker` |
| **Round-trip latency (client) p50/p95** | From `ws.send` to the result arriving in the browser. Excludes canvas capture and JPEG encode. | browser, rolling 60 frames |
| **Backend processing latency** now / p50 / p95 | Parse + JPEG decode + `FrameProcessor` on the server. | server, `PerfTracker` |

### Measured (automated, no browser)
Python WebSocket client against a live uvicorn server on this machine, 300 frames of 640×480 random-noise JPEG (170 KB, worst case for size and decode): backend processing p50 4.2 ms / p95 4.9 ms; client round trip p50 16.0 ms / p95 17.6 ms; stream/backend FPS ≈ 61. Browser numbers will be lower because canvas capture and JPEG encode happen in the loop. Record the real figures from manual test 1 below.

### How to run
```powershell
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```
Open http://localhost:8000. Webcam access needs `localhost`/`127.0.0.1` or HTTPS.

### How to test
- `pytest -q` (15 tests): `tests/test_perf.py` (FPS, percentiles, window eviction), `tests/test_ws.py` (protocol parsing incl. max uint32, transport-independent processor, `/api/config`, WebSocket round trip, and the socket staying alive after a bad JPEG, a short message or a text message).

### Deviations from the spec / decisions
- **Byte order:** the spec doesn't name one, so it is fixed as little-endian.
- **Extra fields** in `perf`: `backend_p50_ms`, `backend_p95_ms` (additive).
- **`GET /api/config`** added so the browser reads frame size and JPEG quality from the YAML instead of hard-coding them.
- **`bad_message` status:** a malformed or text message gets a reply instead of closing the socket, so the one-in-flight client loop can't stall.

### Known issues / notes
- Video-file timestamps go backwards on seek or replay. That's harmless now; Phase 2 (MediaPipe VIDEO mode) has to guard against non-increasing timestamps, as PHASES.md already notes.
- After a video ends, the stream closes. Load the file again to restream it (pressing play on an ended video won't restart streaming).
- `fastapi.testclient` prints a Starlette deprecation warning about `httpx` (tests still pass).
