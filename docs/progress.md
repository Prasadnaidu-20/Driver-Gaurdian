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

---

## Phase 2 — Face landmarks and geometric features

### What was built
- **Dependency:** `mediapipe>=0.10.14` (installed: 1.1.0 on Python 3.12.8).
- `scripts/download_models.py`: downloads `face_landmarker.task` (3.8 MB) into `models_store/`, using the URL and path from config. It skips the download if the file exists (`--force` re-downloads) and writes via a `.part` file. Phase 4 adds its weights to `model_specs()`.
- `config/default.yaml`: `face:` is now typed (`FaceSettings`): model path/URL, confidences, `timestamp_reset_ms`, `send_landmarks`, `crop_size`, `crop_margin`. There is a new `ui:` section with `chart_window_s: 10`.
- `app/pipeline/face.py`:
  - `FaceAnalyzer` runs the Face Landmarker in VIDEO mode with `num_faces=1`, blendshapes and transformation matrices, one instance per WebSocket connection. `FrameProcessor.close()` releases it when the socket closes.
  - **Timestamp guard** (`MonotonicClock`): MediaPipe needs strictly increasing integer ms. A duplicate or slightly backward timestamp (≤ `timestamp_reset_ms`) is nudged to `last + 1`. A larger backward jump (video seek or replay) **recreates the landmarker**. The client timestamp is still the one echoed and used for all temporal logic.
  - A missing or broken model gives `face.available = false` and is logged once. An exception on a frame is reported as no face. The stream never crashes.
- `app/pipeline/features.py` (pure NumPy):
  - EAR left/right/mean (6-point) and MAR (8-point inner lip), computed in pixel space.
  - Head pose yaw/pitch/roll from the transformation matrix.
  - Gaze h/v iris ratios averaged over both eyes.
  - Blendshapes `eyeBlinkLeft`, `eyeBlinkRight`, `jawOpen`.
  - Normalized bbox.
  - Overlay point groups (eye contours, mouth, irises; 82 points, about 1.2 KB JSON).
- **Sign conventions** (documented in the `features.py` docstring; each sign is a `*_SIGN` constant, so a flip is a one-line fix):
  - yaw + = driver turns head to **their left**;
  - pitch + = **up** (looking down is negative, as Phase 3 assumes);
  - roll + = head tilts to the driver's **left** shoulder;
  - gaze_h 0 = image-left eye corner, 1 = image-right; gaze_v 0 = upper lid, 1 = lower lid; about 0.5/0.5 when centred.
  - Eye "left/right" = the driver's anatomical side (in an unmirrored image the driver's right eye is on the image left).
- `app/pipeline/preprocess.py`: `crop_face(frame_bgr, landmarks, size, margin)` gives a square RGB crop centred on the landmark bbox, black-padded at frame edges. It is not used live yet.
- `app/schemas.py`: `FrameStatus` gains `no_face`. `FrameResult.face = FaceBlock{available, detected, bbox [x,y,w,h] normalized, features dict, landmarks groups}`.
  - Status rules: `decode_error` if the JPEG is bad; `no_face` if analysis ran and found no face; otherwise `ok`. If the model is unavailable or disabled, the status is `ok` with `face.available=false`.
- Dashboard:
  - "Show landmarks" toggle; the overlay draws the face box (always) and eye/mouth/iris points.
  - Driver State card with a status badge (face detected / no face / model unavailable) and all live numbers.
  - Rolling 10 s Chart.js (CDN) line chart of EAR mean and MAR, with gaps while there is no face; it resets on a seek or a new stream.

### Measured (automated, no browser)
- Scripted WebSocket client against live uvicorn, 640×480 frames containing a face (MediaPipe sample portrait): **backend p50 21.1 ms / p95 24.2 ms**, client round trip p50 28.5 ms, stream FPS ≈ 27. That is the pipeline-bound ceiling with one frame in flight.
- On a frontal, smiling portrait: EAR ≈ 0.22, MAR ≈ 0.27 (teeth showing), jawOpen 0.11, eyeBlink ≈ 0.2, gaze ≈ 0.51/0.48, yaw/pitch/roll ≈ 1.5/−5.3/−0.7°.
- Same run: 200 face frames, a duplicate timestamp, a 50 ms backward step, 20 blank frames (→ `no_face`), then a seek back to 0 (→ landmarker recreated). No errors.
- Raw roll sign verified on in-plane rotated images (±20° → ±20° raw). Yaw and pitch signs follow from the same confirmed axis frame (x right, y up, z toward camera, translation z < 0). **Confirm live with manual test 2.**

### How to run
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_models.py
uvicorn app.main:app --reload
```

### How to test
`pytest -q` (39 tests):
- `test_features.py`: EAR/MAR known values, closed eye/mouth, scale invariance, degenerate points; rotation → Euler for identity, pure yaw/pitch/roll with signs, combined round trips with translation ignored; gaze centred and shifted; bbox clipping; feature keys.
- `test_preprocess.py`: shape, dtype, RGB order, edge padding, face partly outside the frame.
- `test_face.py`: `MonotonicClock` cases; a missing model gives `available=false` with no crash; with the real model, blank frames give `no_face` across duplicate, small and large backward timestamps (skipped if the model isn't downloaded).
- `test_ws.py`: updated for the `face` block, `no_face` and `chart_window_s`.

### Deviations from the spec / decisions
- **Letterboxed capture:** Phase 1 stretched every source into 640×480. For 16:9 files that distorts EAR/MAR and misaligns the overlay with the letterboxed `<video>`. The browser now letterboxes into the capture canvas (black bars). Webcam at 4:3 is unchanged.
- **Status when the model is unavailable** is `ok` with `face.available=false` rather than a new status value, because "unavailable" is per component (CLAUDE.md).
- **Landmark indices** are named constants in `features.py`, not config: they are fixed mesh topology, not tunable thresholds.
- **New `ui` config section**; `/api/config` now also returns `chart_window_s`, flat alongside the stream fields.
- **`crop_face` takes `size`/`margin` as arguments** (from `face.crop_size` / `face.crop_margin`), so `training/` can use it without importing `app.config`.

### Known issues / notes
- Blendshape left/right naming is MediaPipe's own. Check with a wink whether `eyeBlinkLeft` matches `ear_left` (the driver's left eye). If it's swapped, note it; Phase 3 uses the mean anyway.
- Creating the landmarker costs a few hundred ms at stream start and on every large backward seek.
- MediaPipe prints a few C++ `W0000`/`INFO` lines to stderr at startup; these are harmless.
- Chart.js comes from a CDN. Offline, the chart is disabled with a console warning and everything else works.

---

## Phase 3 — Calibration + rule-based drowsiness and distraction

### What was built
- **Temporal utilities** (`app/temporal/`). All of them run on client frame timestamps and tolerate duplicate or backward timestamps.
  - `window.py` `TimeWindow`: time-weighted sliding window. Each sample counts until the next one, capped at `max_dt_s`, so gaps don't count. `time_mean(min_span_s)` gives PERCLOS-style fractions.
  - `ema.py` `EMA`: time-based, `alpha = 1 − exp(−dt/τ)`, so it behaves the same at any FPS.
  - `hysteresis.py` `HysteresisStateMachine`: multi-level, with an enter/exit threshold per level. The level the score asks for must persist `min_enter_s` (going up) or `min_exit_s` (going down) before the state switches. Phase 5 can reuse it for risk levels.
  - `episodes.py` `EpisodeTracker`: duration of a continuous condition (eyes closed, mouth open, off road). Dropouts shorter than `max_gap_s` don't split an episode.
- **Estimator interface** (`app/estimators/base.py`):
  - `TaskOutput{score, label, confidence, probabilities, reasons, details}`.
  - `Estimator.update(features | None, timestamp_ms)`, plus `set_baseline` and `reset`.
  - `app/estimators/__init__.py` `build_estimators()` picks implementations by `mode`. Only `rules` exists, so `ml`/`hybrid` log a warning and use rules.
- **Calibration** (`app/pipeline/calibration.py`):
  - `Baseline{ear_open, mar_closed, yaw, pitch, gaze_h, gaze_v, calibrated}` holds medians over `calibration.duration_s` (10 s) of **frame time**.
  - Calibration fails, keeping the previous baseline, if the face was seen in < 60% of frames, the median EAR < 0.15 (eyes closed) or the median MAR > 0.30 (mouth open).
  - When uncalibrated, the baseline is `calibration.defaults`.
- **Drowsiness rules** (`drowsiness_rules.py`):
  - Eye closed when `EAR < 0.7 × baseline EAR` or mean eyeBlink > 0.55.
  - Closure episodes < 1.0 s are blinks (count and mean duration over 60 s, shown but not scored). Longer ones are long closures, with a component ramping 1.0 → 1.5 s.
  - PERCLOS is taken over 60 s, with a denominator of at least 30 s.
  - Yawn: MAR > `max(0.5, baseline MAR + 0.35)` for > 1.5 s, counted over 5 min.
  - `score = clamp(max(closure, PERCLOS) + 0.4 × min(1, yawns/3))`, then EMA (τ 0.2 s), then hysteresis → `alert | slightly_drowsy | drowsy` (enter 0.3/0.6, exit 0.2/0.45, 0.3 s up, 2.0 s down).
- **Distraction rules** (`distraction_rules.py`):
  - Off road, relative to the baseline, when `|Δyaw| > 30°`, `Δpitch < −20°`, `Δpitch > 25°`, or gaze is more than 0.2 (h) / 0.25 (v) from the neutral gaze. Gaze is ignored while the eyes are closed.
  - Continuous component: 0 below 2 s (glances), ramping to 1 at 3 s.
  - Glance-fraction component: share of the last 10 s spent in short glances, from 0 at 30% to 1 at 70%.
  - `score = max`, then EMA, then hysteresis → `attentive | looking_away | distracted` (0.3 s up, 1.5 s down).
- **Pipeline** (`FrameProcessor`):
  - Owns the `Calibrator` and the estimators.
  - Exposes `start_calibration()`, `set_baseline(dict)` and `clear_calibration()`; the Phase 6 offline runner can call them directly.
  - A backward timestamp jump > `face.timestamp_reset_ms` (seek/replay) resets the estimators and restarts a running calibration. The baseline is kept.
  - No-face frames give the label `unknown` with reason "no face".
- **Contract** (`FrameResult`): new `calibration{state, progress, remaining_s, message, baseline}` plus `drowsiness` and `distraction` (`TaskOutput`, or `null` if disabled or the face model is unavailable).
- **WebSocket**: text messages are JSON control commands: `{"type":"calibrate"}`, `{"type":"clear_calibration"}`, `{"type":"set_baseline","baseline":{…}}`.
  - Valid commands get **no reply**, so the one-frame-in-flight loop is unaffected.
  - Invalid text still gets `bad_message`.
- **Dashboard**:
  - Calibration card with **Calibrate / Recalibrate**, **Use defaults**, a status badge (uncalibrated / calibrating N s / calibrated), the message (including failure reasons) and the baseline values.
  - During calibration, a banner over the video reads "Look at the road, eyes open, mouth closed" with a countdown and progress bar.
  - Drowsiness and Distraction cards each show a label badge, colour-coded score bar, score, reasons list and a details line (PERCLOS, closed s, blinks, mean blink ms, yawns / off-road s, glance %, Δyaw, Δpitch).
  - The webcam baseline is saved in `localStorage` and re-sent when the socket opens, so Stop/Start and page refresh keep the calibration. Video files always start uncalibrated.

### Measured (automated, no browser)
Scripted WS client against live uvicorn with a real portrait:
- `calibrate` → calibrated after 10 s of frame time. Baseline EAR 0.203, MAR 0.285 (smiling, teeth showing; close to the 0.30 sanity limit), yaw 1.9°, pitch −4.9°.
- Output while still: `alert` / `attentive`.
- 30 blank frames → `no_face` / `unknown`.
- Backend p50 is unchanged (the estimators cost well under 1 ms).

### How to run
Same as Phase 2: `uvicorn app.main:app --reload` → http://localhost:8000, Start webcam, press **Calibrate**.

### How to test
`pytest -q` (79 tests). New:
- `test_temporal.py`: window fractions, gaps, pruning; EMA time constant and FPS independence; episodes and dropouts; hysteresis delays, band and anti-flicker.
- `test_calibration.py`: medians with blinks, completion by frame time, failure on no face, closed eyes or open mouth (keeping the previous baseline), restart on seek, baseline validation.
- `test_drowsiness_rules.py`:
  - 60 s of 150–300 ms blinks → always alert.
  - A 2 s closure → drowsy with "eyes closed …", back to alert only after `min_exit_s`.
  - A 0.9 s closure → alert.
  - 3 yawns → count 3 and slightly_drowsy; 1 s mouth openings → no yawn.
  - High PERCLOS → drowsy.
  - The calibrated EAR threshold, look-down suppression, no face, duplicate timestamps, reset.
- `test_distraction_rules.py`:
  - A 1 s glance → attentive; a glance at stream start → attentive.
  - A 4 s look-down → distracted, then back to attentive.
  - Frequent 1.5 s glances → raised fraction.
  - Camera offset absorbed by calibration; cone boundaries and gaze; no face.
- `test_ws.py`: new blocks, control commands (no reply; invalid → `bad_message`), calibration progress and clean failure on blank frames.

### Deviations from the spec / decisions
- **Looking down vs. eye closure:** looking down lowers EAR with the eyes open. When `Δpitch < −15°` (`ear_ignore_pitch_down_deg`), eye closure uses only the eyeBlink blendshape. Weakness: a drowsy head-nod then relies on the blendshape alone.
- **Gaze range is relative to a calibrated neutral gaze** (`gaze_h`/`gaze_v` are added to the baseline), so camera offset is absorbed as it is for head pose.
- **Yawn threshold** = `max(absolute 0.5, closed-mouth baseline + 0.35)`, so the calibrated MAR is used.
- **Hysteresis has separate escalate/de-escalate durations** (`min_enter_s`, `min_exit_s`) instead of the single "minimum duration" in the spec. This is what gives "returns after the exit delay".
- **Distraction fraction counts only glance time** (off-road time while the episode is still < 2 s). If all off-road time were counted, one 4 s look-down would keep the 10 s fraction at 40% and hold `looking_away` for ~6 s after the driver looked back. Long episodes are already scored by the continuous component.
- **Rule estimators return empty `probabilities`**: there are no calibrated probabilities until the ML model (Phase 8). `confidence` is the share of the estimator's window covered by face frames.
- **`TaskOutput.details`** is an additive field holding numeric stats for the cards and later session logging.
- **Calibration persistence** in browser `localStorage` (webcam only).

### Known issues / notes
- Distraction is `unknown` when the face is lost. A head turned so far that MediaPipe loses the face is therefore not counted as off road; that belongs to Phase 5 fusion.
- Thresholds are first guesses. Tune them in `config/default.yaml` after the manual tests (e.g. `closed_ratio`, `blink_threshold`, `yaw_limit_deg`).
- Smiling with teeth during calibration gives a high closed-mouth MAR (≈0.28), which raises the yawn threshold. Calibrate with a neutral, closed mouth.
- Yaw/pitch signs come from Phase 2. If manual test 4 shows "head turned right" while you turn left, flip `YAW_SIGN` in `features.py`.

---

## Phase 4 — Activity detection (phone, drinking) and emotion

### What was built
- **Dependencies:**
  - `ultralytics` (installed: 8.4.174). It pulls in torch 2.14.1 CPU and torchvision, about 300 MB.
  - `onnxruntime` (installed: 1.30.0).
  - Both work with the existing numpy 2.5.3, opencv 5.0 and mediapipe, and `pip check` is clean.
  - The pip install upgraded protobuf to 7.x; MediaPipe and all tests still pass.
- **⚠ License:** ultralytics (code and YOLO weights) is **AGPL-3.0**. That is fine for an academic project. Distributing the app as a product would require AGPL compliance or an Ultralytics enterprise licence.
- **Models** (downloaded by `scripts/download_models.py` into `models_store/`; URLs in config):
  - `yolo11n.pt`: YOLO11 nano, COCO, 5.6 MB.
  - `enet_b0_8_best_vgaf.onnx`: HSEmotion EfficientNet-B0 trained on AffectNet, 16 MB.
- **Objects** (`app/estimators/objects.py`):
  - `ObjectDetector` keeps only the configured classes (`cell phone`, `cup`, `bottle`) and runs every `objects.every_n_frames` (5) frames on the full frame.
  - The model is loaded once per process and shared by streams (a lock guards inference).
  - Missing or broken weights give `objects.available=false`. The existence check also stops ultralytics from auto-downloading into the working directory. `YOLO_OFFLINE` is set, so there are no network calls at runtime.
- **Activity rules** (`ActivityRules`, pure logic): fed only on frames where YOLO ran. Episodes use `EpisodeTracker` with `max_gap_s` 1.0 s, which is longer than the YOLO interval.
  - `phone_use`: a phone box overlaps the *phone region* for ≥ `phone_min_s` (1.0 s). The region is the face box grown by 1 face width left and right, 0.5 face heights up and 2.5 face heights down, which covers both phone at the ear and texting at chest height. **With no face, a phone anywhere in the frame counts**, because looking down at a phone often loses the face.
  - `drinking`: a cup or bottle box touches a square around the mouth centre (half-side 0.6 × face width) for ≥ `drink_min_s` (0.5 s). If the cup hides the face, the last face box and mouth are reused for `face_hold_s` (1 s).
- **Distraction:**
  - New components: phone use ramps from 0 to `phone_weight` (1.0) over `phone_full_s` (2 s) of `phone_s`; drinking adds a constant `drinking_weight` (0.2, below `looking_away` on its own).
  - `raw = max(continuous, glance fraction, activity)`.
  - Phone use is scored **even with no face**. With no face and no activity the output is still `unknown`.
  - Reasons put "phone use N s" first when it drives the score.
- **Emotion** (`app/estimators/emotion.py`):
  - Input: the `crop_face` output (224 RGB, the same function training will use), every `emotion.every_n_frames` (3) frames with a face.
  - Preprocessing: ImageNet normalization, onnxruntime on CPU.
  - Output mapping: softmax, then **contempt is dropped and the other 7 renormalized** (neutral, happy, sad, surprise, fear, disgust, anger), then a per-class time-based EMA (τ 0.7 s, so about 95 % of a change after about 2 s), then renormalized.
  - Fields: `label` = argmax, `confidence` = its probability, `probabilities` = all 7. `score` = summed probability of sad, fear, disgust and anger, which Phase 5 will use. **Emotion does not affect risk.**
  - With no face it reports `unknown`; the EMA is kept through short dropouts and reset on seek.
  - If the model is missing, `emotion` is `null`.
- **Interface:**
  - `Estimator.update(features, timestamp_ms, context=None)`. The new optional `FrameContext{face_crop, activities}` carries the non-feature inputs, so emotion uses the same interface as the rule estimators.
  - `Activities` lives in `estimators/base.py`.
- **Pipeline** (`FrameProcessor`):
  - Order: landmarks → objects → activities → estimators.
  - A per-stream frame counter decides which frames run the heavy models; other frames reuse the last detections and emotion. The counter, detections and activities reset on seek, so the models run on the first frame after one.
  - Objects still run when face analysis is unavailable.
- **Contract** (`FrameResult`):
  - New `objects{available, detections[{label, kind, conf, bbox}], activities{phone_use, drinking, phone_s, drinking_s, reasons}}` and `emotion` (`TaskOutput | null`).
  - `perf.components{landmarks_ms, objects_ms, emotion_ms, pipeline_ms}`: the *last run* time of each heavy component, and the pipeline time for this frame.
  - `/api/config` also returns `objects_every_n_frames` and `emotion_every_n_frames`.
- **Dashboard:**
  - Activity card with Phone and Drinking badges: red or amber when active, with the duration, plus the list of seen objects.
  - Emotion card: label badge plus top-3 probability bars.
  - Perf panel rows: Landmarks, Objects (YOLO, every N frames), Emotion (every N frames), Pipeline total.
  - The overlay draws YOLO boxes with label and confidence (phone red, cup/bottle amber) under the "Show landmarks & objects" toggle.

### Measured (automated, no browser)
- **Model check on CPU** (scratch script before integration):
  - Emotion ONNX: about 10 ms per crop.
  - YOLO11n at imgsz 640: about 49 ms per frame.
  - On the MediaPipe sample portrait (smiling), the emotion model gives happy = 1.00, which confirms the class order `anger, contempt, disgust, fear, happy, neutral, sad, surprise`.
- **Scripted WebSocket client against live uvicorn**, 300 frames of that portrait (640×480):

  | Measure | Value |
  |---|---|
  | Backend latency | p50 **20.4 ms**, p95 **92.5 ms** (p95 is the frames where YOLO runs) |
  | Client round trip | p50 24.8 ms |
  | Stream FPS | ≈ **25** (Phase 3: 27) |
  | Landmarks (median) | 16 ms |
  | YOLO (median) | 65 ms, every 5th frame |
  | Emotion (median) | 13 ms, crop + inference, every 3rd frame |

  Output: emotion `happy`, no objects detected, distraction `attentive`.

### How to run
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt      # adds ultralytics (+ torch CPU) and onnxruntime
python scripts/download_models.py    # adds yolo11n.pt and the emotion .onnx
uvicorn app.main:app --reload
```

### How to test
`pytest -q` (111 tests). New:
- `test_activity_rules.py`:
  - Phone at the ear or while texting → `phone_use` only after 1 s.
  - A phone far from the face is ignored, but counts anywhere when there is no face.
  - Short YOLO misses don't split an episode, and phone use ends after the gap.
  - A cup at the mouth → `drinking`; a cup on the table → no.
  - The face is held while the cup hides it, then expires.
  - A backward timestamp resets everything.
- `test_emotion.py`:
  - Contempt is dropped and the rest renormalized; preprocessing shape and normalization.
  - EMA: neutral→happy is not instant but completes within 2 s.
  - Skipped frames reuse the output; no face → unknown; score = negative probability mass; waiting before the first estimate.
  - The real model runs on CPU (skipped if not downloaded), and a missing model is unavailable.
- `test_distraction_rules.py`:
  - Phone use while looking ahead → `distracted`, with "phone use" first.
  - Phone use with no face still scores; no face and no activity → unknown.
  - Drinking alone stays attentive.
- `test_frame_processor.py` (fake face, YOLO and emotion models):
  - YOLO runs every 5th frame and emotion every 3rd.
  - Phone activity reaches distraction; a seek restarts the counter and activities.
  - With no face, objects still run.
  - Missing models → `objects.available=false`, `emotion=null`.
  - Real YOLO on a blank frame.
- `test_ws.py`: the `perf.components`, `objects` and `emotion` fields, plus the new `/api/config` fields.

### Deviations from the spec / decisions
- **ONNX instead of the `hsemotion` PyTorch package.** It is the same HSEmotion model, run with onnxruntime. The `hsemotion` package loads pickled models that need an old `timm` and break with current `timm`/`torch`. The spec allows "an equivalent ONNX model".
- **8 → 7 classes:** the AffectNet model also has *contempt*. It is dropped and the other probabilities renormalized. The model's class order and the reported classes are both in config (`model_classes`, `classes`).
- **No hand detector.** "Near the face/hand region" is approximated by a face-relative region (beside, above and well below the face). A MediaPipe Hand Landmarker would add CPU cost per frame. Weakness: a phone held in the lap, below the region, is missed while the face is visible.
- **The phone with no face counts anywhere in the frame**, because the driver looking down at a phone often loses face detection. Distraction then scores the phone alone.
- **Drinking feeds distraction only weakly** (0.2, so it stays attentive). The spec only asks for phone use to "strongly" raise distraction. Drinking shows as a reason and a badge.
- **Activities persist up to `max_gap_s` (1 s)** after the object disappears, a debounce that comes from the episode gap tolerance.
- **Object boxes on the overlay** (not in the spec), to make the manual tests easier to read.
- **Fix after manual test 2: a glass is read as "cell phone" while drinking.**
  - Held at the mouth, a hand-covered glass is misclassified by YOLO11n as `cell phone`, although the same glass in front of the camera is a `cup`.
  - Two changes:
    1. `wine glass` is added to `drink_classes`.
    2. `phone_at_mouth_is_drink`: a phone box that overlaps a small square at the mouth centre (half-side `mouth_point_margin` 0.1 × face width) is relabelled `drink (cell phone at mouth)` and counts toward drinking, not phone use. This also applies while the face is hidden, using the held face.
  - Why the rule is safe: phones at the ear sit beside the face and texting is below it, so neither covers the mouth.
  - Weakness: a phone held in front of the mouth (e.g. speakerphone) reads as drinking. Set `phone_at_mouth_is_drink: false` to disable the rule.
- **`emotion.score`** = negative-emotion probability mass. It is defined now for the Phase 5 modifier, but nothing uses it yet.

### Known issues / notes
- **Latency jitter:** YOLO runs on every 5th frame (about 50–65 ms), and with one frame in flight that frame takes longer, so the video stutters slightly. If FPS drops too much:
  - raise `objects.every_n_frames` (keep `max_gap_s` above the resulting interval), or
  - lower `objects.imgsz` to 480. That is faster but misses small or far phones.
- COCO's "cell phone" class is weak on phones seen edge-on or mostly covered by the hand, for example held flat to the ear. Lower `objects.conf` (0.35) if manual test 1 misses it.
- AffectNet models read "neutral" faces of some people as sad or angry, so expect some bias. Emotion is context only.
- ultralytics creates `%APPDATA%\Ultralytics\settings.json` on first import. It is harmless.
- The first stream after server start takes about 2–3 s longer, while torch, YOLO and onnxruntime load. Later streams reuse the loaded models.
