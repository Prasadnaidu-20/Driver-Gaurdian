# DriveGuardian — Project Context for Claude Code

## What this project is
A real-time driver monitoring web application. The browser captures the webcam (or plays a video file), streams frames to a Python backend over a WebSocket, and the backend runs face analysis, drowsiness / distraction / emotion estimation, temporal smoothing, risk fusion and alert generation. It returns a JSON state per frame that the dashboard renders.

This is an academic project with limited time. A working, measurable, demonstrable system matters more than visual polish or architectural cleverness.

## How we work (IMPORTANT — follow strictly)
1. The project is built **phase by phase** from `PHASES.md`. Implement **only the phase the user names**. Do not start the next phase and do not pre-build future features; leave clean extension points instead.
2. Before writing code for a phase: read that phase in `PHASES.md` and read `docs/progress.md`. Then present a short plan (files to create/modify, new dependencies, open questions) and **wait for approval**.
3. At the end of a phase:
   - run `pytest -q` and fix failures,
   - start the app if the phase touches it and confirm it boots,
   - update `docs/progress.md` (what was built, how to run it, how to test it, known issues, any deviation from the spec and why),
   - give the user a numbered manual test checklist for this phase.
   Do **not** run `git commit`; the user commits after testing.
4. If something in the spec is wrong, infeasible or a library doesn't install, stop and say so, and propose an alternative. Never silently deviate.
5. Keep it simple. No extra frameworks, no database, no auth, no Docker unless asked.

## Tech stack
- Python 3.11 (MediaPipe supports 3.9–3.12; do not use 3.13).
- Backend: FastAPI, uvicorn, WebSockets, OpenCV (`opencv-python`), MediaPipe Tasks (Face Landmarker), NumPy, pydantic, PyYAML, psutil.
- Later phases: ultralytics (YOLO, Phase 4), an off-the-shelf emotion model (Phase 4), PyTorch + timm (Phase 7+), scikit-learn for splits/metrics.
- Frontend: plain HTML + CSS + vanilla JavaScript served by FastAPI. No React, no build step. Chart.js from a CDN is allowed.
- Tests: pytest.
- `requirements.txt` for the app; `training/requirements.txt` for training-only extras.

## Repository layout
```
driveguardian/
  CLAUDE.md, PHASES.md, README.md, requirements.txt
  config/default.yaml          # ALL thresholds, weights, windows, model paths
  app/
    main.py                    # FastAPI app, static files, routes
    schemas.py                 # pydantic FrameResult and sub-models (backend→frontend contract)
    config.py                  # loads config/default.yaml into typed settings
    api/ws.py                  # /ws/stream WebSocket handler
    pipeline/
      frame_processor.py       # orchestrates one frame end to end (used by web AND offline eval)
      face.py                  # MediaPipe Face Landmarker wrapper
      features.py              # EAR, MAR, head pose, gaze, blendshape features
      preprocess.py            # face crop used identically in inference and training
      calibration.py           # per-driver baseline
    estimators/
      base.py                  # common estimator interface + TaskOutput
      drowsiness_rules.py, distraction_rules.py, objects.py, emotion.py
      ml_multitask.py          # Phase 8
    temporal/                  # windows, EMA, hysteresis, (GRU in Phase 8)
    risk/fusion.py, risk/alerts.py
    utils/perf.py, utils/session_log.py
  web/index.html, web/app.js, web/styles.css
  scripts/download_models.py   # fetches .task / .pt / .onnx weights into models_store/
  models_store/                # weights (gitignored)
  evaluation/                  # offline video runner, metrics, benchmark, annotations/
  training/                    # dataset adapters, manifests, train.py, eval.py, export.py
  experiments/                 # run outputs (gitignored except summaries)
  sessions/                    # live session logs (gitignored)
  data/                        # raw + processed datasets (gitignored)
  tests/
  docs/progress.md
```

## Architecture rules
- **No magic numbers.** Every threshold, weight, window length, frame-skip value and model path lives in `config/default.yaml`, with a short comment.
- **Swappable estimators.** Every drowsiness / distraction / emotion estimator implements the interface in `app/estimators/base.py` and returns a `TaskOutput(score: float 0–1, label: str, confidence: float, probabilities: dict, reasons: list[str])`. Which estimator runs is chosen in config (`mode: rules | ml | hybrid`).
- **One pipeline.** `FrameProcessor` is the single code path used by both the live WebSocket and the offline evaluation runner. Never duplicate pipeline logic.
- **Stable contract.** The backend→frontend message is `FrameResult` in `app/schemas.py`. Change it only deliberately, and update the frontend in the same phase.
- **Client timestamps.** Temporal logic uses the frame timestamp (ms) sent by the client or read from the video file, not server wall-clock, so webcam and video files behave identically.
- **Heavy models run every N frames** (configurable); intermediate frames reuse the last result.
- **Emotion is context only.** Emotion can modify risk slightly but can never on its own raise risk above Low.
- **Same preprocessing in training and inference.** Training code imports the face-crop function from `app/pipeline/preprocess.py`. Otherwise `training/` does not import from `app/`.
- **Fail gracefully.** No face → state `no_face`; a model that fails to load → that component reports `unavailable`; the stream must never crash.

## Commands
- Install: `python -m venv .venv`, activate it, then `pip install -r requirements.txt` and `python scripts/download_models.py`
- Run app: `uvicorn app.main:app --reload` → open http://localhost:8000
- Tests: `pytest -q`
- Offline evaluation (Phase 6+): `python -m evaluation.run_video --video <path> --annotations <csv>`

## Coding conventions
Type hints everywhere, dataclasses or pydantic models, small focused modules, docstrings on public functions, `logging` instead of `print`. Unit-test all pure logic (feature math, temporal logic, fusion, metrics). Document sign conventions (e.g. which yaw direction is positive) in code comments.
