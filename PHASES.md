# DriveGuardian — Phased Build Plan

## Scope decisions (changes from the original proposal, and why)

The proposal is kept, but re-ordered and trimmed so you get a working app early and the research parts are added on top of it.

1. **Working system first, trained model second.** Phases 1–6 build a complete real-time app using MediaPipe geometry, per-driver calibration, temporal logic and pretrained models. Phases 7–8 add the trained multi-task network. This guarantees a demo even if dataset access or training slips, and the rule-based system becomes the "threshold-based baseline" the proposal claims to beat, which you need for the comparison anyway.
2. **Fewer datasets.** Core: **DMD** (distraction + drowsiness, face camera) and **NTHU-DDD** (drowsiness). Emotion: **AffectNet** if access is approved in time, otherwise **FER2013** or **RAF-DB**. **Drive&Act, AIDE, 3MDAD** move to optional/future work (Drive&Act is multi-view IR and heavy to process).
3. **Phone use and drinking via pretrained object detection** (COCO YOLO classes *cell phone, cup, bottle*) in the early system, instead of training on Drive&Act.
4. **Emotion via a pretrained AffectNet-trained model** in the early system; the trained multi-task head replaces/compares against it in Phase 8.
5. **Default backbone EfficientNet-B0** for real-time on CPU. EfficientNetV2-S and ConvNeXt-Tiny are kept as comparison experiments if GPU time allows.
6. **Temporal modelling in two levels:** sliding windows + EMA + hysteresis (always), GRU as an optional upgrade in Phase 8.
7. **Added: per-driver calibration** (≈10 s baseline). This directly addresses the proposal's own criticism that fixed EAR/MAR thresholds ignore individual differences.
8. **Added: a small self-recorded test set** (your own annotated clips) for system-level metrics (false alarms, alert recall, detection latency). Public datasets rarely support these metrics directly.
9. Single-user local app, no database, no login. Sessions are logged to CSV/JSON.



## Priority if time runs out
- **Must:** Phases 0–6 (complete working app + evaluation) and Phase 9 (docs).
- **Should:** Phase 7 (trained multi-task model, shared vs independent comparison).
- **Could:** Phase 8 GRU and the full ablation matrix.

---

## How to run each phase with Claude Code

1. Start a fresh context: `/clear` (or a new session) at the beginning of every phase.
2. Switch to **Plan Mode** (Shift+Tab) and paste:
   > Read CLAUDE.md, the "Phase N" section of PHASES.md, and docs/progress.md. We are implementing **Phase N only**. Give me your plan: files to create or modify, new dependencies, and anything in the spec you think is wrong or risky. Wait for my approval.
3. Review the plan, adjust it, then approve and let it implement.
4. When it finishes, follow the **Manual test** list for that phase yourself. For each problem, report it like this:
   > Phase N test: I did [action]. Expected [X]. Got [Y]. [Paste error/log]. Fix this without starting the next phase.
5. When the phase passes: `git add -A && git commit -m "Phase N: <title>" && git tag phase-N`. If a later phase breaks things, you can return with `git checkout phase-N`.
6. If a phase feels too big for one go, ask Claude Code to split it into parts (e.g. "Do only the calibration part of Phase 3 now").

---

## Phase 0 — Project setup

**Goal:** an empty but runnable skeleton.

**Build**
- Folder structure from CLAUDE.md, `.gitignore` (`.venv/`, `models_store/`, `data/`, `experiments/*/`, `sessions/`, `__pycache__/`).
- `requirements.txt` with only what Phase 0–1 need (fastapi, uvicorn[standard], opencv-python, numpy, pydantic, pyyaml, psutil, pytest).
- `config/default.yaml` with top-level sections (`server`, `stream`, `face`, `calibration`, `drowsiness`, `distraction`, `objects`, `emotion`, `risk`, `alerts`, `logging`, `mode`) — values can be placeholders.
- `app/config.py` that loads the YAML into typed settings; `app/main.py` with `GET /health` returning `{"status": "ok"}` and serving `web/` as static files at `/`.
- Placeholder `web/index.html` titled "DriveGuardian".
- `docs/progress.md` and a README stub with setup steps.
- One passing test (config loads).

**Manual test**
1. Create venv, install requirements, run `uvicorn app.main:app --reload`.
2. `http://localhost:8000/health` → `{"status":"ok"}`; `http://localhost:8000` shows the page.
3. `pytest -q` passes.

---

## Phase 1 — Live video loop (webcam/video → backend → dashboard)

**Goal:** frames flow from browser to backend and back with measured FPS and latency. No AI yet.

**Build**
- Frontend: Start/Stop webcam (`getUserMedia`), and a "Load video file" option that plays a local file in a `<video>` element. Frames are drawn to a canvas at the configured size (default 640×480), JPEG-encoded (quality from config, default 0.7) and sent over WebSocket `/ws/stream` as a binary message: 8-byte float64 timestamp (ms) + 4-byte uint32 frame id + JPEG bytes. For video files, the timestamp is `video.currentTime * 1000`.
- **Flow control:** only one frame in flight. The client sends the next frame only after receiving the result for the previous one, so latency never builds up.
- Backend: decode with OpenCV, pass through a stub `FrameProcessor`, return `FrameResult` JSON `{frame_id, timestamp_ms, status, perf: {backend_ms, fps}}`.
- `app/utils/perf.py`: rolling FPS and latency percentiles (p50/p95) over the last N frames.
- Dashboard layout skeleton: video on the left with a transparent overlay canvas on top; right panel with "Performance" (processing FPS, round-trip latency, backend latency). Leave empty placeholder cards for Driver State, Drowsiness, Distraction, Emotion, Risk, Alerts.

**Manual test**
1. Start webcam → video visible, FPS and latency numbers updating (expect roughly 15–30 FPS locally).
2. Load an .mp4 → plays and FPS/latency update.
3. Stop and restart several times; refresh the page; close the tab. No errors in the server console.

---

## Phase 2 — Face landmarks and geometric features

**Goal:** reliable per-frame facial features shown live.

**Build**
- `scripts/download_models.py` downloads MediaPipe `face_landmarker.task` into `models_store/`.
- `app/pipeline/face.py`: Face Landmarker in VIDEO running mode, `num_faces=1`, with `output_face_blendshapes=True` and `output_facial_transformation_matrixes=True`. MediaPipe requires strictly increasing timestamps in VIDEO mode, so recreate the landmarker on a new stream/session and guard against non-increasing timestamps.
- `app/pipeline/features.py`:
  - EAR left/right/mean (standard 6-point formula on MediaPipe eye landmarks), MAR.
  - Head pose yaw/pitch/roll in degrees from the facial transformation matrix (document sign conventions).
  - Gaze: horizontal and vertical iris position ratio within each eye (iris landmarks 468–477), averaged.
  - Blendshapes: `eyeBlinkLeft`, `eyeBlinkRight`, `jawOpen`.
  - Face bounding box.
- `app/pipeline/preprocess.py`: `crop_face(frame, landmarks) -> 224×224 RGB` square crop with margin (used later for emotion and training).
- Extend `FrameResult` with a `face` block (`detected`, bbox, features dict) and status `no_face` when no face.
- Frontend: overlay face box + eye/mouth/iris landmarks (toggle "show landmarks"); Driver State card shows the live numbers; a rolling 10-second Chart.js line chart of EAR and MAR.
- Unit tests: EAR/MAR on synthetic points; rotation-matrix → Euler conversion.

**Manual test**
1. Blink → EAR dips sharply and `eyeBlink` rises. Open mouth wide → MAR and `jawOpen` rise.
2. Turn head left/right, up/down → yaw/pitch change with consistent signs.
3. Look left/right with eyes only → gaze ratio changes.
4. Test with glasses and in dimmer light; note how values change (useful for your report).
5. Leave the frame → `no_face`, no crash. FPS remains acceptable.

---

## Phase 3 — Calibration + rule-based drowsiness and distraction with temporal logic

**Goal:** the first real driver-state outputs, robust to normal blinks and quick glances.

**Build**
- **Calibration** (`calibration.py`): on "Calibrate", show "Look at the road, eyes open, mouth closed" for N seconds (config, default 10). Store median open-eye EAR, closed-mouth MAR, and neutral yaw/pitch (this also absorbs camera-placement offset). "Recalibrate" button. If skipped, use config defaults and show "uncalibrated".
- **Temporal utilities** (`app/temporal/`): time-based sliding window, EMA, and a hysteresis state machine (enter threshold, exit threshold, minimum duration before switching).
- **Base interface** (`estimators/base.py`): `update(features, timestamp_ms) -> TaskOutput`.
- **Drowsiness (rules)**:
  - eye closed when EAR < `closed_ratio` × baseline EAR (default 0.7) or mean eyeBlink > threshold;
  - blink detection and blink duration; PERCLOS over a 60 s window;
  - long closure / microsleep when eyes stay closed > 1.0 s;
  - yawn event when MAR stays above threshold > 1.5 s; yawn count over last 5 min.
  - Combine into a score 0–1 and label `alert | slightly_drowsy | drowsy`, with human-readable reasons (e.g. "PERCLOS 22%", "eyes closed 1.4 s").
- **Distraction (rules)**:
  - "eyes off road" when head pose relative to neutral leaves a configurable cone (default |yaw| > 30°, pitch < −20° i.e. looking down) or gaze ratio leaves its range;
  - continuous off-road duration and off-road fraction over the last 10 s;
  - glances shorter than 2 s ignored.
  - Score 0–1, label `attentive | looking_away | distracted`, reasons.
- Frontend: calibration flow; Drowsiness and Distraction cards with score bars, labels and reasons.
- Unit tests on synthetic feature sequences: normal blinks (150–300 ms) do not trigger; a 2 s closure triggers; a 1 s glance is ignored; a 4 s look-away triggers; hysteresis prevents flicker.

**Manual test (follow this script)**
1. Calibrate. Sit normally for 1 minute, blinking naturally → stays alert/attentive.
2. Close eyes 2–3 s → drowsy rises with reason; open eyes → returns after the exit delay.
3. Yawn 2–3 times → yawn count and score increase.
4. Glance at a mirror for 1 s → nothing. Look down at lap for 4 s → distracted.
5. Recalibrate with the camera moved slightly → neutral pose still correct.

---

## Phase 4 — Activity detection (phone, drinking) and emotion

**Goal:** the remaining signals from the proposal using pretrained models.

**Build**
- **Objects** (`estimators/objects.py`): ultralytics YOLO nano COCO weights (e.g. `yolo11n.pt` or `yolov8n.pt`), run every N frames (config, default 5) on the full frame. Classes from config: cell phone, cup, bottle. Activity rules: phone detected near the face/hand region for > X s → `phone_use`; cup/bottle near the mouth → `drinking`. Feed activities into the distraction estimator (phone use strongly raises distraction score). Note in progress.md that ultralytics is AGPL-3.0.
- **Emotion** (`estimators/emotion.py`): pretrained facial-expression model on the `crop_face` output, run every N frames. Suggested: the `hsemotion` package (EfficientNet models trained on AffectNet) or an equivalent ONNX model. Verify it installs and runs on CPU; if not, stop and propose an alternative. Map outputs to 7 classes (neutral, happy, sad, surprise, fear, disgust, anger), smooth probabilities with EMA over ~2 s. Output label + probabilities. Emotion does not affect risk yet.
- Dashboard: activity badges (phone / drinking) and an Emotion card with top-3 probabilities. Perf panel shows per-component time (landmarks, objects, emotion, total).

**Manual test**
1. Hold a phone to your ear, then in front of you as if texting → `phone_use` and higher distraction.
2. Drink from a bottle or cup → `drinking`.
3. Make clear expressions (smile, surprise, frown) → emotion follows within ~2 s.
4. Check FPS. If it drops too much on CPU, raise the frame-skip values in config and retest.

---

## Phase 5 — Risk fusion, alerts and session logging

**Goal:** a single composite risk score with graded, non-flickering alerts. After this phase the app is feature-complete.

**Build**
- `risk/fusion.py`: `risk = clamp(w_d·drowsiness + w_x·distraction + w_both·min(drowsiness, distraction))`, with emotion as a small capped modifier (default ±0.1) that can never on its own push risk above Low. All weights in config. Levels: Low < 0.4 ≤ Medium < 0.7 ≤ High (config), with hysteresis and minimum persistence (escalate only after the level holds for M seconds, de-escalate after K seconds).
- `risk/alerts.py`: alert events with start/end time, level and cause; cooldown between repeated alerts. Medium → visual banner; High → visual banner + audio beep generated with the Web Audio API (no audio files) + "Acknowledge" button.
- `utils/session_log.py`: each session writes `sessions/<timestamp>/frames.csv` (timestamp, key features, task scores, risk, level) and `alerts.json`.
- Dashboard final layout: video + overlay, a large colour-coded risk gauge, a 60-second risk timeline chart with alert markers, the task cards, an alerts log list and the perf panel. Show "uncalibrated" and "no face" states clearly.
- Unit tests: fusion math, emotion cap, level hysteresis, alert cooldown.

**Manual test**
1. Drive "normally" in front of the camera for 3 minutes → no alerts (note any false alarms).
2. Sustained eye closure → Medium then High with sound; acknowledge works.
3. Phone use for 10 s → distraction-driven alert.
4. Check `sessions/` contains a CSV and alerts JSON that match what you saw.

**Checkpoint:** tag `v1-rules`. You now have a complete demo.

---

## Phase 6 — Offline video mode and evaluation harness

**Goal:** turn the app into something you can measure.

**Build**
- `evaluation/run_video.py`: runs the same `FrameProcessor` (no browser) on a video file, using video timestamps; writes per-frame CSV + alerts JSON to `experiments/<run_name>/`; optional `--save-video` writes an annotated .mp4. `--config` lets you override config values for experiments.
- Annotation format `evaluation/annotations/<video_name>.csv`: `start_s,end_s,event` where event ∈ {drowsy, distracted, phone, yawn}.
- `evaluation/metrics.py`:
  - Event level: alert precision, alert recall, missed-event rate, false alarms per hour, detection latency (mean/median from event start to first alert). An alert matches an event if it starts within the event window plus a tolerance (config).
  - Temporal stability: risk-level changes per minute, number of flickers (level changes lasting < 1 s).
  - Frame level (where frame labels exist): accuracy, precision, recall, F1, confusion matrix.
- `evaluation/benchmark.py`: runs a long (looped) video for a configurable duration and records latency p50/p95/p99 per component, FPS, CPU %, RSS memory over time (to detect leaks/drift), and GPU memory if a GPU is used.
- Every run writes `metrics.json` and a short `summary.md`.
- `evaluation/README.md` explaining how to record and annotate the self-recorded test set.

**Your task in parallel: record the test set.** 8–12 clips, 1–3 minutes each, laptop/phone camera placed where a dashboard camera would be: normal driving posture, natural blinking, simulated drowsiness (slow blinks, long closures, head nodding), yawning, mirror glances, looking down, phone call, texting, drinking, with/without glasses, a darker clip. Annotate start/end times in a CSV while watching in any video player. Ideally include 1–2 other people.

**Manual test**
1. Run `run_video` on one clip → CSV, alerts and metrics appear; annotated video looks right.
2. Spot-check one metric by hand (e.g. count false alarms yourself).
3. Run a 15–30 minute benchmark → memory does not grow steadily.

---

## Phase 7 — Dataset pipeline and multi-task model training

**Prerequisite:** datasets downloaded into `data/raw/<dataset_name>/`. If you have no local GPU, the training scripts must also run in Google Colab or Kaggle notebooks (ask Claude Code to add a minimal notebook that clones the repo and calls the scripts).

**Build (in `training/`)**
- **Dataset adapters** (one file per dataset) → a unified manifest CSV with columns: `sample_id, dataset, subject_id, video_id, image_path, timestamp_s, drowsy, distraction, emotion` (−1 = no label for that task).
- **Unified labels**, documented in `training/LABELS.md` (source label → unified label, and which labels are dropped):
  - drowsiness: 0 alert, 1 drowsy;
  - distraction: 0 attentive, 1 looking_away, 2 phone, 3 drinking/eating, 4 other — reduce classes to what DMD actually supports;
  - emotion: 7 classes.
- **Frame extraction:** sample videos at 3–5 fps to avoid near-duplicate frames; crop faces with `app/pipeline/preprocess.py` (identical to inference); cache 224×224 crops in `data/processed/`. Log and skip frames with no detected face.
- **Splits:** subject-independent split with `GroupShuffleSplit` on `subject_id` (≈70/15/15, fixed seed), saved into the manifests, with an assertion that no subject appears in two splits. Use official splits for AffectNet/FER. Write a dataset statistics report (class counts per dataset and split).
- **Model** (`training/model.py`): `timm` pretrained backbone selected in config (default `efficientnet_b0`; alternatives `tf_efficientnetv2_s`, `convnext_tiny`) → shared pooled features → three heads (dropout + linear).
- **Masked multi-task loss:** each head computes cross-entropy only on samples that have a label for that task; per-task loss weights; class weights for imbalance; a sampler that mixes datasets so every batch contains all three tasks.
- **Training** (`train.py`): AdamW, cosine schedule, mixed precision, early stopping on mean validation macro-F1. Stage 1: frozen backbone for a few epochs; stage 2: full fine-tuning. Augmentation: brightness/contrast, blur, grayscale, small rotation and scale. Flag `--tasks drowsy,distraction,emotion` so the same script trains **single-task baselines** (`--tasks drowsy`) for the comparison.
- **Evaluation** (`eval.py`): per task accuracy, precision, recall, macro-F1, confusion-matrix PNG, ROC-AUC for drowsiness; parameter count; per-image latency on CPU (and GPU if present). Results to `experiments/<run>/`.
- **Export** (`export.py`): best checkpoint → TorchScript or ONNX in `models_store/`, plus a metadata JSON (label maps, input size, normalisation, backbone name).

**Manual test**
1. Smoke run first: 1 epoch on ~500 samples finishes end to end and produces metrics.
2. Check the split-leakage assertion and the dataset stats report.
3. Full training of: multi-task model, and three single-task models with the same backbone.
4. Compare results in the experiment summaries.

---

## Phase 8 — Model integration, temporal GRU, comparisons and ablations

**Build**
- `app/estimators/ml_multitask.py`: implements the estimator interface, loads the exported model from config, runs on the face crop every N frames, outputs per-task probabilities.
- **Modes** `rules | ml | hybrid`, switchable in config and in a dashboard dropdown. Hybrid combines ML probabilities with geometric scores (weighted sum from config, or a logistic regression fitted on validation data).
- **GRU (optional):** use `run_video` to dump per-frame feature vectors (ML probabilities + EAR, MAR, yaw, pitch, gaze, blink blendshapes) from DMD/NTHU videos; train a small GRU (1 layer, 32–64 hidden) on 2–4 s windows to predict drowsy/distracted; plug it in as a temporal option and compare against EMA + hysteresis.
- `evaluation/run_experiments.py` runs a configured experiment matrix and writes one combined markdown table:
  - **E1 Shared vs independent:** multi-task vs three single-task models — per-task macro-F1, total parameters, total latency, memory.
  - **E2 Backbones (if time):** EfficientNet-B0 vs EfficientNetV2-S vs ConvNeXt-Tiny.
  - **E3 System ablations on your test clips:** rules only; ML visual only; hybrid (ML + EAR/MAR/landmarks); with/without temporal smoothing; with/without GRU; risk fusion vs a simple "any task high → alert" rule — measured with Phase 6 metrics (false alarms/hour, alert recall, detection latency, stability, FPS).

**Manual test**
1. Switch modes live in the dashboard; all three work and FPS is acceptable.
2. Run the experiment matrix; check the table makes sense (e.g. temporal smoothing should cut false alarms).

---

## Phase 9 — Hardening and documentation

**Build**
- 30–60 minute live or looped stability run; fix leaks or slowdowns.
- README: setup, model download, running the app, evaluation, training, configuration reference, architecture diagram (Mermaid), limitations (camera angle, night/IR, dataset licences and bias, not a safety-certified system), future work (NVIDIA Jetson / Raspberry Pi deployment, IR camera, Drive&Act/AIDE/3MDAD).
- `docs/results.md` collecting all metric tables and confusion matrices.
- `docs/demo_script.md`: a 5-minute demonstration sequence.
- Remove dead code, tidy config, all tests pass.

**Manual test:** follow the README on a clean clone in a new venv and confirm everything runs.
