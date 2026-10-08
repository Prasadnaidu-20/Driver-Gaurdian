# DriveGuardian

Real-time driver monitoring web app: the browser streams webcam/video frames to a Python backend, which estimates drowsiness, distraction and emotion, then fuses them into a risk level and alerts.

## Setup (Windows, Python 3.11 or 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
uvicorn app.main:app --reload
```

Open http://localhost:8000 (health check: http://localhost:8000/health).
Click **Start webcam** or **Load video file**. Frames stream to the backend over `/ws/stream`, and the Performance card shows stream/backend FPS, client round-trip latency and backend processing latency.

## Test

```powershell
pytest -q
```

Configuration lives in `config/default.yaml`. Build progress is tracked in `docs/progress.md`.
