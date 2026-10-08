# DriveGuardian — Progress

## Phase 0 — Project setup ✅ (2026-10-08)

**Built**
- Folder skeleton per CLAUDE.md (the repo root acts as `driveguardian/`). There are empty package markers under `app/` and `.gitkeep` files in the dirs that later phases fill.
- `.gitignore`, `requirements.txt`, `pytest.ini`, `README.md`.
- `config/default.yaml`: all 12 top-level sections. `server`, `stream`, `calibration`, `logging` and `mode` have real values. The other sections are `enabled: false` placeholders.
- `app/config.py`: pydantic models. `load_settings(path)` loads the YAML and `get_settings()` caches the default. Placeholder sections use a permissive `SectionSettings` that later phases replace with typed models.
- `app/main.py`: `GET /health` → `{"status": "ok"}`. `web/` is served as static files at `/`.
- `web/index.html` placeholder, plus empty `web/app.js` and `web/styles.css` stubs.

**Run**
```powershell
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload    # http://localhost:8000, /health
```

**Test**: `pytest -q`. There are 2 tests: the config loads with all sections, and the health and index endpoints respond.

**Deviations from spec**
- The venv uses **Python 3.12.8**, the only version installed (CLAUDE.md says 3.11). MediaPipe supports 3.12, so this should be fine.
- I added `httpx` to `requirements.txt`, because FastAPI's `TestClient` needs it.

**Known issues**
- pytest shows a Starlette deprecation warning recommending `httpx2` over `httpx` for `TestClient`. It doesn't affect anything, so I'm leaving it for now.
- The folder is not a git repository yet. Run `git init` before the first commit.
