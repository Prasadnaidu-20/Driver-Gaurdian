"""FastAPI application: health route and static dashboard."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.ws import router as ws_router
from app.config import PROJECT_ROOT, StreamSettings, get_settings

settings = get_settings()
logging.basicConfig(
    level=settings.logging.level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="DriveGuardian")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/api/config")
def client_config() -> StreamSettings:
    """Stream settings the browser needs (capture size, JPEG quality)."""
    return settings.stream


app.include_router(ws_router)


# Mounted last so it does not shadow API routes.
app.mount("/", StaticFiles(directory=PROJECT_ROOT / "web", html=True), name="web")
logger.info("DriveGuardian started (mode=%s)", settings.mode)
