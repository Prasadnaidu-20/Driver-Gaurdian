"""Load config/default.yaml into typed settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class SectionSettings(BaseModel):
    """Permissive section for components not built yet; later phases replace it with a typed model."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = False


class ServerSettings(BaseModel):
    host: str
    port: int


class StreamSettings(BaseModel):
    frame_width: int
    frame_height: int
    jpeg_quality: float
    perf_window_frames: int


class FaceSettings(BaseModel):
    enabled: bool
    model_path: str
    model_url: str
    num_faces: int
    min_detection_confidence: float
    min_presence_confidence: float
    min_tracking_confidence: float
    timestamp_reset_ms: float
    send_landmarks: bool
    crop_size: int
    crop_margin: float

    def resolved_model_path(self) -> Path:
        """Model path as an absolute path (relative paths are relative to the project root)."""
        path = Path(self.model_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


class UISettings(BaseModel):
    chart_window_s: float


class CalibrationSettings(BaseModel):
    duration_s: float


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseModel):
    """Root settings object mirroring the top-level sections of the YAML file."""

    model_config = ConfigDict(extra="forbid")

    server: ServerSettings
    stream: StreamSettings
    face: FaceSettings
    ui: UISettings
    calibration: CalibrationSettings
    drowsiness: SectionSettings
    distraction: SectionSettings
    objects: SectionSettings
    emotion: SectionSettings
    risk: SectionSettings
    alerts: SectionSettings
    logging: LoggingSettings
    mode: Literal["rules", "ml", "hybrid"]


def load_settings(path: Path | None = None) -> Settings:
    """Read a YAML config file (default: config/default.yaml) and validate it into `Settings`."""
    config_path = path or DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the default settings, loaded once per process."""
    return load_settings()
