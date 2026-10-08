"""Load config/default.yaml into typed settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

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


class BaselineDefaults(BaseModel):
    ear_open: float
    mar_closed: float
    yaw: float
    pitch: float
    gaze_h: float
    gaze_v: float


class CalibrationSettings(BaseModel):
    duration_s: float
    min_valid_fraction: float
    min_open_ear: float
    max_closed_mar: float
    defaults: BaselineDefaults


class LevelSettings(BaseModel):
    """Hysteresis levels shared by the rule estimators (lowest level first)."""

    levels: list[str]
    enter: list[float]
    exit: list[float]
    min_enter_s: float
    min_exit_s: float
    score_ema_tau_s: float

    @model_validator(mode="after")
    def _check_lengths(self) -> "LevelSettings":
        if not (len(self.levels) == len(self.enter) == len(self.exit)) or len(self.levels) < 2:
            raise ValueError("levels, enter and exit must have the same length (≥ 2)")
        return self


class DrowsinessSettings(LevelSettings):
    enabled: bool
    closed_ratio: float
    blink_threshold: float
    ear_ignore_pitch_down_deg: float
    max_gap_s: float
    max_dt_s: float
    microsleep_s: float
    closure_full_s: float
    perclos_window_s: float
    perclos_min_span_s: float
    perclos_low: float
    perclos_high: float
    blink_window_s: float
    yawn_mar_threshold: float
    yawn_mar_margin: float
    yawn_min_s: float
    yawn_window_s: float
    yawn_count_full: float
    yawn_weight: float


class DistractionSettings(LevelSettings):
    enabled: bool
    yaw_limit_deg: float
    pitch_down_limit_deg: float
    pitch_up_limit_deg: float
    use_gaze: bool
    gaze_h_max_offset: float
    gaze_v_max_offset: float
    closed_ratio: float
    max_gap_s: float
    max_dt_s: float
    glance_ignore_s: float
    glance_full_s: float
    window_s: float
    fraction_low: float
    fraction_high: float


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
    drowsiness: DrowsinessSettings
    distraction: DistractionSettings
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
