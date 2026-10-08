"""Common estimator interface and its output type.

Every drowsiness / distraction / emotion estimator (rules, ML, hybrid) implements
`Estimator` and returns a `TaskOutput`, so `FrameProcessor` and risk fusion do not care
which implementation runs. Inputs besides the geometric features (the face crop for
emotion, detected activities for distraction) arrive in an optional `FrameContext`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from app.config import LevelSettings
from app.pipeline.calibration import Baseline
from app.temporal.hysteresis import HysteresisStateMachine

UNKNOWN_LABEL = "unknown"  # no face in this frame: nothing to estimate


class TaskOutput(BaseModel):
    """Result of one estimator for one frame."""

    score: float = 0.0                                          # 0-1, higher = worse
    label: str = UNKNOWN_LABEL                                  # task-specific label
    confidence: float = 0.0                                     # 0-1
    probabilities: dict[str, float] = Field(default_factory=dict)  # per-label; empty for rule estimators
    reasons: list[str] = Field(default_factory=list)            # human-readable, most important first
    details: dict[str, float] = Field(default_factory=dict)     # numeric stats (PERCLOS, durations, counts)


class Activities(BaseModel):
    """Driver activities from object detection (Phase 4). Durations are of the current episode."""

    phone_use: bool = False
    drinking: bool = False
    phone_s: float = 0.0       # how long a phone has been near the driver
    drinking_s: float = 0.0    # how long a cup/bottle has been at the mouth
    reasons: list[str] = Field(default_factory=list)


@dataclass
class FrameContext:
    """Per-frame inputs besides the features. Fields are None when not available on this frame."""

    face_crop: np.ndarray | None = None    # crop_face output (RGB); only on frames where emotion should run
    activities: Activities | None = None   # latest activities; None when object detection is off


class Estimator(ABC):
    """Stateful per-stream estimator fed one frame at a time, in frame-timestamp order."""

    @abstractmethod
    def update(self, features: dict[str, float] | None, timestamp_ms: float,
               context: FrameContext | None = None) -> TaskOutput:
        """Consume one frame's features (None = no face) plus optional context; return the current estimate."""

    @abstractmethod
    def set_baseline(self, baseline: Baseline) -> None:
        """Use a new per-driver baseline from now on."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all temporal state (new stream, or a video seek)."""


def hysteresis_from(cfg: LevelSettings) -> HysteresisStateMachine:
    """Hysteresis state machine configured from an estimator's level settings."""
    return HysteresisStateMachine(cfg.levels, cfg.enter, cfg.exit, cfg.min_enter_s, cfg.min_exit_s)


def no_face_output() -> TaskOutput:
    """Output for a frame without a face: nothing can be estimated."""
    return TaskOutput(label=UNKNOWN_LABEL, reasons=["no face"])


def ramp(x: float, lo: float, hi: float) -> float:
    """0 at/below `lo`, 1 at/above `hi`, linear in between."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))
