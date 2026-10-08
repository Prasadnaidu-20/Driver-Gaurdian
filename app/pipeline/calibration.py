"""Per-driver calibration: a neutral baseline captured while the driver looks at the road.

The baseline also absorbs camera placement. If the camera sits to the side, the driver's
"looking at the road" yaw is not 0, and head pose is judged relative to the calibrated
yaw/pitch.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from app.config import CalibrationSettings

logger = logging.getLogger(__name__)

CalibrationState = Literal["uncalibrated", "calibrating", "calibrated"]

BASELINE_FIELDS = ("ear_open", "mar_closed", "yaw", "pitch", "gaze_h", "gaze_v")
# Baseline field -> feature key it is the median of.
_FEATURE_FOR = {
    "ear_open": "ear_mean", "mar_closed": "mar", "yaw": "yaw",
    "pitch": "pitch", "gaze_h": "gaze_h", "gaze_v": "gaze_v",
}


@dataclass(frozen=True)
class Baseline:
    """Neutral per-driver values. `calibrated=False` means config defaults."""

    ear_open: float
    mar_closed: float
    yaw: float
    pitch: float
    gaze_h: float
    gaze_v: float
    calibrated: bool = False

    @classmethod
    def defaults(cls, cfg: CalibrationSettings) -> "Baseline":
        return cls(**cfg.defaults.model_dump(), calibrated=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Baseline":
        """Build a calibrated baseline from a dict (e.g. sent back by the browser). Raises ValueError if invalid."""
        try:
            values = {k: float(data[k]) for k in BASELINE_FIELDS}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid baseline: {exc}") from exc
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("invalid baseline: non-finite value")
        return cls(**values, calibrated=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Calibrator:
    """Collects features for `duration_s` of frame time, then computes the median baseline."""

    def __init__(self, cfg: CalibrationSettings) -> None:
        self.cfg = cfg
        self.baseline = Baseline.defaults(cfg)
        self.state: CalibrationState = "uncalibrated"
        self.message = "uncalibrated: using default baseline"
        self._start_ts: float | None = None
        self._last_ts: float | None = None
        self._pending = False
        self._samples: list[dict[str, float]] = []
        self._frames = 0

    # ---- control ----

    def start(self) -> None:
        """Begin (or restart) calibration on the next frame."""
        self._pending = True
        self._start_ts = None
        self._last_ts = None
        self._samples = []
        self._frames = 0
        self.state = "calibrating"
        self.message = "Look at the road, eyes open, mouth closed"

    def restart_if_running(self) -> None:
        """Called on a timestamp jump (video seek): collected samples are no longer contiguous."""
        if self.state == "calibrating":
            self.start()

    def set_baseline(self, baseline: Baseline) -> None:
        self.baseline = baseline
        self._pending = False
        self.state = "calibrated" if baseline.calibrated else "uncalibrated"
        self.message = "calibrated" if baseline.calibrated else "uncalibrated: using default baseline"


    # ---- progress ----

    @property
    def elapsed_s(self) -> float:
        if self._start_ts is None or self._last_ts is None:
            return 0.0
        return max(0.0, (self._last_ts - self._start_ts) / 1000.0)

    @property
    def progress(self) -> float:
        if self.state != "calibrating":
            return 1.0 if self.state == "calibrated" else 0.0
        return min(1.0, self.elapsed_s / self.cfg.duration_s) if self.cfg.duration_s > 0 else 1.0

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.cfg.duration_s - self.elapsed_s) if self.state == "calibrating" else 0.0

    # ---- per frame ----

    def feed(self, features: dict[str, float] | None, timestamp_ms: float) -> Baseline | None:
        """Feed one frame while calibrating. Returns the new baseline when calibration succeeds."""
        if self.state != "calibrating":
            return None
        if self._pending or self._start_ts is None:
            self._pending = False
            self._start_ts = timestamp_ms
        self._last_ts = timestamp_ms
        self._frames += 1
        if features is not None:
            self._samples.append(features)
        if self.elapsed_s < self.cfg.duration_s:
            return None
        return self._finish()

    def _finish(self) -> Baseline | None:
        coverage = len(self._samples) / max(1, self._frames)
        error = None
        if coverage < self.cfg.min_valid_fraction:
            error = f"face visible {coverage:.0%} of the time"
        else:
            medians = {
                name: float(np.median([s[key] for s in self._samples]))
                for name, key in _FEATURE_FOR.items()
            }
            if medians["ear_open"] < self.cfg.min_open_ear:
                error = f"eyes looked closed (EAR {medians['ear_open']:.2f})"
            elif medians["mar_closed"] > self.cfg.max_closed_mar:
                error = f"mouth looked open (MAR {medians['mar_closed']:.2f})"
        self._samples = []
        if error is not None:
            # Keep the previous baseline (calibrated or defaults).
            self.state = "calibrated" if self.baseline.calibrated else "uncalibrated"
            self.message = f"Calibration failed: {error}"
            logger.info(self.message)
            return None
        self.baseline = Baseline(**medians, calibrated=True)
        self.state = "calibrated"
        self.message = "calibrated"
        logger.info("Calibration done: %s", self.baseline)
        return self.baseline
