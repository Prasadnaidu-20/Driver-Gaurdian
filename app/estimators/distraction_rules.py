"""Rule-based distraction: eyes off road, from head pose and gaze relative to the calibrated neutral.

Per frame, "off road" means any of:
* |yaw − neutral| > yaw_limit_deg          (yaw + = driver turns to their left)
* pitch − neutral < −pitch_down_limit_deg  (pitch + = up, so negative = looking down)
* pitch − neutral > pitch_up_limit_deg
* gaze ratio further than its max offset from the neutral gaze (skipped while the eyes are closed)

Continuous component: 0 for off-road episodes shorter than `glance_ignore_s` (mirror
glances), ramping to 1 at `glance_full_s`. Fraction component: share of the last `window_s` spent in short glances
(off-road time while the episode is still shorter than `glance_ignore_s`), which catches
frequent glances that are each too short to count on their own.
score = max(both), then a time-based EMA, then hysteresis → attentive | looking_away | distracted.
"""

from __future__ import annotations

from app.config import DistractionSettings
from app.estimators.base import Estimator, TaskOutput, hysteresis_from, no_face_output, ramp
from app.pipeline.calibration import Baseline
from app.temporal.ema import EMA
from app.temporal.episodes import EpisodeTracker
from app.temporal.window import TimeWindow


class DistractionRules(Estimator):
    """Rule-based distraction estimator (one instance per stream)."""

    def __init__(self, cfg: DistractionSettings, baseline: Baseline) -> None:
        self.cfg = cfg
        self.baseline = baseline
        self.reset()

    def set_baseline(self, baseline: Baseline) -> None:
        self.baseline = baseline

    def reset(self) -> None:
        cfg = self.cfg
        self._offroad = EpisodeTracker(cfg.max_gap_s)
        self._window = TimeWindow(cfg.window_s, cfg.max_dt_s)
        self._ema = EMA(cfg.score_ema_tau_s)
        self._levels = hysteresis_from(cfg)
        self._last_ts: float | None = None
        self._last_output: TaskOutput | None = None

    def off_road(self, features: dict[str, float]) -> list[str]:
        """Why this frame is off road (empty list = on road). Each entry is a short cause."""
        cfg, b = self.cfg, self.baseline
        causes: list[str] = []
        d_yaw = features["yaw"] - b.yaw
        d_pitch = features["pitch"] - b.pitch
        if d_yaw > cfg.yaw_limit_deg:
            causes.append(f"head turned left, yaw {d_yaw:+.0f}°")
        elif d_yaw < -cfg.yaw_limit_deg:
            causes.append(f"head turned right, yaw {d_yaw:+.0f}°")
        if d_pitch < -cfg.pitch_down_limit_deg:
            causes.append(f"looking down, pitch {d_pitch:+.0f}°")
        elif d_pitch > cfg.pitch_up_limit_deg:
            causes.append(f"looking up, pitch {d_pitch:+.0f}°")
        eyes_open = features["ear_mean"] >= cfg.closed_ratio * b.ear_open
        if cfg.use_gaze and eyes_open:
            if (abs(features["gaze_h"] - b.gaze_h) > cfg.gaze_h_max_offset
                    or abs(features["gaze_v"] - b.gaze_v) > cfg.gaze_v_max_offset):
                causes.append("gaze off road")
        return causes

    def update(self, features: dict[str, float] | None, timestamp_ms: float) -> TaskOutput:
        if self._last_ts is not None and timestamp_ms <= self._last_ts and self._last_output is not None:
            return self._last_output  # duplicate / out-of-order frame: nothing new
        self._last_ts = timestamp_ms
        cfg = self.cfg

        if features is None:
            self._offroad.update(None, timestamp_ms)
            self._last_output = no_face_output()
            return self._last_output

        causes = self.off_road(features)
        off = bool(causes)
        self._offroad.update(off, timestamp_ms)
        off_s = self._offroad.duration_s()
        # Only glance time feeds the fraction: long episodes are already scored by the
        # continuous component and would otherwise keep the score up long after the driver looks back.
        glancing = off and off_s < cfg.glance_ignore_s
        self._window.add(timestamp_ms, 1.0 if glancing else 0.0)
        fraction = self._window.time_mean(cfg.window_s)

        cont_c = ramp(off_s, cfg.glance_ignore_s, cfg.glance_full_s)
        frac_c = ramp(fraction, cfg.fraction_low, cfg.fraction_high)
        raw = max(cont_c, frac_c)
        score = self._ema.update(raw, timestamp_ms)
        label = self._levels.update(score, timestamp_ms)

        reasons: list[str] = []
        if off_s >= cfg.glance_ignore_s:
            detail = f" ({'; '.join(causes)})" if causes else ""
            reasons.append(f"eyes off road {off_s:.1f} s{detail}")
        if frac_c > 0:
            reasons.append(f"frequent glances: off road {fraction:.0%} of last {cfg.window_s:.0f} s")

        span = self._window.span_s()
        self._last_output = TaskOutput(
            score=score,
            label=label,
            confidence=self._window.covered_s() / span if span > 0 else 1.0,
            reasons=reasons,
            details={
                "raw_score": raw,
                "off_road": 1.0 if off else 0.0,
                "off_road_s": off_s,
                "glance_fraction": fraction,
                "rel_yaw": features["yaw"] - self.baseline.yaw,
                "rel_pitch": features["pitch"] - self.baseline.pitch,
            },
        )
        return self._last_output
