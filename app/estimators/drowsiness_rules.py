"""Rule-based drowsiness: eye closure, PERCLOS, long closures (microsleep) and yawns.

Per frame:
* eye closed  = EAR < closed_ratio × baseline open-eye EAR, or mean eyeBlink > blink_threshold.
  If the head is pitched down more than `ear_ignore_pitch_down_deg` from neutral, only the
  blendshape is used, because looking down makes EAR drop with the eyes open.
* closure episodes shorter than `microsleep_s` are blinks; longer ones are long closures.
* PERCLOS     = time fraction of closed eyes over `perclos_window_s`.
* yawn        = MAR above the yawn threshold for longer than `yawn_min_s`, counted once per episode.

score = clamp(max(closure component, PERCLOS component) + yawn_weight × yawn component),
then a time-based EMA, then hysteresis → alert | slightly_drowsy | drowsy.
"""

from __future__ import annotations

from collections import deque

from app.config import DrowsinessSettings
from app.estimators.base import Estimator, TaskOutput, hysteresis_from, no_face_output, ramp
from app.pipeline.calibration import Baseline
from app.temporal.ema import EMA
from app.temporal.episodes import EpisodeTracker
from app.temporal.window import TimeWindow


class DrowsinessRules(Estimator):
    """Rule-based drowsiness estimator (one instance per stream)."""

    def __init__(self, cfg: DrowsinessSettings, baseline: Baseline) -> None:
        self.cfg = cfg
        self.baseline = baseline
        self.reset()

    def set_baseline(self, baseline: Baseline) -> None:
        self.baseline = baseline

    def reset(self) -> None:
        cfg = self.cfg
        self._closure = EpisodeTracker(cfg.max_gap_s)
        self._mouth = EpisodeTracker(cfg.max_gap_s)
        self._perclos = TimeWindow(cfg.perclos_window_s, cfg.max_dt_s)
        self._blinks: deque[tuple[float, float]] = deque()       # (end ts ms, duration s)
        self._microsleeps: deque[tuple[float, float]] = deque()  # (end ts ms, duration s)
        self._yawns: deque[float] = deque()                       # ts ms when each yawn was counted
        self._yawn_counted_for: float | None = None               # start ts of the mouth episode already counted
        self._ema = EMA(cfg.score_ema_tau_s)
        self._levels = hysteresis_from(cfg)
        self._last_ts: float | None = None
        self._last_output: TaskOutput | None = None

    # ---- thresholds (depend on the baseline) ----

    @property
    def ear_threshold(self) -> float:
        return self.cfg.closed_ratio * self.baseline.ear_open

    @property
    def yawn_threshold(self) -> float:
        return max(self.cfg.yawn_mar_threshold, self.baseline.mar_closed + self.cfg.yawn_mar_margin)

    def eye_closed(self, features: dict[str, float]) -> bool:
        """Per-frame eye-closed decision (see module docstring)."""
        blink = 0.5 * (features["eye_blink_left"] + features["eye_blink_right"])
        if blink > self.cfg.blink_threshold:
            return True
        head_down = features["pitch"] - self.baseline.pitch < -self.cfg.ear_ignore_pitch_down_deg
        return (not head_down) and features["ear_mean"] < self.ear_threshold

    # ---- per frame ----

    def update(self, features: dict[str, float] | None, timestamp_ms: float) -> TaskOutput:
        if self._last_ts is not None and timestamp_ms <= self._last_ts and self._last_output is not None:
            return self._last_output  # duplicate / out-of-order frame: nothing new
        self._last_ts = timestamp_ms
        cfg = self.cfg

        if features is None:
            self._record_closure_end(self._closure.update(None, timestamp_ms), timestamp_ms)
            self._mouth.update(None, timestamp_ms)
            self._last_output = no_face_output()
            return self._last_output

        # Eyes
        closed = self.eye_closed(features)
        self._record_closure_end(self._closure.update(closed, timestamp_ms), timestamp_ms)
        closed_s = self._closure.duration_s()
        self._perclos.add(timestamp_ms, 1.0 if closed else 0.0)
        perclos = self._perclos.time_mean(cfg.perclos_min_span_s)

        # Mouth / yawns
        self._mouth.update(features["mar"] > self.yawn_threshold, timestamp_ms)
        if (self._mouth.active and self._mouth.duration_s() >= cfg.yawn_min_s
                and self._yawn_counted_for != self._mouth.start_ms):
            self._yawns.append(timestamp_ms)
            self._yawn_counted_for = self._mouth.start_ms
        self._prune(timestamp_ms)
        yawns = len(self._yawns)

        # Score
        closure_c = ramp(closed_s, cfg.microsleep_s, cfg.closure_full_s)
        perclos_c = ramp(perclos, cfg.perclos_low, cfg.perclos_high)
        yawn_c = min(1.0, yawns / cfg.yawn_count_full) if cfg.yawn_count_full > 0 else 0.0
        raw = min(1.0, max(closure_c, perclos_c) + cfg.yawn_weight * yawn_c)
        score = self._ema.update(raw, timestamp_ms)
        label = self._levels.update(score, timestamp_ms)

        reasons: list[str] = []
        if closed_s >= cfg.microsleep_s:
            reasons.append(f"eyes closed {closed_s:.1f} s")
        if perclos_c > 0:
            reasons.append(f"PERCLOS {perclos:.0%}")
        if yawns:
            reasons.append(f"{yawns} yawn{'s' if yawns != 1 else ''} in last {cfg.yawn_window_s / 60:.0f} min")
        if self._microsleeps and closed_s < cfg.microsleep_s:
            n = len(self._microsleeps)
            reasons.append(f"{n} long closure{'s' if n != 1 else ''} in last {cfg.blink_window_s:.0f} s")

        span = self._perclos.span_s()
        blink_durs = [d for _, d in self._blinks]
        self._last_output = TaskOutput(
            score=score,
            label=label,
            confidence=self._perclos.covered_s() / span if span > 0 else 1.0,
            reasons=reasons,
            details={
                "raw_score": raw,
                "eye_closed": 1.0 if closed else 0.0,
                "closed_s": closed_s,
                "perclos": perclos,
                "blinks": float(len(self._blinks)),
                "last_blink_ms": blink_durs[-1] * 1000.0 if blink_durs else 0.0,
                "mean_blink_ms": 1000.0 * sum(blink_durs) / len(blink_durs) if blink_durs else 0.0,
                "long_closures": float(len(self._microsleeps)),
                "yawns": float(yawns),
                "mouth_open_s": self._mouth.duration_s(),
                "ear_threshold": self.ear_threshold,
                "yawn_threshold": self.yawn_threshold,
            },
        )
        return self._last_output

    def _record_closure_end(self, duration_s: float | None, timestamp_ms: float) -> None:
        if duration_s is None:
            return
        target = self._blinks if duration_s < self.cfg.microsleep_s else self._microsleeps
        target.append((timestamp_ms, duration_s))

    def _prune(self, timestamp_ms: float) -> None:
        blink_cutoff = timestamp_ms - self.cfg.blink_window_s * 1000.0
        for events in (self._blinks, self._microsleeps):
            while events and events[0][0] < blink_cutoff:
                events.popleft()
        yawn_cutoff = timestamp_ms - self.cfg.yawn_window_s * 1000.0
        while self._yawns and self._yawns[0] < yawn_cutoff:
            self._yawns.popleft()
